use crate::{
    now_ms, Assignment, Attestation, ControlKey, EncryptedAssignment, Error, Keys, Receipt, Result,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64URL, Engine};
use serde_json::Value;

pub struct Coordinator {
    base: reqwest::Url,
    client: reqwest::blocking::Client,
    worker_id: String,
    miner_id: String,
    keys: Keys,
    session_id: Option<String>,
}
impl Coordinator {
    pub fn new(base: &str, worker_id: String, miner_id: String, keys: Keys) -> Result<Self> {
        let base = reqwest::Url::parse(base).map_err(|_| Error::Protocol("coordinator_url"))?;
        if base.scheme() != "https" {
            return Err(Error::Protocol("coordinator_https_required"));
        }
        Ok(Self {
            base,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(45))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            worker_id,
            miner_id,
            keys,
            session_id: None,
        })
    }
    /// Local loopback-only constructor for offline/smoke harnesses. Production
    /// binaries must always use `Coordinator::new` which enforces HTTPS.
    pub fn new_loopback(
        base: &str,
        worker_id: String,
        miner_id: String,
        keys: Keys,
    ) -> Result<Self> {
        let base = reqwest::Url::parse(base).map_err(|_| Error::Protocol("coordinator_url"))?;
        if base.scheme() != "http" || base.host_str() != Some("127.0.0.1") {
            return Err(Error::Protocol("coordinator_https_required"));
        }
        Ok(Self {
            base,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(45))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            worker_id,
            miner_id,
            keys,
            session_id: None,
        })
    }
    #[cfg(test)]
    pub(crate) fn test_new(base: &str, worker_id: String, miner_id: String, keys: Keys) -> Result<Self> {
        Ok(Self {
            base: reqwest::Url::parse(base).map_err(|_| Error::Protocol("coordinator_url"))?,
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(45))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            worker_id,
            miner_id,
            keys,
            session_id: Some("session".into()),
        })
    }
    pub fn with_session(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }
    pub fn current_session(&self) -> Result<&str> {
        self.session_id
            .as_deref()
            .ok_or(Error::Protocol("runtime_session_required"))
    }
    fn signed(&self, action: &str, payload: Value) -> Result<Value> {
        let session_id = self
            .session_id
            .as_deref()
            .ok_or(Error::Protocol("runtime_session_required"))?;
        let body = serde_json::json!({"workerId":self.worker_id,"minerId":self.miner_id,"publicKey":self.keys.public(),"sessionId":session_id,"time":now_ms(),"action":action,"payload":payload});
        Ok(
            serde_json::json!({"body":body,"signature":self.keys.sign("attelier/api-miner-runtime-action/v2",&body)?}),
        )
    }
    fn signed_control(&self, control: &ControlKey, action: &str, payload: Value) -> Result<Value> {
        let body = serde_json::json!({"workerId":self.worker_id,"minerId":self.miner_id,"publicKey":control.public(),"time":now_ms(),"action":action,"payload":payload});
        Ok(
            serde_json::json!({"body":body,"signature":control.sign("attelier/api-miner-control-action/v2",&body)?}),
        )
    }
    fn post(&self, path: &str, action: &str, payload: Value) -> Result<Value> {
        let url = self
            .base
            .join(path)
            .map_err(|_| Error::Protocol("coordinator_path"))?;
        let response = self
            .client
            .post(url)
            .json(&self.signed(action, payload)?)
            .send()?;
        if !response.status().is_success() {
            return Err(Error::Provider(format!(
                "coordinator:{}",
                response.status().as_u16()
            )));
        }
        Ok(response.json()?)
    }
    fn post_control(
        &self,
        control: &ControlKey,
        path: &str,
        action: &str,
        payload: Value,
    ) -> Result<Value> {
        let url = self
            .base
            .join(path)
            .map_err(|_| Error::Protocol("coordinator_path"))?;
        let response = self
            .client
            .post(url)
            .json(&self.signed_control(control, action, payload)?)
            .send()?;
        if !response.status().is_success() {
            return Err(Error::Provider(format!(
                "coordinator:{}",
                response.status().as_u16()
            )));
        }
        Ok(response.json()?)
    }
    pub fn claim(&self) -> Result<Option<Assignment>> {
        self.claim_excluding(&[])
    }
    pub fn claim_excluding(&self, exclude_job_ids: &[String]) -> Result<Option<Assignment>> {
        let session_id = self
            .session_id
            .as_deref()
            .ok_or(Error::Protocol("runtime_session_required"))?;
        let payload = if exclude_job_ids.is_empty() {
            serde_json::json!({})
        } else {
            serde_json::json!({"excludeJobIds": exclude_job_ids})
        };
        let value = self.post("v1/api-miners/claim", "claim", payload)?;
        if value.get("state").and_then(Value::as_str) == Some("idle") {
            return Ok(None);
        };
        let envelope: EncryptedAssignment = serde_json::from_value(
            value
                .get("encryptedAssignment")
                .cloned()
                .ok_or(Error::Protocol("encrypted_assignment_missing"))?,
        )?;
        let assignment: Assignment =
            serde_json::from_value(self.keys.decrypt_assignment(&envelope, session_id)?)?;
        assignment.validate(&self.worker_id, &self.miner_id)?;
        if assignment.session_id != session_id {
            return Err(Error::Protocol("assignment_session_mismatch"));
        }
        Ok(Some(assignment))
    }

    /// Publish an offer with the attested process's current runtime key.
    /// It intentionally lives on the running worker path: runtime keys never
    /// persist to disk and a separate CLI process cannot impersonate a session.
    pub fn offer(&self, payload: Value) -> Result<Value> {
        self.post("v1/api-miners/offer", "offer", payload)
    }
    pub fn recover(&self, hint: &Assignment) -> Result<Value> {
        self.post(
            "v1/api-miners/recover",
            "recover",
            serde_json::json!({"jobId":hint.job_id,"attemptId":hint.attempt_id,"grantNonce":hint.grant_nonce}),
        )
    }
    pub(crate) fn decrypt_recovered_assignment(&self, value: &Value) -> Result<Assignment> {
        let session_id = self
            .session_id
            .as_deref()
            .ok_or(Error::Protocol("runtime_session_required"))?;
        let envelope: EncryptedAssignment = serde_json::from_value(
            value
                .get("encryptedAssignment")
                .cloned()
                .ok_or(Error::Protocol("recovery_assignment_missing"))?,
        )?;
        let assignment: Assignment =
            serde_json::from_value(self.keys.decrypt_assignment(&envelope, session_id)?)?;
        assignment.validate(&self.worker_id, &self.miner_id)?;
        if assignment.session_id != session_id {
            return Err(Error::Protocol("recovery_session_mismatch"));
        }
        Ok(assignment)
    }
    pub fn start(
        &self,
        a: &Assignment,
        provider_ref: &str,
        provider_response_hash: &str,
    ) -> Result<Option<String>> {
        if provider_response_hash.len() != 64
            || !provider_response_hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::Protocol("provider_response_hash_invalid"));
        }
        let value = self.post("v1/api-miners/start","start",serde_json::json!({"jobId":a.job_id,"attemptId":a.attempt_id,"grantNonce":a.grant_nonce,"providerRef":provider_ref,"providerResponseHash":provider_response_hash}))?;
        Ok(value
            .get("state")
            .and_then(Value::as_str)
            .map(str::to_owned))
    }
    pub fn unknown(&self, a: &Assignment, reason: &str) -> Result<()> {
        self.post("v1/api-miners/unknown","unknown",serde_json::json!({"jobId":a.job_id,"attemptId":a.attempt_id,"grantNonce":a.grant_nonce,"reason":reason}))?;
        Ok(())
    }
    pub fn fail(
        &self,
        a: &Assignment,
        provider_ref: Option<&str>,
        reason: &str,
        http_status: Option<u16>,
        provider_code: Option<&str>,
    ) -> Result<()> {
        if !matches!(
            reason,
            "provider_rejected" | "provider_failed" | "provider_cancelled"
        ) {
            return Err(Error::Protocol("provider_failure_reason_invalid"));
        }
        let mut payload = serde_json::json!({
            "jobId":a.job_id,
            "attemptId":a.attempt_id,
            "grantNonce":a.grant_nonce,
            "reason":reason,
        });
        if let Some(reference) = provider_ref {
            payload["providerRef"] = Value::String(reference.to_owned());
        }
        if let Some(status) = http_status {
            payload["httpStatus"] = serde_json::json!(status);
        }
        if let Some(code) = provider_code {
            if reason != "provider_rejected"
                || code.is_empty()
                || code.len() > 80
                || !code.as_bytes()[0].is_ascii_alphabetic()
                || !code
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
            {
                return Err(Error::Protocol("provider_failure_code_invalid"));
            }
            payload["providerCode"] = Value::String(code.to_owned());
        }
        self.post("v1/api-miners/fail", "fail", payload)?;
        Ok(())
    }
    pub fn complete(&self, receipt: &Receipt, media: &[u8]) -> Result<()> {
        self.post(
            "v1/api-miners/complete",
            "complete",
            serde_json::json!({"receipt":receipt,"mediaBase64":B64URL.encode(media)}),
        )?;
        Ok(())
    }
    pub fn challenge(&self, control: &ControlKey) -> Result<String> {
        let value = self.post_control(
            control,
            "v1/api-miners/challenge",
            "challenge",
            serde_json::json!({}),
        )?;
        value
            .get("challenge")
            .or_else(|| value.get("nonce"))
            .and_then(Value::as_str)
            .map(str::to_owned)
            .filter(|v| !v.is_empty())
            .ok_or(Error::Protocol("attestation_challenge_missing"))
    }
    pub fn attest(
        &self,
        control: &ControlKey,
        challenge: &str,
        report_data: &str,
        quote: &Value,
    ) -> Result<(Attestation, String)> {
        let raw_quote = quote
            .get("quote")
            .cloned()
            .ok_or(Error::Protocol("phala_quote_missing"))?;
        let event_log = quote
            .get("event_log")
            .or_else(|| quote.get("eventLog"))
            .cloned()
            .ok_or(Error::Protocol("phala_event_log_missing"))?;
        let value=self.post_control(control,"v1/api-miners/attest","attest",serde_json::json!({"challenge":challenge,"reportData":report_data,"quote":raw_quote,"eventLog":event_log,"runtimeSigningPublicKey":self.keys.public(),"runtimeEncryptionPublicKey":self.keys.encryption_public()}))?;
        let attestation: Attestation = serde_json::from_value(
            value
                .get("attestation")
                .cloned()
                .ok_or(Error::Protocol("attestation_missing"))?,
        )?;
        let session = value
            .get("sessionId")
            .and_then(Value::as_str)
            .filter(|v| !v.is_empty())
            .ok_or(Error::Protocol("attestation_session_missing"))?
            .to_owned();
        attestation.validate()?;
        Ok((attestation, session))
    }
}
