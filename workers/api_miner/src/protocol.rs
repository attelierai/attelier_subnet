use crate::{
    canonical, now_ms, sha256, verify_canonical, Error, Keys, Result, RECEIPT_SCHEMA,
};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Attestation {
    pub measurement: String,
    pub issued_at: i64,
    pub expires_at: i64,
    #[serde(default = "verified_status")]
    pub status: String,
}
fn verified_status() -> String {
    "verified".into()
}
impl Attestation {
    pub fn validate(&self) -> Result<()> {
        if self.measurement.len() != 64
            || self.status != "verified"
            || self.issued_at > now_ms()
            || self.expires_at <= now_ms()
        {
            Err(Error::Protocol("attestation_not_current"))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Assignment {
    pub schema: String,
    pub job_id: String,
    pub attempt_id: String,
    pub worker_id: String,
    pub miner_id: String,
    pub session_id: String,
    #[serde(default)]
    pub contract_id: String,
    pub provider: String,
    pub model: String,
    #[serde(default = "text_to_video")]
    pub modality: String,
    #[serde(default)]
    pub inputs: Vec<AssignmentInput>,
    #[serde(default)]
    pub input_hash: String,
    pub request_hash: String,
    #[serde(alias = "nonce")]
    pub grant_nonce: String,
    pub expires_at: i64,
    #[serde(alias = "quoteMicrousd")]
    pub bid_microusd: u64,
    pub request: Value,
}
impl Assignment {
    pub fn validate(&self, worker: &str, miner_id: &str) -> Result<()> {
        if self.schema != RECEIPT_SCHEMA
            || self.worker_id != worker
            || self.miner_id != miner_id
            || self.job_id.is_empty()
            || self.attempt_id.is_empty()
            || self.session_id.is_empty()
            || self.grant_nonce.is_empty()
            || self.request_hash.len() != 64
            || self.expires_at <= now_ms()
            || !["text-to-video", "image-to-video", "text-to-image", "image-to-image"]
                .contains(&self.modality.as_str())
            || ![
                "byteplus",
                "kling",
                "google-cloud",
                "fal",
                "atlascloud",
                "kie",
                "higgsfield",
            ]
            .contains(&self.provider.as_str())
        {
            return Err(Error::Protocol("invalid_assignment"));
        }
        if sha256(&canonical(&self.request)?) != self.request_hash {
            return Err(Error::Protocol("request_hash_mismatch"));
        }
        if matches!(self.modality.as_str(), "text-to-video" | "text-to-image")
            && !self.inputs.is_empty()
        {
            return Err(Error::Protocol("unexpected_assignment_input"));
        }
        if matches!(self.modality.as_str(), "image-to-video" | "image-to-image")
            && self.inputs.len() != 1
        {
            return Err(Error::Protocol("assignment_image_required"));
        }
        for input in &self.inputs {
            input.validate()?;
        }
        Ok(())
    }
}
fn text_to_video() -> String {
    "text-to-video".into()
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssignmentInput {
    pub id: String,
    #[serde(rename = "mimeType")]
    pub mime: String,
    pub sha256: String,
    pub bytes_base64: String,
}
impl AssignmentInput {
    fn validate(&self) -> Result<()> {
        if self.id.is_empty()
            || !matches!(
                self.mime.as_str(),
                "image/png" | "image/jpeg" | "image/webp"
            )
            || self.sha256.len() != 64
        {
            return Err(Error::Protocol("assignment_input_invalid"));
        }
        let bytes = B64
            .decode(&self.bytes_base64)
            .map_err(|_| Error::Protocol("assignment_input_invalid"))?;
        if bytes.is_empty() || bytes.len() > 10 * 1024 * 1024 || sha256(&bytes) != self.sha256 {
            return Err(Error::Protocol("assignment_input_invalid"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Receipt {
    pub schema: String,
    pub session_id: String,
    pub miner_id: String,
    pub worker_id: String,
    pub job_id: String,
    pub attempt_id: String,
    pub request_hash: String,
    pub provider: String,
    pub model: String,
    pub provider_ref: String,
    pub provider_response_hash: String,
    pub output_hash: String,
    pub bytes: usize,
    pub signature: String,
}
impl Receipt {
    fn payload(&self) -> Value {
        serde_json::json!({"schema":self.schema,"jobId":self.job_id,"attemptId":self.attempt_id,"minerId":self.miner_id,"workerId":self.worker_id,"sessionId":self.session_id,"requestHash":self.request_hash,"provider":self.provider,"model":self.model,"providerRef":self.provider_ref,"providerResponseHash":self.provider_response_hash,"outputHash":self.output_hash,"bytes":self.bytes})
    }
    pub fn signed(
        a: &Assignment,
        attestation: Attestation,
        reference: String,
        provider_response: &[u8],
        output: &[u8],
        keys: &Keys,
    ) -> Result<Self> {
        Self::signed_with_provider_hash(
            a,
            attestation,
            reference,
            sha256(provider_response),
            output,
            keys,
        )
    }
    pub fn signed_with_provider_hash(
        a: &Assignment,
        attestation: Attestation,
        reference: String,
        provider_response_hash: String,
        output: &[u8],
        keys: &Keys,
    ) -> Result<Self> {
        Self::signed_with_provider_hash_in_session(
            a,
            attestation,
            &a.session_id,
            reference,
            provider_response_hash,
            output,
            keys,
        )
    }
    /// A renewed attestation keeps the same runtime key but receives a fresh
    /// session. The receipt must bind that current session, not the session
    /// under which the assignment was originally claimed.
    pub fn signed_with_provider_hash_in_session(
        a: &Assignment,
        attestation: Attestation,
        session_id: &str,
        reference: String,
        provider_response_hash: String,
        output: &[u8],
        keys: &Keys,
    ) -> Result<Self> {
        attestation.validate()?;
        if session_id.is_empty() {
            return Err(Error::Protocol("runtime_session_required"));
        }
        if provider_response_hash.len() != 64
            || !provider_response_hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::Protocol("provider_response_hash_invalid"));
        }
        let mut receipt = Self {
            schema: RECEIPT_SCHEMA.into(),
            session_id: session_id.to_owned(),
            miner_id: a.miner_id.clone(),
            worker_id: a.worker_id.clone(),
            job_id: a.job_id.clone(),
            attempt_id: a.attempt_id.clone(),
            request_hash: a.request_hash.clone(),
            provider: a.provider.clone(),
            model: a.model.clone(),
            provider_ref: reference,
            provider_response_hash,
            output_hash: sha256(output),
            bytes: output.len(),
            signature: String::new(),
        };
        receipt.signature = keys.sign_canonical(&receipt.payload())?;
        Ok(receipt)
    }
    pub fn verify_with(&self, public_key: &str) -> Result<()> {
        verify_canonical(public_key, &self.payload(), &self.signature)
    }
}

/// Enough context is sealed in the miner's state to resume a paid request after
/// a restart.  In particular, `provider_response` is the exact submission
/// response used in the eventual receipt; we never submit a second job merely
/// because the worker process restarted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TerminalFailure {
    pub reason: String,
    #[serde(default)]
    pub http_status: Option<u16>,
    #[serde(default)]
    pub provider_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptedAttempt {
    pub assignment: Assignment,
    pub provider_ref: Option<String>,
    #[serde(default)]
    pub provider_response: String,
    #[serde(default)]
    pub provider_response_hash: Option<String>,
    pub state: String,
    #[serde(default)]
    pub terminal_failure: Option<TerminalFailure>,
}
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DurableState {
    pub session_id: Option<String>,
    pub accepted: BTreeMap<String, AcceptedAttempt>,
    pub paused: bool,
}
impl DurableState {
    pub fn load(path: &Path) -> Result<Self> {
        if path.exists() {
            Ok(serde_json::from_slice(&fs::read(path)?)?)
        } else {
            Ok(Self::default())
        }
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?
        };
        let parent = path
            .parent()
            .ok_or(Error::Protocol("state_parent_missing"))?;
        // The accepted provider reference must be durable before the process
        // can report work started.  This survives a power loss after rename.
        // A PID alone is not unique across a CVM reboot. A crash can leave its
        // old temp file behind and Linux may reuse the PID before the next
        // save, so retain create_new but add cryptographically random entropy.
        let (tmp, mut file) = (0..16)
            .find_map(|_| {
                let mut suffix = [0u8; 16];
                OsRng.fill_bytes(&mut suffix);
                let candidate = path.with_extension(format!(
                    "{}.{}.tmp",
                    std::process::id(),
                    hex::encode(suffix)
                ));
                match fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&candidate)
                {
                    Ok(file) => Some(Ok((candidate, file))),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => None,
                    Err(error) => Some(Err(Error::Io(error))),
                }
            })
            .transpose()?
            .ok_or(Error::Protocol("state_tmp_name_exhausted"))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&tmp, fs::Permissions::from_mode(0o600))?;
        }
        use std::io::Write;
        file.write_all(&serde_json::to_vec(self)?)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp, path)?;
        fs::File::open(parent)?.sync_all()?;
        Ok(())
    }
}

/// Pause is a separate control file rather than a state.json rewrite.  The
/// running worker owns state.json while it is recording a paid attempt; a
/// lifecycle command must never overwrite that newer durable state.
pub fn pause_requested(state_path: &Path, state: &DurableState) -> bool {
    let Some(parent) = state_path.parent() else {
        return state.paused;
    };
    if parent.join("pause.flag").exists() {
        true
    } else if parent.join("resume.flag").exists() {
        false
    } else {
        // Compatibility with a state volume written by the initial worker.
        state.paused
    }
}
