//! Fail-closed worker protocol for API-backed Attelier miners.
//!
//! The gateway owns routing. This crate accepts only a fresh, one-use grant and
//! persists it before an upstream request, so an uncertain paid submission is
//! reconciled rather than replayed.

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use base64::{
    engine::general_purpose::{STANDARD as B64, URL_SAFE_NO_PAD as B64URL},
    Engine,
};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use hmac::{Hmac, Mac};
use rand::rngs::OsRng;
use rand::RngCore;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};

pub const ASSIGNMENT_SCHEMA: &str = "attelier/api-miner-assignment/v1";
pub const RECEIPT_SCHEMA: &str = "attelier/api-miner/v1";
/// Must stay at or below the gateway's accepted `mediaBase64` body boundary.
/// 100 MiB raw media expands to roughly 133 MiB when base64 encoded.
pub const MAX_MEDIA_BYTES: usize = 100 * 1024 * 1024;

#[derive(Debug, Error)]
pub enum Error {
    #[error("protocol: {0}")]
    Protocol(&'static str),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("provider: {0}")]
    Provider(String),
    /// The upstream has authoritatively rejected or terminated this exact
    /// request. This is distinct from an HTTP/transport ambiguity: it can be
    /// reported as a non-rewardable failure, never retried as paid work.
    #[error("provider terminal: {reason}")]
    ProviderTerminal {
        reason: &'static str,
        http_status: Option<u16>,
        provider_code: Option<String>,
    },
}
pub type Result<T> = std::result::Result<T, Error>;
pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}
pub fn sha256(data: &[u8]) -> String {
    hex::encode(Sha256::digest(data))
}

/// The accepted protocol subset has objects/arrays/strings/booleans/integers.
/// Floats are forbidden so the bytes being signed cannot vary by runtime.
pub fn canonical(value: &Value) -> Result<Vec<u8>> {
    fn sort(v: &Value) -> Result<Value> {
        Ok(match v {
            Value::Object(map) => {
                let mut ordered = BTreeMap::new();
                for (key, item) in map {
                    ordered.insert(key.clone(), sort(item)?);
                }
                Value::Object(ordered.into_iter().collect::<Map<_, _>>())
            }
            Value::Array(items) => Value::Array(items.iter().map(sort).collect::<Result<_>>()?),
            Value::Number(n) if n.is_f64() => {
                return Err(Error::Protocol("floating_point_not_allowed"))
            }
            _ => v.clone(),
        })
    }
    Ok(serde_json::to_vec(&sort(value)?)?)
}
fn domain(name: &str, value: &Value) -> Result<Vec<u8>> {
    let mut out = name.as_bytes().to_vec();
    out.push(0);
    out.extend(canonical(value)?);
    Ok(out)
}

#[derive(Clone)]
pub struct ControlKey {
    signing: SigningKey,
}
impl ControlKey {
    pub fn load_or_create(path: &Path) -> Result<Self> {
        if let Ok(raw) = fs::read(path) {
            let array: [u8; 32] = raw
                .try_into()
                .map_err(|_| Error::Protocol("invalid_control_key"))?;
            return Ok(Self {
                signing: SigningKey::from_bytes(&array),
            });
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let signing = SigningKey::generate(&mut OsRng);
        fs::write(path, signing.to_bytes())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
        }
        Ok(Self { signing })
    }
    pub fn public(&self) -> String {
        ed25519_public(&self.signing)
    }
    pub fn sign(&self, name: &str, value: &Value) -> Result<String> {
        Ok(B64URL.encode(self.signing.sign(&domain(name, value)?).to_bytes()))
    }
}

fn ed25519_public(signing: &SigningKey) -> String {
    let mut der = vec![
        0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00,
    ];
    der.extend(signing.verifying_key().to_bytes());
    B64URL.encode(der)
}

/// Per-process receipt/action identity. Never call `load_or_create` in a
/// production path: its private material exists only in this process memory.
#[derive(Clone)]
pub struct Keys {
    signing: SigningKey,
    encryption: StaticSecret,
}
impl Keys {
    pub fn fresh() -> Self {
        Self {
            signing: SigningKey::generate(&mut OsRng),
            encryption: StaticSecret::random_from_rng(OsRng),
        }
    }
    #[cfg(test)]
    pub fn load_or_create(path: &Path) -> Result<Self> {
        if let Ok(raw) = fs::read(path) {
            let array: [u8; 32] = raw
                .try_into()
                .map_err(|_| Error::Protocol("invalid_worker_key"))?;
            let encryption_path = path.with_extension("x25519");
            let raw_encryption = fs::read(&encryption_path)?;
            let enc: [u8; 32] = raw_encryption
                .try_into()
                .map_err(|_| Error::Protocol("invalid_worker_encryption_key"))?;
            return Ok(Self {
                signing: SigningKey::from_bytes(&array),
                encryption: StaticSecret::from(enc),
            });
        }
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let key = SigningKey::generate(&mut OsRng);
        let encryption = StaticSecret::random_from_rng(OsRng);
        fs::write(path, key.to_bytes())?;
        let encryption_path = path.with_extension("x25519");
        fs::write(&encryption_path, encryption.to_bytes())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
            fs::set_permissions(encryption_path, fs::Permissions::from_mode(0o600))?;
        }
        Ok(Self {
            signing: key,
            encryption,
        })
    }
    /// Ed25519 SubjectPublicKeyInfo DER, base64url without padding. This is the
    /// gateway admission format; raw 32-byte public keys are not accepted.
    pub fn public(&self) -> String {
        ed25519_public(&self.signing)
    }
    pub fn encryption_public(&self) -> String {
        let mut der = vec![
            0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x6e, 0x03, 0x21, 0x00,
        ];
        der.extend(X25519PublicKey::from(&self.encryption).as_bytes());
        B64URL.encode(der)
    }
    pub fn sign(&self, name: &str, value: &Value) -> Result<String> {
        Ok(B64URL.encode(self.signing.sign(&domain(name, value)?).to_bytes()))
    }
    pub fn sign_canonical(&self, value: &Value) -> Result<String> {
        Ok(B64URL.encode(self.signing.sign(&canonical(value)?).to_bytes()))
    }
}

/// An encrypted assignment is the only accepted transport for request bodies
/// and provider-upload references. The coordinator supplies an ephemeral X25519
/// key and AES-GCM ciphertext, all bound to the attested session ID.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EncryptedAssignment {
    pub ephemeral_public_key: String,
    pub nonce: String,
    pub ciphertext: String,
}
impl Keys {
    pub fn decrypt_assignment(
        &self,
        envelope: &EncryptedAssignment,
        session_id: &str,
    ) -> Result<Value> {
        let der = B64URL
            .decode(&envelope.ephemeral_public_key)
            .map_err(|_| Error::Protocol("invalid_assignment_ephemeral_key"))?;
        if der.len() != 44
            || der[..12]
                != [
                    0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x6e, 0x03, 0x21, 0x00,
                ]
        {
            return Err(Error::Protocol("invalid_assignment_ephemeral_key"));
        }
        let peer = X25519PublicKey::from(
            <[u8; 32]>::try_from(&der[12..])
                .map_err(|_| Error::Protocol("invalid_assignment_ephemeral_key"))?,
        );
        let shared = self.encryption.diffie_hellman(&peer);
        let key = Sha256::digest([shared.as_bytes().as_slice(), session_id.as_bytes()].concat());
        let cipher =
            Aes256Gcm::new_from_slice(&key).map_err(|_| Error::Protocol("assignment_cipher"))?;
        let nonce = B64URL
            .decode(&envelope.nonce)
            .map_err(|_| Error::Protocol("invalid_assignment_nonce"))?;
        if nonce.len() != 12 {
            return Err(Error::Protocol("invalid_assignment_nonce"));
        }
        let bytes = cipher
            .decrypt(
                Nonce::from_slice(&nonce),
                B64URL
                    .decode(&envelope.ciphertext)
                    .map_err(|_| Error::Protocol("invalid_assignment_ciphertext"))?
                    .as_ref(),
            )
            .map_err(|_| Error::Protocol("assignment_decrypt_failed"))?;
        Ok(serde_json::from_slice(&bytes)?)
    }
}
pub fn verify(public: &str, name: &str, value: &Value, signature: &str) -> Result<()> {
    let der = B64URL
        .decode(public)
        .map_err(|_| Error::Protocol("invalid_public_key"))?;
    if der.len() != 44
        || der[..12]
            != [
                0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00,
            ]
    {
        return Err(Error::Protocol("invalid_public_key"));
    }
    let array: [u8; 32] = der[12..]
        .try_into()
        .map_err(|_| Error::Protocol("invalid_public_key"))?;
    let key =
        VerifyingKey::from_bytes(&array).map_err(|_| Error::Protocol("invalid_public_key"))?;
    let raw = B64URL
        .decode(signature)
        .map_err(|_| Error::Protocol("invalid_signature"))?;
    let sig = ed25519_dalek::Signature::from_slice(&raw)
        .map_err(|_| Error::Protocol("invalid_signature"))?;
    key.verify_strict(&domain(name, value)?, &sig)
        .map_err(|_| Error::Protocol("bad_signature"))
}
pub fn verify_canonical(public: &str, value: &Value, signature: &str) -> Result<()> {
    let der = B64URL
        .decode(public)
        .map_err(|_| Error::Protocol("invalid_public_key"))?;
    if der.len() != 44
        || der[..12]
            != [
                0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00,
            ]
    {
        return Err(Error::Protocol("invalid_public_key"));
    }
    let key = VerifyingKey::from_bytes(
        &der[12..]
            .try_into()
            .map_err(|_| Error::Protocol("invalid_public_key"))?,
    )
    .map_err(|_| Error::Protocol("invalid_public_key"))?;
    let sig = ed25519_dalek::Signature::from_slice(
        &B64URL
            .decode(signature)
            .map_err(|_| Error::Protocol("invalid_signature"))?,
    )
    .map_err(|_| Error::Protocol("invalid_signature"))?;
    key.verify_strict(&canonical(value)?, &sig)
        .map_err(|_| Error::Protocol("bad_signature"))
}

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
            || !["text-to-video", "image-to-video"].contains(&self.modality.as_str())
            || !["byteplus", "kling", "google-cloud", "fal"].contains(&self.provider.as_str())
        {
            return Err(Error::Protocol("invalid_assignment"));
        }
        if sha256(&canonical(&self.request)?) != self.request_hash {
            return Err(Error::Protocol("request_hash_mismatch"));
        }
        if self.modality == "text-to-video" && !self.inputs.is_empty() {
            return Err(Error::Protocol("unexpected_assignment_input"));
        }
        if self.modality == "image-to-video" && self.inputs.len() != 1 {
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
pub struct ProviderSubmission {
    pub reference: String,
    pub response: Vec<u8>,
}
pub trait Provider {
    fn submit(&self, assignment: &Assignment) -> Result<ProviderSubmission>;
    fn collect(&self, assignment: &Assignment, submission: &ProviderSubmission) -> Result<Vec<u8>>;
}

/// Direct adapters expose no caller-controlled base URL.  Contracts must map a
/// model to one approved provider endpoint before a job can be assigned.
pub struct DirectProvider {
    configured_provider: String,
    client: reqwest::blocking::Client,
    key: zeroize::Zeroizing<String>,
    kling_access_key: Option<zeroize::Zeroizing<String>>,
    kling_secret_key: Option<zeroize::Zeroizing<String>>,
    // The signed token is sensitive too. Keep it only inside the process and
    // refresh before expiry; it is never persisted with assignment state.
    kling_token: Mutex<Option<(i64, zeroize::Zeroizing<String>)>>,
    google_project: Option<String>,
    google_location: Option<String>,
    google_output_bucket: Option<String>,
    #[cfg(test)]
    test_origin: Option<reqwest::Url>,
}
impl DirectProvider {
    /// Queue status/result routes are addressed to a fal app, never a nested
    /// model endpoint. Keep the app mapping explicit so adding a model needs
    /// review rather than a lossy path heuristic.
    fn fal_app(model: &str) -> Result<&'static str> {
        match model {
            "bytedance/seedance-2.5/text-to-video" => Ok("bytedance/seedance-2.5"),
            "bytedance/seedance-2.0/fast/image-to-video" => Ok("bytedance/seedance-2.0"),
            "fal-ai/kling-video/v3/turbo/standard/text-to-video" => Ok("fal-ai/kling-video"),
            "fal-ai/veo3.1"
            | "fal-ai/veo3.1/fast"
            | "fal-ai/veo3.1/image-to-video"
            | "fal-ai/veo3.1/fast/image-to-video" => Ok("fal-ai/veo3.1"),
            _ => Err(Error::Protocol("provider_model_not_approved")),
        }
    }
    fn fal_status_url(model: &str, reference: &str) -> Result<reqwest::Url> {
        reqwest::Url::parse(&format!(
            "https://queue.fal.run/{}/requests/{reference}/status",
            Self::fal_app(model)?
        ))
        .map_err(|_| Error::Protocol("provider_reference_invalid"))
    }
    fn fal_result_url(model: &str, reference: &str) -> Result<reqwest::Url> {
        reqwest::Url::parse(&format!(
            "https://queue.fal.run/{}/requests/{reference}",
            Self::fal_app(model)?
        ))
        .map_err(|_| Error::Protocol("provider_reference_invalid"))
    }
    fn approved_model(provider: &str, model: &str) -> bool {
        match provider {
            "byteplus" => matches!(
                model,
                "dreamina-seedance-2-5-260628"
                    | "dreamina-seedance-2-0-260128"
                    | "dreamina-seedance-2-0-mini-260615"
                    | "dreamina-seedance-2-0-fast-260128"
            ),
            "kling" => model == "kling-v2-6",
            "google-cloud" => matches!(model, "veo-3.1-generate-001" | "veo-3.1-fast-generate-001"),
            // These are the two existing Attelier remote-provider definitions
            // whose text-only schemas are the same. Adding a model is a
            // release/review change, never a caller-selected endpoint or body.
            "fal" => matches!(
                model,
                "bytedance/seedance-2.5/text-to-video"
                    | "fal-ai/kling-video/v3/turbo/standard/text-to-video"
                    | "bytedance/seedance-2.0/fast/image-to-video"
                    | "fal-ai/veo3.1"
                    | "fal-ai/veo3.1/fast"
                    | "fal-ai/veo3.1/image-to-video"
                    | "fal-ai/veo3.1/fast/image-to-video"
            ),
            _ => false,
        }
    }
    pub fn from_env(provider: &str) -> Result<Self> {
        let variable = match provider {
            "byteplus" => "BYTEPLUS_API_KEY",
            "kling" => "",
            "google-cloud" => "GOOGLE_CLOUD_ACCESS_TOKEN",
            "fal" => "FAL_KEY",
            _ => return Err(Error::Protocol("unsupported_provider")),
        };
        let key = if provider == "kling" {
            String::new()
        } else {
            std::env::var(variable).map_err(|_| Error::Protocol("provider_key_unavailable"))?
        };
        if provider != "kling" && key.trim().is_empty() {
            return Err(Error::Protocol("provider_key_unavailable"));
        }
        let kling_access_key = if provider == "kling" {
            Some(zeroize::Zeroizing::new(
                std::env::var("KLING_ACCESS_KEY")
                    .map_err(|_| Error::Protocol("kling_access_key_unavailable"))?,
            ))
        } else {
            None
        };
        let kling_secret_key = if provider == "kling" {
            Some(zeroize::Zeroizing::new(
                std::env::var("KLING_SECRET_KEY")
                    .map_err(|_| Error::Protocol("kling_secret_key_unavailable"))?,
            ))
        } else {
            None
        };
        if kling_access_key
            .as_deref()
            .is_some_and(|key| key.trim().is_empty())
            || kling_secret_key
                .as_deref()
                .is_some_and(|key| key.trim().is_empty())
        {
            return Err(Error::Protocol("kling_credentials_unavailable"));
        }
        let project = std::env::var("GOOGLE_CLOUD_PROJECT").ok();
        let location = std::env::var("GOOGLE_CLOUD_LOCATION").ok();
        let bucket = std::env::var("GOOGLE_CLOUD_OUTPUT_BUCKET").ok();
        let safe = |v: &str| {
            v.chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
                && v.len() >= 3
        };
        if provider == "google-cloud"
            && (!project.as_deref().is_some_and(safe)
                || !location.as_deref().is_some_and(safe)
                || !bucket.as_deref().is_some_and(safe))
        {
            return Err(Error::Protocol("google_cloud_project_location_or_bucket"));
        }
        Ok(Self {
            configured_provider: provider.into(),
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(45))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            key: zeroize::Zeroizing::new(key),
            kling_access_key,
            kling_secret_key,
            kling_token: Mutex::new(None),
            google_project: project,
            google_location: location,
            google_output_bucket: bucket,
            #[cfg(test)]
            test_origin: None,
        })
    }
    #[cfg(test)]
    fn fixture(origin: &str, configured_provider: &str) -> Self {
        Self {
            configured_provider: configured_provider.into(),
            client: reqwest::blocking::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap(),
            key: zeroize::Zeroizing::new("fixture-key".into()),
            kling_access_key: Some(zeroize::Zeroizing::new("fixture-access".into())),
            kling_secret_key: Some(zeroize::Zeroizing::new("fixture-secret".into())),
            kling_token: Mutex::new(None),
            google_project: Some("fixture-project".into()),
            google_location: Some("us-central1".into()),
            google_output_bucket: Some("fixture-bucket".into()),
            test_origin: Some(reqwest::Url::parse(origin).unwrap()),
        }
    }
    fn authorization(&self, provider: &str) -> Result<String> {
        if provider != self.configured_provider {
            return Err(Error::Protocol("configured_provider_mismatch"));
        }
        if provider == "fal" {
            return Ok(format!("Key {}", self.key.as_str()));
        }
        if provider != "kling" {
            return Ok(format!("Bearer {}", self.key.as_str()));
        }
        // Kling v2.6 uses the legacy API design: generate a short-lived HS256
        // JWT from the sealed AccessKey/SecretKey on every upstream call.
        let access = self
            .kling_access_key
            .as_deref()
            .ok_or(Error::Protocol("kling_access_key_unavailable"))?;
        let secret = self
            .kling_secret_key
            .as_deref()
            .ok_or(Error::Protocol("kling_secret_key_unavailable"))?;
        let now = now_ms() / 1_000;
        let mut cached = self
            .kling_token
            .lock()
            .map_err(|_| Error::Protocol("kling_token_lock"))?;
        if let Some((expires_at, token)) = cached.as_ref() {
            if *expires_at > now + 60 {
                return Ok(format!("Bearer {}", token.as_str()));
            }
        }
        let header = B64URL.encode(br#"{"alg":"HS256","typ":"JWT"}"#);
        let payload = B64URL.encode(serde_json::to_vec(
            &serde_json::json!({"iss":access,"exp":now + 1_800,"nbf":now - 5}),
        )?);
        let signed = format!("{header}.{payload}");
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(secret.as_bytes())
            .map_err(|_| Error::Protocol("kling_secret_key_invalid"))?;
        mac.update(signed.as_bytes());
        let token = zeroize::Zeroizing::new(format!(
            "{signed}.{}",
            B64URL.encode(mac.finalize().into_bytes())
        ));
        *cached = Some((now + 1_800, token));
        Ok(format!(
            "Bearer {}",
            cached.as_ref().expect("just set").1.as_str()
        ))
    }
    fn endpoint(&self, provider: &str, model: &str, modality: &str) -> Result<reqwest::Url> {
        if !Self::approved_model(provider, model) {
            return Err(Error::Protocol("provider_model_not_approved"));
        }
        #[cfg(test)]
        if let Some(origin) = &self.test_origin {
            return origin
                .join(&format!("upstream/{provider}/submit"))
                .map_err(|_| Error::Protocol("provider_endpoint"));
        }
        let endpoint=match provider {
   "byteplus"=>"https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks",
   "kling"=>if modality=="image-to-video"{"https://api-singapore.klingai.com/v1/videos/image2video"}else{"https://api-singapore.klingai.com/v1/videos/text2video"},
   "google-cloud"=>return reqwest::Url::parse(&format!("https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:predictLongRunning",self.google_location.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,self.google_project.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,self.google_location.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,model)).map_err(|_|Error::Protocol("provider_endpoint")),
   "fal"=>{
       if model.contains("..") || !model.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b==b'/' || b==b'.' || b==b'_' || b==b'-') { return Err(Error::Protocol("provider_model_invalid")); }
       return reqwest::Url::parse(&format!("https://queue.fal.run/{model}")).map_err(|_|Error::Protocol("provider_endpoint"));
   }
   _=>return Err(Error::Protocol("unsupported_provider")),
  };
        reqwest::Url::parse(endpoint).map_err(|_| Error::Protocol("provider_endpoint"))
    }
    fn reference(value: &Value) -> Option<String> {
        ["/id", "/task_id", "/data/task_id", "/name", "/request_id"]
            .iter()
            .find_map(|path| {
                value
                    .pointer(path)
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
    }
    fn status_url(&self, a: &Assignment, reference: &str) -> Result<reqwest::Url> {
        let encoded: String = reference
            .bytes()
            .map(|b| {
                if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'/') {
                    (b as char).to_string()
                } else {
                    format!("%{b:02X}")
                }
            })
            .collect();
        #[cfg(test)]
        if let Some(origin) = &self.test_origin {
            return origin
                .join(&format!("upstream/{}/status/{encoded}", a.provider))
                .map_err(|_| Error::Protocol("provider_reference_invalid"));
        }
        let endpoint=match a.provider.as_str(){
     "byteplus"=>format!("https://ark.ap-southeast.bytepluses.com/api/v3/contents/generations/tasks/{encoded}"),
     "kling"=>format!("https://api-singapore.klingai.com/v1/videos/{}/{encoded}",if a.modality=="image-to-video"{"image2video"}else{"text2video"}),
     // Vertex Veo is polled through the documented model operation method,
     // rather than treating an operation name as a generic REST resource.
     "google-cloud"=>format!("https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:fetchPredictOperation",self.google_location.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,self.google_project.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,self.google_location.as_deref().ok_or(Error::Protocol("google_cloud_project_or_location"))?,a.model),
     "fal"=>return Self::fal_status_url(&a.model,&encoded),
     _=>return Err(Error::Protocol("unsupported_provider")),
   };
        reqwest::Url::parse(&endpoint).map_err(|_| Error::Protocol("provider_reference_invalid"))
    }
    fn payload(&self, a: &Assignment) -> Result<Value> {
        let object = a
            .request
            .as_object()
            .ok_or(Error::Protocol("provider_request_invalid"))?;
        if object.keys().any(|key| {
            !matches!(key.as_str(), "prompt" | "duration" | "aspectRatio")
                && !(key == "imageId" && a.modality == "image-to-video")
        }) {
            return Err(Error::Protocol("provider_request_unsupported"));
        }
        let prompt = object
            .get("prompt")
            .and_then(Value::as_str)
            .filter(|v| !v.trim().is_empty() && v.len() <= 8_000)
            .ok_or(Error::Protocol("provider_prompt_invalid"))?
            .trim();
        let duration = object
            .get("duration")
            .and_then(Value::as_u64)
            .filter(|v| *v >= 1 && *v <= 15)
            .ok_or(Error::Protocol("provider_duration_invalid"))?;
        let aspect = object
            .get("aspectRatio")
            .and_then(Value::as_str)
            .filter(|v| matches!(*v, "16:9" | "9:16"));
        let image = || -> Result<&AssignmentInput> {
            a.inputs
                .first()
                .ok_or(Error::Protocol("assignment_image_required"))
        };
        if a.modality == "image-to-video" {
            if let Some(image_id) = object.get("imageId") {
                if image_id.as_str() != a.inputs.first().map(|input| input.id.as_str()) {
                    return Err(Error::Protocol("assignment_image_id_mismatch"));
                }
            }
        }
        let data_uri =
            |input: &AssignmentInput| format!("data:{};base64,{}", input.mime, input.bytes_base64);
        Ok(match a.provider.as_str() {
            // ModelArk's video endpoint names the aspect field `ratio`; this is a
            // deliberately text-only v1 contract (no caller URLs or arbitrary JSON).
            "byteplus" if a.modality == "image-to-video" => {
                let input = image()?;
                // Seedance 2.5 derives the output ratio from its first frame.
                // ModelArk rejects a fixed ratio together with first_frame.
                serde_json::json!({"model":a.model,"content":[{"type":"text","text":prompt},{"type":"image_url","image_url":{"url":data_uri(input)},"role":"first_frame"}],"duration":duration,"ratio":"adaptive","resolution":"720p","watermark":false})
            }
            "byteplus" => {
                serde_json::json!({"model":a.model,"content":[{"type":"text","text":prompt}],"duration":duration,"ratio":aspect,"resolution":"720p","watermark":false})
            }
            "kling" if a.modality == "image-to-video" => {
                if !matches!(duration, 5 | 10) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                let input = image()?;
                serde_json::json!({"model_name":a.model,"image":input.bytes_base64,"prompt":prompt,"duration":duration.to_string(),"mode":"pro","sound":"off"})
            }
            "kling" => {
                if !matches!(duration, 5 | 10) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"model_name":a.model,"prompt":prompt,"duration":duration.to_string(),"aspect_ratio":aspect})
            }
            "google-cloud" => {
                let mut instance = serde_json::json!({"prompt":prompt});
                if a.modality == "image-to-video" {
                    let input = a
                        .inputs
                        .first()
                        .ok_or(Error::Protocol("assignment_image_required"))?;
                    // Vertex's image-to-video operation consumes the source
                    // frame as `instance.image`; referenceImages are a
                    // different asset/style-conditioning capability.
                    instance["image"] = serde_json::json!({"bytesBase64Encoded":input.bytes_base64,"mimeType":input.mime});
                }
                serde_json::json!({"instances":[instance],"parameters":{"aspectRatio":aspect,"durationSeconds":duration,"storageUri":format!("gs://{}/attelier-api-miner/",self.google_output_bucket.as_deref().ok_or(Error::Protocol("google_cloud_project_location_or_bucket"))?),"sampleCount":1}})
            }
            // Each fal route owns its own body schema. In particular, the
            // Seedance Queue OpenAPI uses string durations; do not infer a
            // payload shape from a model family or an older provider API.
            "fal" if a.model.starts_with("fal-ai/veo3.1") && a.modality == "image-to-video" => {
                if !matches!(duration, 4 | 6 | 8) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                let input = image()?;
                serde_json::json!({"prompt":prompt,"image_url":data_uri(input),"duration":format!("{duration}s"),"aspect_ratio":aspect,"resolution":"720p"})
            }
            "fal" if a.model.starts_with("fal-ai/veo3.1") => {
                if !matches!(duration, 4 | 6 | 8) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"prompt":prompt,"duration":format!("{duration}s"),"aspect_ratio":aspect,"resolution":"720p"})
            }
            "fal" if a.model.starts_with("fal-ai/kling-video/") => {
                if !(3..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"prompt":prompt,"duration":duration.to_string(),"aspect_ratio":aspect})
            }
            "fal" if a.model == "bytedance/seedance-2.0/fast/image-to-video" => {
                if !(4..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                let input = image()?;
                serde_json::json!({"prompt":prompt,"image_url":data_uri(input),"duration":duration.to_string(),"aspect_ratio":aspect,"resolution":"720p"})
            }
            "fal" if a.model == "bytedance/seedance-2.5/text-to-video" => {
                // The provider supports up to 30 seconds. The worker keeps
                // the existing 15-second assignment cap to bound spend.
                if !(4..=15).contains(&duration) {
                    return Err(Error::Protocol("provider_duration_invalid"));
                }
                serde_json::json!({"prompt":prompt,"duration":duration.to_string(),"aspect_ratio":aspect,"resolution":"720p"})
            }
            "fal" => return Err(Error::Protocol("provider_model_not_approved")),
            _ => return Err(Error::Protocol("unsupported_provider")),
        })
    }
    fn output_url(&self, provider: &str, value: &Value) -> Option<String> {
        let candidates = [
            "/content/video_url",
            "/video/url",
            "/data/task_result/videos/0/url",
            "/response/generatedVideos/0/video/uri",
            "/response/videos/0/gcsUri",
            "/output/url",
            "/response/video/url",
        ];
        let url = candidates
            .iter()
            .find_map(|p| value.pointer(p).and_then(Value::as_str))?;
        let parsed = reqwest::Url::parse(url).ok()?;
        #[cfg(test)]
        if self
            .test_origin
            .as_ref()
            .is_some_and(|origin| origin.origin() == parsed.origin())
        {
            return Some(url.to_owned());
        }
        let host = parsed.host_str()?;
        let child_or_root = |suffix: &str| {
            host == suffix
                || host
                    .strip_suffix(suffix)
                    .is_some_and(|prefix| prefix.ends_with('.'))
        };
        let permitted = match provider {
            "byteplus" => {
                child_or_root("bytepluscdn.com")
                    || child_or_root("bytepluses.com")
                    || child_or_root("volces.com")
            }
            "kling" => child_or_root("klingai.com") || child_or_root("kuaishou.com"),
            "google-cloud" => {
                child_or_root("googleusercontent.com") || child_or_root("googleapis.com")
            }
            // fal's own schemas show result files at both fal.media and its
            // dedicated Google storage bucket. Do not permit arbitrary GCS
            // hosts: this exact host is the documented result origin.
            "fal" => {
                child_or_root("fal.media")
                    || child_or_root("fal.ai")
                    || host == "storage.googleapis.com"
            }
            _ => false,
        };
        if parsed.scheme() == "https" && permitted {
            Some(url.to_owned())
        } else {
            None
        }
    }
    fn media(response: reqwest::blocking::Response) -> Result<Vec<u8>> {
        if response
            .content_length()
            .is_some_and(|n| n > MAX_MEDIA_BYTES as u64)
        {
            return Err(Error::Protocol("invalid_output_bytes"));
        }
        let mut media = Vec::new();
        response
            .take(MAX_MEDIA_BYTES as u64 + 1)
            .read_to_end(&mut media)?;
        if media.is_empty() || media.len() > MAX_MEDIA_BYTES {
            return Err(Error::Protocol("invalid_output_bytes"));
        }
        Ok(media)
    }
}
impl Provider for DirectProvider {
    fn submit(&self, a: &Assignment) -> Result<ProviderSubmission> {
        if a.provider != self.configured_provider {
            return Err(Error::Protocol("configured_provider_mismatch"));
        }
        let endpoint = self.endpoint(&a.provider, &a.model, &a.modality)?;
        let authorization = self.authorization(&a.provider)?;
        let response = self
            .client
            .post(endpoint)
            .header("authorization", authorization)
            .header("idempotency-key", format!("{}:{}", a.job_id, a.attempt_id))
            .json(&self.payload(a)?)
            .send()?;
        let status = response.status();
        let raw = response.bytes()?.to_vec();
        if !status.is_success() {
            if matches!(status.as_u16(), 400 | 401 | 403 | 404 | 422) {
                let provider_code = serde_json::from_slice::<Value>(&raw)
                    .ok()
                    .and_then(|value| {
                        value
                            .pointer("/error/code")
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .filter(|code| {
                        !code.is_empty()
                            && code.len() <= 80
                            && code.as_bytes()[0].is_ascii_alphabetic()
                            && code.bytes().all(|byte| {
                                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')
                            })
                    });
                return Err(Error::ProviderTerminal {
                    reason: "provider_rejected",
                    http_status: Some(status.as_u16()),
                    provider_code,
                });
            }
            return Err(Error::Provider(format!(
                "{}:{}",
                a.provider,
                status.as_u16()
            )));
        }
        let value: Value = serde_json::from_slice(&raw)?;
        let reference =
            Self::reference(&value).ok_or(Error::Protocol("provider_reference_missing"))?;
        Ok(ProviderSubmission {
            reference,
            response: raw,
        })
    }
    fn collect(&self, a: &Assignment, submission: &ProviderSubmission) -> Result<Vec<u8>> {
        if a.provider != self.configured_provider {
            return Err(Error::Protocol("configured_provider_mismatch"));
        }
        let deadline = std::time::Instant::now()
            + std::time::Duration::from_millis((a.expires_at - now_ms()).max(0) as u64);
        loop {
            if std::time::Instant::now() >= deadline {
                return Err(Error::Protocol("provider_deadline"));
            }
            let authorization = self.authorization(&a.provider)?;
            let response = if a.provider == "google-cloud" {
                self.client
                    .post(self.status_url(a, &submission.reference)?)
                    .header("authorization", self.authorization("google-cloud")?)
                    .json(&serde_json::json!({"operationName":submission.reference}))
                    .send()?
            } else {
                self.client
                    .get(self.status_url(a, &submission.reference)?)
                    .header("authorization", authorization)
                    .send()?
            };
            let status = response.status();
            let raw = response.bytes()?.to_vec();
            if !status.is_success() {
                return Err(Error::Provider(format!(
                    "{}-poll:{}",
                    a.provider,
                    status.as_u16()
                )));
            }
            let value: Value = serde_json::from_slice(&raw)?;
            let status = if a.provider == "google-cloud" {
                if value.get("done").and_then(Value::as_bool) == Some(true) {
                    "done"
                } else {
                    "running"
                }
            } else {
                value
                    .pointer("/status")
                    .or_else(|| value.pointer("/data/task_status"))
                    .and_then(Value::as_str)
                    .unwrap_or("running")
            };
            let normalized_status = status.to_ascii_lowercase();
            if normalized_status == "cancelled" {
                return Err(Error::ProviderTerminal {
                    reason: "provider_cancelled",
                    http_status: None,
                    provider_code: None,
                });
            }
            if matches!(normalized_status.as_str(), "failed" | "error")
                || (a.provider == "fal" && status == "COMPLETED" && value.get("error").is_some())
            {
                return Err(Error::ProviderTerminal {
                    reason: "provider_failed",
                    http_status: None,
                    provider_code: None,
                });
            }
            if matches!(
                normalized_status.as_str(),
                "succeeded" | "succeed" | "completed" | "done"
            ) {
                let completed = if a.provider == "fal" {
                    #[cfg(test)]
                    let result_url = self
                        .test_origin
                        .as_ref()
                        .map(|origin| {
                            origin
                                .join(&format!("upstream/fal/result/{}", submission.reference))
                                .unwrap()
                        })
                        .unwrap();
                    #[cfg(not(test))]
                    let result_url = Self::fal_result_url(&a.model, &submission.reference)?;
                    let result = self
                        .client
                        .get(result_url)
                        .header("authorization", self.authorization("fal")?)
                        .send()?;
                    if !result.status().is_success() {
                        // fal can mark a queue request COMPLETED yet reject the
                        // result payload with a documented 422 policy/model
                        // validation error. At this point the provider ref is
                        // authoritative, so close it non-rewardably rather
                        // than endlessly resuming the same result request.
                        if result.status().as_u16() == 422 {
                            return Err(Error::ProviderTerminal {
                                reason: "provider_failed",
                                http_status: None,
                                provider_code: None,
                            });
                        }
                        return Err(Error::Provider("fal_result".into()));
                    };
                    serde_json::from_slice::<Value>(&result.bytes()?)?
                } else {
                    value
                };
                if a.provider == "google-cloud" {
                    if let Some(bytes) = completed
                        .pointer("/response/videos/0/bytesBase64Encoded")
                        .and_then(Value::as_str)
                    {
                        let media = B64
                            .decode(bytes)
                            .map_err(|_| Error::Protocol("google_output_base64"))?;
                        if media.is_empty() || media.len() > MAX_MEDIA_BYTES {
                            return Err(Error::Protocol("invalid_output_bytes"));
                        };
                        return Ok(media);
                    }
                    let gcs = completed
                        .pointer("/response/videos/0/gcsUri")
                        .and_then(Value::as_str)
                        .ok_or(Error::Protocol("google_output_uri"))?;
                    let path = gcs
                        .strip_prefix("gs://")
                        .ok_or(Error::Protocol("google_output_uri"))?;
                    let (bucket, object) = path
                        .split_once('/')
                        .ok_or(Error::Protocol("google_output_uri"))?;
                    if bucket
                        != self
                            .google_output_bucket
                            .as_deref()
                            .ok_or(Error::Protocol("google_cloud_project_location_or_bucket"))?
                        || object.is_empty()
                    {
                        return Err(Error::Protocol("google_output_bucket"));
                    }
                    #[cfg(not(test))]
                    let escaped: String = object
                        .bytes()
                        .map(|b| {
                            if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.') {
                                (b as char).to_string()
                            } else {
                                format!("%{b:02X}")
                            }
                        })
                        .collect();
                    #[cfg(test)]
                    let url = self
                        .test_origin
                        .as_ref()
                        .unwrap()
                        .join("upstream/google-cloud/output")
                        .unwrap();
                    #[cfg(not(test))] let url=reqwest::Url::parse(&format!("https://storage.googleapis.com/storage/v1/b/{bucket}/o/{escaped}?alt=media")).map_err(|_|Error::Protocol("google_output_uri"))?;
                    let output = self
                        .client
                        .get(url)
                        .header("authorization", self.authorization("google-cloud")?)
                        .send()?;
                    if !output.status().is_success() {
                        return Err(Error::Provider("google_output_download".into()));
                    };
                    return Self::media(output);
                }
                let url = self
                    .output_url(&a.provider, &completed)
                    .ok_or(Error::Protocol("provider_output_url"))?;
                let output = self.client.get(url).send()?;
                if !output.status().is_success() {
                    return Err(Error::Provider("output_download".into()));
                };
                return Self::media(output);
            }
            // Do not turn a near-expiry assignment into an unbounded polling
            // loop merely because the normal provider cadence is three seconds.
            std::thread::sleep(std::cmp::min(
                std::time::Duration::from_secs(3),
                deadline.saturating_duration_since(std::time::Instant::now()),
            ));
        }
    }
}

/// Narrow HTTPS client for the reviewed coordinator.  There is deliberately no
/// configurable path or generic signing endpoint: changing a path requires a
/// released worker contract.
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
    #[cfg(test)]
    fn test_new(base: &str, worker_id: String, miner_id: String, keys: Keys) -> Result<Self> {
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
        let session_id = self
            .session_id
            .as_deref()
            .ok_or(Error::Protocol("runtime_session_required"))?;
        let value = self.post("v1/api-miners/claim", "claim", serde_json::json!({}))?;
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
    fn decrypt_recovered_assignment(&self, value: &Value) -> Result<Assignment> {
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

/// Obtain raw evidence from a Phala dstack socket. Verification is deliberately
/// performed by the coordinator's pinned verifier, never by this worker.
pub fn phala_quote(report_data: &str) -> Result<Value> {
    if report_data.len() != 64 || !report_data.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::Protocol("invalid_report_data"));
    }
    let socket = std::env::var("DSTACK_SOCKET").unwrap_or_else(|_| "/var/run/dstack.sock".into());
    let body = serde_json::json!({"report_data":report_data}).to_string();
    let output = Command::new("curl")
        .args([
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "3",
            "--max-time",
            "10",
            "--unix-socket",
            &socket,
            "-X",
            "POST",
            "http://dstack/GetQuote",
            "-H",
            "Content-Type: application/json",
            "--data",
            &body,
        ])
        .output()
        .map_err(Error::Io)?;
    if !output.status.success() {
        return Err(Error::Protocol("phala_quote_failed"));
    }
    let value: Value = serde_json::from_slice(&output.stdout)?;
    if !value.get("quote").and_then(Value::as_str).is_some() {
        return Err(Error::Protocol("phala_quote_missing"));
    }
    let event_log = value
        .get("event_log")
        .or_else(|| value.get("eventLog"))
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty() && v.len() <= 1_000_000)
        .ok_or(Error::Protocol("phala_event_log_missing"))?;
    // Parse to reject a malformed dstack reply, but retain the original bytes:
    // the verifier pins SHA-256 over this exact JSON string.
    let _: Value =
        serde_json::from_str(event_log).map_err(|_| Error::Protocol("phala_event_log_invalid"))?;
    Ok(value)
}

/// Read only the dstack VM configuration needed by the independent OS
/// measurement reproducer. Never export the complete /Info response.
pub fn phala_vm_config() -> Result<Value> {
    let socket = std::env::var("DSTACK_SOCKET").unwrap_or_else(|_| "/var/run/dstack.sock".into());
    let output = Command::new("curl")
        .args([
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "3",
            "--max-time",
            "10",
            "--unix-socket",
            &socket,
            "http://dstack/Info",
        ])
        .output()
        .map_err(Error::Io)?;
    if !output.status.success() {
        return Err(Error::Protocol("phala_info_failed"));
    }
    let info: Value = serde_json::from_slice(&output.stdout)?;
    let raw = info
        .get("vm_config")
        .or_else(|| info.get("vmConfig"))
        .ok_or(Error::Protocol("phala_vm_config_missing"))?;
    let config = match raw {
        Value::Object(_) => raw.clone(),
        Value::String(json) => serde_json::from_str::<Value>(json)
            .map_err(|_| Error::Protocol("phala_vm_config_invalid"))?,
        _ => return Err(Error::Protocol("phala_vm_config_invalid")),
    };
    if !config.is_object() {
        return Err(Error::Protocol("phala_vm_config_invalid"));
    }
    Ok(config)
}

/// V2 quote binding for one runtime process. The persistent control key proves
/// reviewed-worker continuity; receipt and assignment keys are fresh runtime
/// material and never appear in the durable state directory.
pub fn attestation_report_data_v2(
    challenge: &str,
    control_public_key: &str,
    runtime_signing_public_key: &str,
    runtime_encryption_public_key: &str,
) -> Result<String> {
    if challenge.is_empty() || challenge.len() > 512 {
        return Err(Error::Protocol("attestation_challenge_invalid"));
    }
    Ok(sha256(
        format!(
            "attelier/api-miner-attestation/v2\0{}\0{}\0{}\0{}",
            challenge,
            control_public_key,
            runtime_signing_public_key,
            runtime_encryption_public_key
        )
        .as_bytes(),
    ))
}

/// Public, one-shot enrollment evidence. The gateway must independently verify
/// this raw quote; this object is never an admission decision.
pub fn enrollment_evidence(
    miner_id: &str,
    worker_id: &str,
    control_public_key: &str,
    runtime_signing_public_key: &str,
    runtime_encryption_public_key: &str,
    challenge: &str,
    quote: &Value,
    vm_config: &Value,
) -> Result<Value> {
    if miner_id.is_empty() || worker_id.is_empty() {
        return Err(Error::Protocol("enrollment_identity_invalid"));
    }
    let report_data = attestation_report_data_v2(
        challenge,
        control_public_key,
        runtime_signing_public_key,
        runtime_encryption_public_key,
    )?;
    let raw_quote = quote
        .get("quote")
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty())
        .map(str::to_owned)
        .ok_or(Error::Protocol("phala_quote_missing"))?;
    let event_log = quote
        .get("event_log")
        .or_else(|| quote.get("eventLog"))
        .and_then(Value::as_str)
        .filter(|v| !v.is_empty() && v.len() <= 1_000_000)
        .map(str::to_owned)
        .ok_or(Error::Protocol("phala_event_log_missing"))?;
    let _: Value =
        serde_json::from_str(&event_log).map_err(|_| Error::Protocol("phala_event_log_invalid"))?;
    if !vm_config.is_object() {
        return Err(Error::Protocol("phala_vm_config_invalid"));
    }
    Ok(serde_json::json!({
        "schema":"attelier/api-miner-enrollment/v2",
        "minerId":miner_id,
        "workerId":worker_id,
        "publicKey":control_public_key,
        "runtimeSigningPublicKey":runtime_signing_public_key,
        "runtimeEncryptionPublicKey":runtime_encryption_public_key,
        "challenge":challenge,
        "reportData":report_data,
        "quote":{"quote":raw_quote,"eventLog":event_log},
        "vmConfig":vm_config,
    }))
}
pub struct Worker<P: Provider> {
    pub worker_id: String,
    pub miner_id: String,
    pub keys: Keys,
    pub attestation: Attestation,
    pub state_path: PathBuf,
    pub provider: P,
}
impl<P: Provider> Worker<P> {
    /// A new process has fresh runtime keys. Durable rows are only recovery
    /// hints: every provider reference, assignment and response hash must be
    /// replaced by the runtime-session-signed gateway recovery response.
    pub fn recover_new_runtime(&self, coordinator: &Coordinator) -> Result<()> {
        let mut state = DurableState::load(&self.state_path)?;
        let rows: Vec<_> = state
            .accepted
            .iter()
            .map(|(id, row)| (id.clone(), row.clone()))
            .collect();
        for (id, hint) in rows {
            let value = coordinator.recover(&hint.assignment)?;
            match value.get("state").and_then(Value::as_str) {
                Some("started") => {
                    let assignment = coordinator.decrypt_recovered_assignment(&value)?;
                    let reference = value
                        .get("providerRef")
                        .and_then(Value::as_str)
                        .filter(|v| !v.is_empty())
                        .ok_or(Error::Protocol("recovery_provider_ref_missing"))?
                        .to_owned();
                    let response_hash = value
                        .get("providerResponseHash")
                        .and_then(Value::as_str)
                        .filter(|v| v.len() == 64 && v.bytes().all(|b| b.is_ascii_hexdigit()))
                        .ok_or(Error::Protocol("recovery_provider_response_hash_missing"))?
                        .to_owned();
                    state.accepted.insert(
                        id,
                        AcceptedAttempt {
                            assignment,
                            provider_ref: Some(reference),
                            provider_response: String::new(),
                            provider_response_hash: Some(response_hash),
                            state: "accepted".into(),
                            terminal_failure: None,
                        },
                    );
                }
                Some("accepted") | Some("failed") | Some("cancelled") => {
                    state.accepted.insert(
                        id,
                        AcceptedAttempt {
                            assignment: hint.assignment,
                            provider_ref: None,
                            provider_response: String::new(),
                            provider_response_hash: None,
                            state: "completed".into(),
                            terminal_failure: None,
                        },
                    );
                }
                Some("unknown") | Some("assigned") | None => {
                    // A fresh runtime key has no authority to mutate an
                    // assignment that never reached canonical `start`.
                    // Leave the gateway to expire/cancel it; never turn this
                    // stale disk hint into a first start or fallback submit.
                    state.accepted.insert(
                        id,
                        AcceptedAttempt {
                            assignment: hint.assignment,
                            provider_ref: None,
                            provider_response: String::new(),
                            provider_response_hash: None,
                            state: "recovered_unknown".into(),
                            terminal_failure: None,
                        },
                    );
                }
                _ => return Err(Error::Protocol("recovery_state_invalid")),
            }
        }
        state.save(&self.state_path)
    }

    fn record_terminal_failure(
        &self,
        coordinator: &Coordinator,
        state: &mut DurableState,
        id: String,
        assignment: Assignment,
        provider_ref: Option<String>,
        provider_response: String,
        reason: &'static str,
        http_status: Option<u16>,
        provider_code: Option<String>,
    ) -> Result<()> {
        // Persist before the ACK attempt: an ACK-loss/restart must retry only
        // this idempotent closure, never submit or poll paid work again.
        let failure = TerminalFailure {
            reason: reason.into(),
            http_status,
            provider_code,
        };
        state.accepted.insert(
            id.clone(),
            AcceptedAttempt {
                assignment: assignment.clone(),
                provider_ref: provider_ref.clone(),
                provider_response: provider_response.clone(),
                provider_response_hash: None,
                state: "failure_pending".into(),
                terminal_failure: Some(failure),
            },
        );
        state.save(&self.state_path)?;
        self.acknowledge_terminal_failure(
            coordinator,
            state,
            id,
            assignment,
            provider_ref,
            provider_response,
        )
    }

    fn acknowledge_terminal_failure(
        &self,
        coordinator: &Coordinator,
        state: &mut DurableState,
        id: String,
        assignment: Assignment,
        provider_ref: Option<String>,
        provider_response: String,
    ) -> Result<()> {
        let failure = state
            .accepted
            .get(&id)
            .and_then(|row| row.terminal_failure.clone())
            .ok_or(Error::Protocol("terminal_failure_missing"))?;
        coordinator.fail(
            &assignment,
            provider_ref.as_deref(),
            &failure.reason,
            failure.http_status,
            failure.provider_code.as_deref(),
        )?;
        state.accepted.insert(
            id,
            AcceptedAttempt {
                assignment,
                provider_ref,
                provider_response,
                provider_response_hash: None,
                state: "provider_failed".into(),
                terminal_failure: Some(failure),
            },
        );
        state.save(&self.state_path)
    }

    fn resume(&self, coordinator: &Coordinator, state: &mut DurableState) -> Result<bool> {
        let pending_failure = state.accepted.iter().find_map(|(id, row)| {
            (row.state == "failure_pending").then(|| (id.clone(), row.clone()))
        });
        if let Some((id, row)) = pending_failure {
            return self
                .acknowledge_terminal_failure(
                    coordinator,
                    state,
                    id,
                    row.assignment,
                    row.provider_ref,
                    row.provider_response,
                )
                .map(|_| true);
        }
        let pending = state.accepted.iter().find_map(|(id, row)| {
            (row.state == "accepted" && row.provider_ref.is_some())
                .then(|| (id.clone(), row.clone()))
        });
        let Some((id, row)) = pending else {
            return Ok(false);
        };
        let reference = row
            .provider_ref
            .ok_or(Error::Protocol("missing_provider_reference"))?;
        let response = B64URL
            .decode(&row.provider_response)
            .map_err(|_| Error::Protocol("saved_provider_response_invalid"))?;
        let submission = ProviderSubmission {
            reference: reference.clone(),
            response,
        };
        // A start ACK can be lost after the gateway has accepted the original
        // provider reference. Replay the same signed start before touching the
        // provider; an `accepted` response means the gateway already closed
        // this attempt, so downloading a potentially expired result is wrong.
        let response_hash = row
            .provider_response_hash
            .as_deref()
            .ok_or(Error::Protocol("provider_response_hash_missing"))?;
        match coordinator
            .start(&row.assignment, &reference, response_hash)?
            .as_deref()
        {
            Some("accepted") => {
                state.accepted.insert(
                    id,
                    AcceptedAttempt {
                        assignment: row.assignment,
                        provider_ref: None,
                        provider_response: String::new(),
                        provider_response_hash: None,
                        state: "completed".into(),
                        terminal_failure: None,
                    },
                );
                state.save(&self.state_path)?;
                return Ok(true);
            }
            Some("started") | None => {}
            _ => return Err(Error::Protocol("coordinator_start_state_invalid")),
        }
        let media = match self.provider.collect(&row.assignment, &submission) {
            Ok(value) => value,
            Err(Error::ProviderTerminal {
                reason,
                http_status,
                provider_code,
            }) => {
                self.record_terminal_failure(
                    coordinator,
                    state,
                    id,
                    row.assignment,
                    Some(reference),
                    row.provider_response,
                    reason,
                    http_status,
                    provider_code,
                )?;
                return Ok(true);
            }
            Err(error) => return Err(error),
        };
        let receipt = Receipt::signed_with_provider_hash_in_session(
            &row.assignment,
            self.attestation.clone(),
            coordinator.current_session()?,
            reference,
            row.provider_response_hash
                .clone()
                .ok_or(Error::Protocol("provider_response_hash_missing"))?,
            &media,
            &self.keys,
        )?;
        receipt.verify_with(&self.keys.public())?;
        coordinator.complete(&receipt, &media)?;
        state.accepted.insert(
            id,
            AcceptedAttempt {
                assignment: row.assignment,
                provider_ref: None,
                provider_response: String::new(),
                provider_response_hash: None,
                state: "completed".into(),
                terminal_failure: None,
            },
        );
        state.save(&self.state_path)?;
        Ok(true)
    }
    pub fn run_assignment(&self, assignment: Assignment) -> Result<Receipt> {
        assignment.validate(&self.worker_id, &self.miner_id)?;
        self.attestation.validate()?;
        let mut state = DurableState::load(&self.state_path)?;
        if pause_requested(&self.state_path, &state) {
            return Err(Error::Protocol("worker_paused"));
        };
        let id = format!("{}:{}", assignment.job_id, assignment.attempt_id);
        if state.accepted.contains_key(&id) {
            return Err(Error::Protocol("attempt_already_accepted_reconcile"));
        };
        state.accepted.insert(
            id.clone(),
            AcceptedAttempt {
                assignment: assignment.clone(),
                provider_ref: None,
                provider_response: String::new(),
                provider_response_hash: None,
                state: "submitting".into(),
                terminal_failure: None,
            },
        );
        state.save(&self.state_path)?;
        if pause_requested(&self.state_path, &state) {
            return Err(Error::Protocol("worker_paused"));
        }
        let submission = match self.provider.submit(&assignment) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        state.accepted.insert(
            id,
            AcceptedAttempt {
                assignment: assignment.clone(),
                provider_ref: Some(submission.reference.clone()),
                provider_response: B64URL.encode(&submission.response),
                provider_response_hash: Some(sha256(&submission.response)),
                state: "accepted".into(),
                terminal_failure: None,
            },
        );
        state.save(&self.state_path)?;
        let media = self.provider.collect(&assignment, &submission)?;
        let receipt = Receipt::signed_with_provider_hash(
            &assignment,
            self.attestation.clone(),
            submission.reference,
            sha256(&submission.response),
            &media,
            &self.keys,
        )?;
        receipt.verify_with(&self.keys.public())?;
        Ok(receipt)
    }
    pub fn run_once(&self, coordinator: &Coordinator) -> Result<bool> {
        let mut state_before = DurableState::load(&self.state_path)?;
        if pause_requested(&self.state_path, &state_before) {
            return Err(Error::Protocol("worker_paused"));
        };
        if self.resume(coordinator, &mut state_before)? {
            return Ok(true);
        };
        let _session_id = state_before
            .session_id
            .as_deref()
            .ok_or(Error::Protocol("attestation_required"))?
            .to_owned();
        let Some(assignment) = coordinator.claim()? else {
            return Ok(false);
        };
        assignment.validate(&self.worker_id, &self.miner_id)?;
        self.attestation.validate()?;
        let id = format!("{}:{}", assignment.job_id, assignment.attempt_id);
        let mut state = state_before;
        if pause_requested(&self.state_path, &state) {
            return Err(Error::Protocol("worker_paused"));
        };
        if state.accepted.contains_key(&id) {
            coordinator.unknown(&assignment, "attempt_already_accepted_reconcile")?;
            return Ok(true);
        }
        state.accepted.insert(
            id.clone(),
            AcceptedAttempt {
                assignment: assignment.clone(),
                provider_ref: None,
                provider_response: String::new(),
                provider_response_hash: None,
                state: "submitting".into(),
                terminal_failure: None,
            },
        );
        state.save(&self.state_path)?;
        if pause_requested(&self.state_path, &state) {
            return Err(Error::Protocol("worker_paused"));
        }
        let submission = match self.provider.submit(&assignment) {
            Ok(value) => value,
            Err(Error::ProviderTerminal {
                reason,
                http_status,
                provider_code,
            }) => {
                self.record_terminal_failure(
                    coordinator,
                    &mut state,
                    id,
                    assignment,
                    None,
                    String::new(),
                    reason,
                    http_status,
                    provider_code,
                )?;
                return Ok(true);
            }
            Err(error) => {
                coordinator.unknown(&assignment, "provider_submission_uncertain")?;
                return Err(error);
            }
        };
        state.accepted.insert(
            id.clone(),
            AcceptedAttempt {
                assignment: assignment.clone(),
                provider_ref: Some(submission.reference.clone()),
                provider_response: B64URL.encode(&submission.response),
                provider_response_hash: Some(sha256(&submission.response)),
                state: "accepted".into(),
                terminal_failure: None,
            },
        );
        state.save(&self.state_path)?;
        coordinator.start(
            &assignment,
            &submission.reference,
            &sha256(&submission.response),
        )?;
        let media = match self.provider.collect(&assignment, &submission) {
            Ok(value) => value,
            Err(Error::ProviderTerminal {
                reason,
                http_status,
                provider_code,
            }) => {
                self.record_terminal_failure(
                    coordinator,
                    &mut state,
                    id,
                    assignment,
                    Some(submission.reference),
                    B64URL.encode(&submission.response),
                    reason,
                    http_status,
                    provider_code,
                )?;
                return Ok(true);
            }
            Err(error) => {
                coordinator.unknown(&assignment, "provider_result_pending_or_uncertain")?;
                return Err(error);
            }
        };
        let receipt = Receipt::signed_with_provider_hash_in_session(
            &assignment,
            self.attestation.clone(),
            coordinator.current_session()?,
            submission.reference,
            sha256(&submission.response),
            &media,
            &self.keys,
        )?;
        receipt.verify_with(&self.keys.public())?;
        coordinator.complete(&receipt, &media)?;
        state.accepted.insert(
            id,
            AcceptedAttempt {
                assignment,
                provider_ref: None,
                provider_response: String::new(),
                provider_response_hash: None,
                state: "completed".into(),
                terminal_failure: None,
            },
        );
        state.save(&self.state_path)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    use std::os::unix::net::UnixListener;
    use std::{
        io::{Read, Write},
        net::TcpListener,
        thread,
    };
    use tempfile::tempdir;
    struct Fake;
    impl Provider for Fake {
        fn submit(&self, _: &Assignment) -> Result<ProviderSubmission> {
            Ok(ProviderSubmission {
                reference: "ref".into(),
                response: b"provider-response".to_vec(),
            })
        }
        fn collect(&self, _: &Assignment, _: &ProviderSubmission) -> Result<Vec<u8>> {
            Ok(b"media".to_vec())
        }
    }
    fn assignment() -> Assignment {
        let request = serde_json::json!({"prompt":"test","duration":5,"aspectRatio":"16:9"});
        Assignment {
            schema: RECEIPT_SCHEMA.into(),
            job_id: "job".into(),
            attempt_id: "attempt".into(),
            worker_id: "worker".into(),
            miner_id: "miner".into(),
            session_id: "session".into(),
            contract_id: "seedance-v1".into(),
            provider: "byteplus".into(),
            model: "seedance".into(),
            modality: "text-to-video".into(),
            inputs: vec![],
            input_hash: "a".repeat(64),
            request_hash: sha256(&canonical(&request).unwrap()),
            grant_nonce: "grant".into(),
            expires_at: now_ms() + 10_000,
            bid_microusd: 1,
            request,
        }
    }
    fn attestation() -> Attestation {
        Attestation {
            measurement: "b".repeat(64),
            issued_at: now_ms() - 1,
            expires_at: now_ms() + 10_000,
            status: "verified".into(),
        }
    }
    fn encrypted(keys: &Keys, session: &str, a: &Assignment) -> Value {
        let ephemeral = StaticSecret::random_from_rng(OsRng);
        let shared = ephemeral.diffie_hellman(&X25519PublicKey::from(&keys.encryption));
        let key = Sha256::digest([shared.as_bytes().as_slice(), session.as_bytes()].concat());
        let cipher = Aes256Gcm::new_from_slice(&key).unwrap();
        let nonce = [7u8; 12];
        let ciphertext = cipher
            .encrypt(
                Nonce::from_slice(&nonce),
                canonical(&serde_json::to_value(a).unwrap())
                    .unwrap()
                    .as_ref(),
            )
            .unwrap();
        let mut der = vec![
            0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x6e, 0x03, 0x21, 0x00,
        ];
        der.extend(X25519PublicKey::from(&ephemeral).as_bytes());
        serde_json::json!({"ephemeralPublicKey":B64URL.encode(der),"nonce":B64URL.encode(nonce),"ciphertext":B64URL.encode(ciphertext)})
    }

    #[test]
    fn attestation_report_data_v2_bounds_the_public_challenge() {
        let challenge = "operator-visible-challenge";
        assert_eq!(
            attestation_report_data_v2(
                challenge,
                "control",
                "runtime-signing",
                "runtime-encryption"
            )
            .unwrap()
            .len(),
            64
        );
        assert!(attestation_report_data_v2("", "control", "signing", "encryption").is_err());
        assert!(
            attestation_report_data_v2(&"a".repeat(513), "control", "signing", "encryption")
                .is_err()
        );
    }

    #[test]
    fn runtime_keys_are_ephemeral_and_v2_quote_binds_both_identities() {
        let d = tempdir().unwrap();
        let control_path = d.path().join("control.ed25519");
        let control = ControlKey::load_or_create(&control_path).unwrap();
        let again = ControlKey::load_or_create(&control_path).unwrap();
        assert_eq!(control.public(), again.public());
        assert!(!control_path.with_extension("x25519").exists());

        let first = Keys::fresh();
        let second = Keys::fresh();
        assert_ne!(first.public(), second.public());
        assert_ne!(first.encryption_public(), second.encryption_public());
        assert_eq!(std::fs::read_dir(d.path()).unwrap().count(), 1);
        let bound = attestation_report_data_v2(
            "challenge",
            &control.public(),
            &first.public(),
            &first.encryption_public(),
        )
        .unwrap();
        assert_ne!(
            bound,
            attestation_report_data_v2(
                "challenge",
                &control.public(),
                &second.public(),
                &second.encryption_public(),
            )
            .unwrap()
        );
    }

    #[cfg(unix)]
    #[test]
    fn enrollment_quote_uses_dstack_socket_and_exact_report_data() {
        let d = tempdir().unwrap();
        let socket = d.path().join("dstack.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        let report =
            attestation_report_data_v2("public", "control", "signing", "encryption").unwrap();
        let expected = report.clone();
        let server = thread::spawn(move || {
            for (path, body) in [
                (
                    "POST /GetQuote HTTP/1.1",
                    r#"{"quote":"aabb","event_log":"{\"events\":[]}"}"#,
                ),
                (
                    "GET /Info HTTP/1.1",
                    r#"{"vm_config":"{\"os\":\"phala\"}"}"#,
                ),
            ] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut raw = [0u8; 4096];
                let size = stream.read(&mut raw).unwrap();
                let request = String::from_utf8_lossy(&raw[..size]);
                assert!(request.contains(path));
                if path.starts_with("POST") {
                    assert!(request.contains(&format!(r#"{{"report_data":"{expected}"}}"#)));
                }
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )
                .unwrap();
            }
        });
        std::env::set_var("DSTACK_SOCKET", &socket);
        let quote = phala_quote(&report).unwrap();
        let vm_config = phala_vm_config().unwrap();
        std::env::remove_var("DSTACK_SOCKET");
        server.join().unwrap();
        assert_eq!(quote["quote"], "aabb");
        assert_eq!(quote["event_log"], r#"{"events":[]}"#);
        assert_eq!(vm_config, serde_json::json!({"os":"phala"}));
        let evidence = enrollment_evidence(
            "miner",
            "worker",
            "control",
            "signing",
            "encryption",
            "public",
            &quote,
            &vm_config,
        )
        .unwrap();
        assert_eq!(evidence["schema"], "attelier/api-miner-enrollment/v2");
        assert_eq!(evidence["reportData"], report);
        assert_eq!(evidence["publicKey"], "control");
        assert_eq!(evidence["runtimeSigningPublicKey"], "signing");
        assert_eq!(evidence["runtimeEncryptionPublicKey"], "encryption");
        assert_eq!(evidence["quote"]["quote"], "aabb");
        assert_eq!(evidence["quote"]["eventLog"], r#"{"events":[]}"#);
        assert_eq!(evidence["vmConfig"], serde_json::json!({"os":"phala"}));
        for forbidden in ["BYTEPLUS_API_KEY", "FAL_KEY", "privateKey", "state"] {
            assert!(evidence.get(forbidden).is_none());
        }
    }
    #[test]
    fn receipt_is_bound_and_replay_is_not_sent() {
        let d = tempdir().unwrap();
        let worker = Worker {
            worker_id: "worker".into(),
            miner_id: "miner".into(),
            keys: Keys::load_or_create(&d.path().join("key")).unwrap(),
            attestation: attestation(),
            state_path: d.path().join("state.json"),
            provider: Fake,
        };
        let receipt = worker.run_assignment(assignment()).unwrap();
        receipt.verify_with(&worker.keys.public()).unwrap();
        assert!(worker.run_assignment(assignment()).is_err());
    }
    #[test]
    fn receipt_tampering_fails() {
        let d = tempdir().unwrap();
        let keys = Keys::load_or_create(&d.path().join("key")).unwrap();
        let mut r = Receipt::signed(
            &assignment(),
            attestation(),
            "ref".into(),
            b"r",
            b"output",
            &keys,
        )
        .unwrap();
        r.output_hash = "0".repeat(64);
        assert!(r.verify_with(&keys.public()).is_err());
    }
    #[test]
    fn durable_save_survives_a_crash_leftover_from_a_reused_pid() {
        let d = tempdir().unwrap();
        let state = d.path().join("state.json");
        // This is the exact deterministic temporary name used by the previous
        // release. It may survive a CVM crash and collide after PID reuse.
        let stale = state.with_extension(format!("{}.tmp", std::process::id()));
        fs::write(&stale, b"partial old state").unwrap();
        let expected = DurableState {
            session_id: Some("fresh-session".into()),
            ..Default::default()
        };
        expected.save(&state).unwrap();
        assert_eq!(
            DurableState::load(&state).unwrap().session_id,
            expected.session_id
        );
        assert!(stale.exists());
    }
    #[test]
    fn paused_state_blocks_submission_before_provider_call() {
        struct Never;
        impl Provider for Never {
            fn submit(&self, _: &Assignment) -> Result<ProviderSubmission> {
                panic!("paused worker submitted")
            }
            fn collect(&self, _: &Assignment, _: &ProviderSubmission) -> Result<Vec<u8>> {
                unreachable!()
            }
        }
        let d = tempdir().unwrap();
        let state = d.path().join("state.json");
        DurableState {
            paused: true,
            ..Default::default()
        }
        .save(&state)
        .unwrap();
        let worker = Worker {
            worker_id: "worker".into(),
            miner_id: "miner".into(),
            keys: Keys::load_or_create(&d.path().join("key")).unwrap(),
            attestation: attestation(),
            state_path: state,
            provider: Never,
        };
        assert!(worker.run_assignment(assignment()).is_err());
    }
    #[test]
    fn worker_pull_loop_posts_started_then_bound_completion() {
        let d = tempdir().unwrap();
        let keys = Keys::load_or_create(&d.path().join("key")).unwrap();
        let encrypted = encrypted(&keys, "session", &assignment());
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            for response in [
                serde_json::json!({"encryptedAssignment":encrypted}),
                serde_json::json!({"ok":true}),
                serde_json::json!({"ok":true}),
            ] {
                let (mut socket, _) = listener.accept().unwrap();
                let mut raw = vec![0; 16384];
                let n = socket.read(&mut raw).unwrap();
                let request = String::from_utf8_lossy(&raw[..n]);
                assert!(request.contains("POST /v1/api-miners/"));
                let body = response.to_string();
                write!(socket,"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",body.len(),body).unwrap();
            }
        });
        let coordinator = Coordinator::test_new(
            &format!("http://{address}/"),
            "worker".into(),
            "miner".into(),
            keys.clone(),
        )
        .unwrap();
        let state_path = d.path().join("state.json");
        DurableState {
            session_id: Some("session".into()),
            ..Default::default()
        }
        .save(&state_path)
        .unwrap();
        let worker = Worker {
            worker_id: "worker".into(),
            miner_id: "miner".into(),
            keys,
            attestation: attestation(),
            state_path,
            provider: Fake,
        };
        assert!(worker.run_once(&coordinator).unwrap());
        handle.join().unwrap();
    }

    #[test]
    fn terminal_provider_failure_is_acknowledged_once_then_next_assignment_runs() {
        struct TerminalThenMedia(Mutex<Vec<String>>);
        impl Provider for TerminalThenMedia {
            fn submit(&self, a: &Assignment) -> Result<ProviderSubmission> {
                self.0.lock().unwrap().push(a.job_id.clone());
                Ok(ProviderSubmission {
                    reference: format!("provider-{}", a.job_id),
                    response: b"provider-response".to_vec(),
                })
            }
            fn collect(&self, a: &Assignment, _: &ProviderSubmission) -> Result<Vec<u8>> {
                if a.job_id == "job" {
                    Err(Error::ProviderTerminal {
                        reason: "provider_failed",
                        http_status: None,
                        provider_code: None,
                    })
                } else {
                    Ok(b"media".to_vec())
                }
            }
        }

        let d = tempdir().unwrap();
        let keys = Keys::load_or_create(&d.path().join("key")).unwrap();
        let first = assignment();
        let mut second = assignment();
        second.job_id = "job-next".into();
        second.attempt_id = "attempt-next".into();
        let first_envelope = encrypted(&keys, "session", &first);
        let second_envelope = encrypted(&keys, "session", &second);
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            for (index, response) in [
                serde_json::json!({"encryptedAssignment":first_envelope}),
                serde_json::json!({"ok":true}),
                serde_json::json!({"ok":true}),
                serde_json::json!({"ok":true}),
                serde_json::json!({"encryptedAssignment":second_envelope}),
                serde_json::json!({"ok":true}),
                serde_json::json!({"ok":true}),
            ]
            .into_iter()
            .enumerate()
            {
                let (mut socket, _) = listener.accept().unwrap();
                let mut raw = vec![0; 16384];
                let n = socket.read(&mut raw).unwrap();
                let request = String::from_utf8_lossy(&raw[..n]);
                if matches!(index, 2 | 3) {
                    assert!(request.starts_with("POST /v1/api-miners/fail "));
                    let body = request.split_once("\r\n\r\n").unwrap().1;
                    let signed: Value = serde_json::from_str(body).unwrap();
                    assert_eq!(signed["body"]["action"], "fail");
                    assert_eq!(signed["body"]["payload"]["reason"], "provider_failed");
                    assert_eq!(signed["body"]["payload"]["providerRef"], "provider-job");
                    assert!(signed["body"]["payload"].get("httpStatus").is_none());
                    assert!(signed["body"]["payload"].get("providerCode").is_none());
                } else {
                    assert!(request.contains("POST /v1/api-miners/"));
                    assert!(!request.contains("/unknown"));
                }
                let body = response.to_string();
                let status = if index == 2 {
                    "500 Internal Server Error"
                } else {
                    "200 OK"
                };
                write!(socket,"HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",body.len(),body).unwrap();
            }
        });
        let coordinator = Coordinator::test_new(
            &format!("http://{address}/"),
            "worker".into(),
            "miner".into(),
            keys.clone(),
        )
        .unwrap();
        let state_path = d.path().join("state.json");
        DurableState {
            session_id: Some("session".into()),
            ..Default::default()
        }
        .save(&state_path)
        .unwrap();
        let worker = Worker {
            worker_id: "worker".into(),
            miner_id: "miner".into(),
            keys,
            attestation: attestation(),
            state_path: state_path.clone(),
            provider: TerminalThenMedia(Mutex::new(Vec::new())),
        };
        assert!(worker.run_once(&coordinator).is_err());
        assert_eq!(
            DurableState::load(&state_path).unwrap().accepted["job:attempt"].state,
            "failure_pending"
        );
        assert!(worker.run_once(&coordinator).unwrap());
        assert!(worker.run_once(&coordinator).unwrap());
        assert_eq!(
            *worker.provider.0.lock().unwrap(),
            vec!["job".to_string(), "job-next".to_string()]
        );
        let state = DurableState::load(&state_path).unwrap();
        assert_eq!(state.accepted["job:attempt"].state, "provider_failed");
        assert_eq!(state.accepted["job-next:attempt-next"].state, "completed");
        handle.join().unwrap();
    }

    #[test]
    fn direct_provider_submit_uses_pinned_http_contracts() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for _ in 0..5 {
                let (mut s, _) = listener.accept().unwrap();
                let mut raw = vec![0; 8192];
                let n = s.read(&mut raw).unwrap();
                let request = String::from_utf8_lossy(&raw[..n]);
                assert!(request.contains("authorization:"));
                let body = match request.find("\r\n\r\n") {
                    Some(i) => &request[i + 4..],
                    None => "",
                };
                let response = if request.contains("/byteplus/") {
                    assert!(body.contains("\"ratio\""));
                    r#"{"id":"byteplus-task"}"#
                } else if request.contains("/kling/") {
                    assert!(body.contains("\"model_name\""));
                    let bearer = request
                        .lines()
                        .find_map(|line| line.strip_prefix("authorization: Bearer "))
                        .expect("Kling bearer token");
                    let parts: Vec<_> = bearer.split('.').collect();
                    assert_eq!(parts.len(), 3);
                    let claims: Value =
                        serde_json::from_slice(&B64URL.decode(parts[1]).unwrap()).unwrap();
                    assert_eq!(claims["iss"], "fixture-access");
                    assert_eq!(
                        claims["exp"].as_i64().unwrap() - claims["nbf"].as_i64().unwrap(),
                        1_805
                    );
                    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(b"fixture-secret").unwrap();
                    mac.update(format!("{}.{}", parts[0], parts[1]).as_bytes());
                    mac.verify_slice(&B64URL.decode(parts[2]).unwrap()).unwrap();
                    r#"{"data":{"task_id":"kling-task"}}"#
                } else if request.contains("/google-cloud/") {
                    assert!(body.contains("\"storageUri\""));
                    r#"{"name":"operations/veo-task"}"#
                } else {
                    assert!(request.contains("Key fixture-key"));
                    r#"{"request_id":"fal-task"}"#
                };
                write!(s,"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",response.len(),response).unwrap();
            }
        });
        let cases = [
            ("byteplus", "dreamina-seedance-2-5-260628"),
            ("kling", "kling-v2-6"),
            ("google-cloud", "veo-3.1-generate-001"),
            ("fal", "bytedance/seedance-2.5/text-to-video"),
            ("fal", "fal-ai/veo3.1"),
        ];
        for (provider, model) in cases {
            let mut a = assignment();
            a.provider = provider.into();
            a.model = model.into();
            if model.starts_with("fal-ai/veo3.1") {
                a.request["duration"] = serde_json::json!(4);
                a.request_hash = sha256(&canonical(&a.request).unwrap());
            }
            DirectProvider::fixture(&format!("http://{address}/"), provider)
                .submit(&a)
                .unwrap();
        }
        server.join().unwrap();
    }

    #[test]
    fn fal_queue_routes_use_the_owner_and_app_not_the_endpoint_path() {
        let cases = [
            (
                "bytedance/seedance-2.5/text-to-video",
                "bytedance/seedance-2.5",
            ),
            (
                "bytedance/seedance-2.0/fast/image-to-video",
                "bytedance/seedance-2.0",
            ),
            (
                "fal-ai/kling-video/v3/turbo/standard/text-to-video",
                "fal-ai/kling-video",
            ),
            ("fal-ai/veo3.1/fast", "fal-ai/veo3.1"),
            ("fal-ai/veo3.1/fast/image-to-video", "fal-ai/veo3.1"),
        ];
        for (model, app) in cases {
            assert_eq!(DirectProvider::fal_app(model).unwrap(), app);
            assert_eq!(
                DirectProvider::fal_status_url(model, "request-id")
                    .unwrap()
                    .as_str(),
                format!("https://queue.fal.run/{app}/requests/request-id/status")
            );
            assert_eq!(
                DirectProvider::fal_result_url(model, "request-id")
                    .unwrap()
                    .as_str(),
                format!("https://queue.fal.run/{app}/requests/request-id")
            );
        }
    }

    #[test]
    fn phala_compose_keeps_identity_and_paid_reconciliation_on_a_named_volume() {
        let compose = include_str!("../docker-compose.phala.yml");
        // dstack persists Docker's data directory on the encrypted disk. A
        // named volume with an explicit name survives a compose/image update;
        // an arbitrary guest-path bind does not give this worker that invariant.
        assert!(compose.contains("- attelier-api-miner-state:/data/attelier-api-miner"));
        assert!(compose.contains("ATTELIER_MINER_STATE_DIR: /data/attelier-api-miner"));
        assert!(compose.contains("attelier-api-miner-state:\n    name: attelier-api-miner-state"));
        assert!(compose.contains("state-init:\n    image:"));
        assert!(compose.contains("condition: service_completed_successfully"));
        assert!(compose.contains("chown 10001:10001 /data/attelier-api-miner"));
        assert!(!compose.contains("ATTELIER_MINER_STATE_DIR:?owner-only host directory"));
        assert!(!compose.contains(":/var/lib/attelier-api-miner"));
    }

    #[test]
    fn fal_payloads_and_result_hosts_are_model_pinned() {
        let direct = DirectProvider::fixture("http://127.0.0.1:1/", "fal");
        let cases = [
            (
                "bytedance/seedance-2.5/text-to-video",
                5,
                "\"5\"",
                Some("720p"),
            ),
            (
                "bytedance/seedance-2.0/fast/image-to-video",
                5,
                "\"5\"",
                Some("720p"),
            ),
            (
                "fal-ai/kling-video/v3/turbo/standard/text-to-video",
                5,
                "\"5\"",
                None,
            ),
            ("fal-ai/veo3.1/fast", 4, "\"4s\"", Some("720p")),
            ("fal-ai/veo3.1/image-to-video", 4, "\"4s\"", Some("720p")),
        ];
        for (model, requested_duration, duration, resolution) in cases {
            let mut a = assignment();
            a.provider = "fal".into();
            a.model = model.into();
            a.request["duration"] = serde_json::json!(requested_duration);
            a.request_hash = sha256(&canonical(&a.request).unwrap());
            if model.ends_with("image-to-video") {
                a.modality = "image-to-video".into();
                a.inputs = vec![AssignmentInput {
                    id: "frame".into(),
                    mime: "image/png".into(),
                    sha256: sha256(b"img"),
                    bytes_base64: B64.encode(b"img"),
                }];
            }
            let payload = direct.payload(&a).unwrap();
            assert_eq!(payload["duration"].to_string(), duration);
            assert_eq!(
                payload.get("resolution").and_then(Value::as_str),
                resolution
            );
            // fal documents end_user_id as optional. The worker has no
            // authenticated end-user identity in this miner contract, so it
            // must not invent one for an upstream provider.
            assert!(payload.get("end_user_id").is_none());
        }
        assert!(direct.output_url("fal", &serde_json::json!({"video":{"url":"https://storage.googleapis.com/falserverless/result.mp4"}})).is_some());
        assert!(direct
            .output_url(
                "fal",
                &serde_json::json!({"video":{"url":"https://evil.example/result.mp4"}})
            )
            .is_none());
    }

    #[test]
    fn provider_duration_enums_are_enforced_before_submission() {
        let mut a = assignment();
        a.provider = "byteplus".into();
        a.model = "dreamina-seedance-2-5-260628".into();
        let native = DirectProvider::fixture("http://127.0.0.1:1/", "byteplus")
            .payload(&a)
            .unwrap();
        assert_eq!(native["duration"], 5);
        assert_eq!(native["resolution"], "720p");

        a.provider = "fal".into();
        a.model = "fal-ai/veo3.1".into();
        assert!(matches!(
            DirectProvider::fixture("http://127.0.0.1:1/", "fal").payload(&a),
            Err(Error::Protocol("provider_duration_invalid"))
        ));

        a.model = "bytedance/seedance-2.0/fast/image-to-video".into();
        a.modality = "image-to-video".into();
        a.request["duration"] = serde_json::json!(3);
        a.request_hash = sha256(&canonical(&a.request).unwrap());
        a.inputs = vec![AssignmentInput {
            id: "frame".into(),
            mime: "image/png".into(),
            sha256: sha256(b"img"),
            bytes_base64: B64.encode(b"img"),
        }];
        assert!(matches!(
            DirectProvider::fixture("http://127.0.0.1:1/", "fal").payload(&a),
            Err(Error::Protocol("provider_duration_invalid"))
        ));

        a.provider = "kling".into();
        a.model = "kling-v2-6".into();
        a.modality = "text-to-video".into();
        a.inputs.clear();
        a.request["duration"] = serde_json::json!(4);
        a.request_hash = sha256(&canonical(&a.request).unwrap());
        assert!(matches!(
            DirectProvider::fixture("http://127.0.0.1:1/", "kling").payload(&a),
            Err(Error::Protocol("provider_duration_invalid"))
        ));
        a.request["duration"] = serde_json::json!(10);
        a.request_hash = sha256(&canonical(&a.request).unwrap());
        assert_eq!(
            DirectProvider::fixture("http://127.0.0.1:1/", "kling")
                .payload(&a)
                .unwrap()["duration"],
            "10"
        );
    }

    #[test]
    fn image_id_is_only_an_i2v_binding_and_never_reaches_provider_payloads() {
        let direct = DirectProvider::fixture("http://127.0.0.1:1/", "fal");
        let mut a = assignment();
        a.provider = "fal".into();
        a.model = "bytedance/seedance-2.0/fast/image-to-video".into();
        a.modality = "image-to-video".into();
        a.inputs = vec![AssignmentInput {
            id: "frame".into(),
            mime: "image/png".into(),
            sha256: sha256(b"img"),
            bytes_base64: B64.encode(b"img"),
        }];
        a.request["imageId"] = serde_json::json!("frame");
        a.request_hash = sha256(&canonical(&a.request).unwrap());
        let payload = direct.payload(&a).unwrap();
        assert!(payload.get("imageId").is_none());
        let byteplus = DirectProvider::fixture("http://127.0.0.1:1/", "byteplus");
        a.provider = "byteplus".into();
        a.model = "dreamina-seedance-2-5-260628".into();
        let byteplus_payload = byteplus.payload(&a).unwrap();
        assert_eq!(byteplus_payload["ratio"], "adaptive");
        a.request["imageId"] = serde_json::json!("other");
        a.request_hash = sha256(&canonical(&a.request).unwrap());
        assert!(matches!(
            direct.payload(&a),
            Err(Error::Protocol("assignment_image_id_mismatch"))
        ));

        let mut text = assignment();
        text.provider = "fal".into();
        text.model = "bytedance/seedance-2.5/text-to-video".into();
        text.request["imageId"] = serde_json::json!("frame");
        text.request_hash = sha256(&canonical(&text.request).unwrap());
        assert!(matches!(
            direct.payload(&text),
            Err(Error::Protocol("provider_request_unsupported"))
        ));
    }

    #[test]
    fn direct_provider_rejects_an_assignment_for_another_provider_before_network() {
        let mut a = assignment();
        a.provider = "byteplus".into();
        a.model = "dreamina-seedance-2-5-260628".into();
        assert!(matches!(
            DirectProvider::fixture("http://127.0.0.1:1/", "fal").submit(&a),
            Err(Error::Protocol("configured_provider_mismatch"))
        ));
    }

    #[test]
    fn definitive_submit_rejection_preserves_only_a_safe_provider_code() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut raw = [0u8; 8192];
            let _ = socket.read(&mut raw).unwrap();
            let body =
                r#"{"error":{"code":"Model.NotActivated","message":"must never leave worker"}}"#;
            write!(socket,"HTTP/1.1 422 Unprocessable Content\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",body.len(),body).unwrap();
        });
        let mut a = assignment();
        a.provider = "fal".into();
        a.model = "bytedance/seedance-2.5/text-to-video".into();
        assert!(matches!(
            DirectProvider::fixture(&format!("http://{address}/"), "fal").submit(&a),
            Err(Error::ProviderTerminal {
                reason: "provider_rejected",
                http_status: Some(422),
                provider_code: Some(code),
            }) if code == "Model.NotActivated"
        ));
        server.join().unwrap();
    }

    #[test]
    fn explicit_provider_terminal_status_is_not_treated_as_poll_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut socket, _) = listener.accept().unwrap();
            let mut raw = [0u8; 8192];
            let n = socket.read(&mut raw).unwrap();
            assert!(String::from_utf8_lossy(&raw[..n]).contains("/upstream/fal/status/"));
            let body = r#"{"status":"FAILED"}"#;
            write!(socket,"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",body.len(),body).unwrap();
        });
        let mut a = assignment();
        a.provider = "fal".into();
        a.model = "bytedance/seedance-2.5/text-to-video".into();
        assert!(matches!(
            DirectProvider::fixture(&format!("http://{address}/"), "fal").collect(
                &a,
                &ProviderSubmission {
                    reference: "provider-job".into(),
                    response: vec![],
                },
            ),
            Err(Error::ProviderTerminal {
                reason: "provider_failed",
                http_status: None,
                provider_code: None,
            })
        ));
        server.join().unwrap();
    }

    #[test]
    fn fal_completed_result_policy_422_is_a_terminal_failure() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for (path, status, body) in [
                (
                    "/upstream/fal/status/",
                    "200 OK",
                    r#"{"status":"COMPLETED"}"#,
                ),
                (
                    "/upstream/fal/result/",
                    "422 Unprocessable Content",
                    r#"{"detail":[{"type":"content_policy_violation"}]}"#,
                ),
            ] {
                let (mut socket, _) = listener.accept().unwrap();
                let mut raw = [0u8; 8192];
                let n = socket.read(&mut raw).unwrap();
                assert!(String::from_utf8_lossy(&raw[..n]).contains(path));
                write!(socket,"HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",body.len(),body).unwrap();
            }
        });
        let mut a = assignment();
        a.provider = "fal".into();
        a.model = "fal-ai/veo3.1".into();
        a.request["duration"] = serde_json::json!(4);
        a.request_hash = sha256(&canonical(&a.request).unwrap());
        assert!(matches!(
            DirectProvider::fixture(&format!("http://{address}/"), "fal").collect(
                &a,
                &ProviderSubmission {
                    reference: "provider-job".into(),
                    response: vec![],
                },
            ),
            Err(Error::ProviderTerminal {
                reason: "provider_failed",
                http_status: None,
                provider_code: None,
            })
        ));
        server.join().unwrap();
    }

    #[test]
    fn direct_provider_polls_native_terminal_forms_and_downloads_media() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let origin = format!("http://{address}/");
        let server = thread::spawn(move || {
            for _ in 0..9 {
                let (mut s, _) = listener.accept().unwrap();
                let mut raw = vec![0; 8192];
                let n = s.read(&mut raw).unwrap();
                let request = String::from_utf8_lossy(&raw[..n]);
                let body = if request.contains("/byteplus/status/") {
                    format!(
                        r#"{{"status":"succeeded","content":{{"video_url":"http://{address}/output/byte"}}}}"#
                    )
                } else if request.contains("/kling/status/") {
                    format!(
                        r#"{{"data":{{"task_status":"succeed","task_result":{{"videos":[{{"url":"http://{address}/output/kling"}}]}}}}}}"#
                    )
                } else if request.contains("/google-cloud/status/") {
                    r#"{"done":true,"response":{"videos":[{"gcsUri":"gs://fixture-bucket/a/b.mp4"}]}}"#.into()
                } else if request.contains("/fal/status/") {
                    r#"{"status":"COMPLETED","request_id":"fal-task"}"#.into()
                } else if request.contains("/fal/result/") {
                    format!(r#"{{"video":{{"url":"http://{address}/output/fal"}}}}"#)
                } else if request.contains("/output/") || request.contains("/google-cloud/output") {
                    let bytes = b"fixture-media";
                    write!(
                        s,
                        "HTTP/1.1 200 OK\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
                        bytes.len()
                    )
                    .unwrap();
                    s.write_all(bytes).unwrap();
                    continue;
                } else {
                    panic!("unexpected {request}")
                };
                write!(s,"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",body.len(),body).unwrap();
            }
        });
        let cases = [
            ("byteplus", "dreamina-seedance-2-5-260628", "byte-task"),
            ("kling", "kling-v2-6", "kling-task"),
            (
                "google-cloud",
                "veo-3.1-generate-001",
                "operations/veo-task",
            ),
            ("fal", "bytedance/seedance-2.5/text-to-video", "fal-task"),
        ];
        for (provider, model, reference) in cases {
            let mut a = assignment();
            a.provider = provider.into();
            a.model = model.into();
            a.expires_at = now_ms() + 10_000;
            let direct = DirectProvider::fixture(&origin, provider);
            assert_eq!(
                direct
                    .collect(
                        &a,
                        &ProviderSubmission {
                            reference: reference.into(),
                            response: vec![]
                        }
                    )
                    .unwrap(),
                b"fixture-media"
            );
        }
        server.join().unwrap();
    }

    #[test]
    fn image_to_video_uses_only_encrypted_inline_bytes_per_provider() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for _ in 0..5 {
                let (mut s, _) = listener.accept().unwrap();
                let mut raw = vec![0; 8192];
                let n = s.read(&mut raw).unwrap();
                let req = String::from_utf8_lossy(&raw[..n]);
                let body = &req[req.find("\r\n\r\n").unwrap() + 4..];
                if req.contains("/byteplus/") {
                    assert!(body.contains("data:image/png;base64,"));
                } else if req.contains("/kling/") {
                    assert!(body.contains("\"image\":\"aW1n\""));
                } else if req.contains("/google-cloud/") {
                    assert!(body.contains("bytesBase64Encoded"));
                    assert!(body.contains("\"image\":{\"bytesBase64Encoded\""));
                    assert!(!body.contains("referenceImages"));
                } else {
                    assert!(body.contains("\"image_url\":\"data:image/png;base64,aW1n\""));
                }
                let response = if req.contains("byteplus") {
                    r#"{"id":"x"}"#
                } else if req.contains("kling") {
                    r#"{"data":{"task_id":"x"}}"#
                } else if req.contains("google-cloud") {
                    r#"{"name":"x"}"#
                } else {
                    r#"{"request_id":"x"}"#
                };
                write!(s,"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",response.len(),response).unwrap();
            }
        });
        let cases = [
            ("byteplus", "dreamina-seedance-2-5-260628"),
            ("kling", "kling-v2-6"),
            ("google-cloud", "veo-3.1-generate-001"),
            ("fal", "bytedance/seedance-2.0/fast/image-to-video"),
            ("fal", "fal-ai/veo3.1/image-to-video"),
        ];
        for (provider, model) in cases {
            let mut a = assignment();
            a.provider = provider.into();
            a.model = model.into();
            a.modality = "image-to-video".into();
            if model.starts_with("fal-ai/veo3.1") {
                a.request["duration"] = serde_json::json!(4);
                a.request_hash = sha256(&canonical(&a.request).unwrap());
            }
            a.inputs = vec![AssignmentInput {
                id: "frame".into(),
                mime: "image/png".into(),
                sha256: sha256(b"img"),
                bytes_base64: B64.encode(b"img"),
            }];
            a.request["imageId"] = serde_json::json!("frame");
            a.request_hash = sha256(&canonical(&a.request).unwrap());
            DirectProvider::fixture(&format!("http://{address}/"), provider)
                .submit(&a)
                .unwrap();
        }
        server.join().unwrap();
    }

    #[test]
    fn restart_resumes_confirmed_reference_without_second_submit() {
        struct ResumeOnly;
        impl Provider for ResumeOnly {
            fn submit(&self, _: &Assignment) -> Result<ProviderSubmission> {
                Err(Error::Provider("must_not_resubmit".into()))
            }
            fn collect(&self, _: &Assignment, submission: &ProviderSubmission) -> Result<Vec<u8>> {
                assert_eq!(submission.reference, "paid-ref");
                Ok(b"media".to_vec())
            }
        }
        let d = tempdir().unwrap();
        let keys = Keys::load_or_create(&d.path().join("key")).unwrap();
        let state_path = d.path().join("state.json");
        let mut a = assignment();
        // The paid provider reference predates the current attestation
        // session. Completion must use the renewed session, with no re-submit.
        a.session_id = "prior-session".into();
        let id = format!("{}:{}", a.job_id, a.attempt_id);
        let mut state = DurableState {
            session_id: Some("session".into()),
            ..Default::default()
        };
        state.accepted.insert(
            id,
            AcceptedAttempt {
                assignment: a,
                provider_ref: Some("paid-ref".into()),
                provider_response: B64URL.encode(b"accepted"),
                provider_response_hash: Some(sha256(b"accepted")),
                state: "accepted".into(),
                terminal_failure: None,
            },
        );
        state.save(&state_path).unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            for (path, body) in [
                ("/v1/api-miners/start", r#"{"state":"started"}"#),
                ("/v1/api-miners/complete", r#"{"ok":true}"#),
            ] {
                let (mut s, _) = listener.accept().unwrap();
                let mut raw = vec![0; 8192];
                let n = s.read(&mut raw).unwrap();
                let request = String::from_utf8_lossy(&raw[..n]);
                assert!(request.contains(path));
                if path.ends_with("/complete") {
                    let body = &request[request.find("\r\n\r\n").unwrap() + 4..];
                    let posted: Value = serde_json::from_str(body).unwrap();
                    assert_eq!(posted["body"]["sessionId"], "session");
                    assert_eq!(posted["body"]["payload"]["receipt"]["sessionId"], "session");
                }
                write!(s,"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",body.len(),body).unwrap();
            }
        });
        let coordinator = Coordinator::test_new(
            &format!("http://{address}/"),
            "worker".into(),
            "miner".into(),
            keys.clone(),
        )
        .unwrap();
        let worker = Worker {
            worker_id: "worker".into(),
            miner_id: "miner".into(),
            keys,
            attestation: attestation(),
            state_path,
            provider: ResumeOnly,
        };
        assert!(worker.run_once(&coordinator).unwrap());
        server.join().unwrap();
    }

    #[test]
    fn new_runtime_recovers_gateway_canonical_ref_without_resubmission() {
        struct RecoverOnly;
        impl Provider for RecoverOnly {
            fn submit(&self, _: &Assignment) -> Result<ProviderSubmission> {
                panic!("must_not_submit")
            }
            fn collect(&self, _: &Assignment, submission: &ProviderSubmission) -> Result<Vec<u8>> {
                assert_eq!(submission.reference, "canonical-ref");
                Ok(b"media".to_vec())
            }
        }
        let d = tempdir().unwrap();
        let runtime = Keys::fresh();
        let canonical = assignment();
        let envelope = encrypted(&runtime, "session", &canonical);
        let mut hint = canonical.clone();
        hint.request["prompt"] = serde_json::json!("tampered disk hint");
        let state_path = d.path().join("state.json");
        DurableState {
            session_id: Some("session".into()),
            accepted: [(
                format!("{}:{}", hint.job_id, hint.attempt_id),
                AcceptedAttempt {
                    assignment: hint,
                    provider_ref: Some("tampered-ref".into()),
                    provider_response: B64URL.encode(b"tampered"),
                    provider_response_hash: Some(sha256(b"tampered")),
                    state: "accepted".into(),
                    terminal_failure: None,
                },
            )]
            .into_iter()
            .collect(),
            paused: false,
        }
        .save(&state_path)
        .unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let response_hash = sha256(b"canonical-submit");
        let server = thread::spawn(move || {
            for (path, body) in [
                (
                    "/v1/api-miners/recover",
                    serde_json::json!({"state":"started","providerRef":"canonical-ref","providerResponseHash":response_hash,"encryptedAssignment":envelope}),
                ),
                (
                    "/v1/api-miners/start",
                    serde_json::json!({"state":"started"}),
                ),
                ("/v1/api-miners/complete", serde_json::json!({"ok":true})),
            ] {
                let (mut s, _) = listener.accept().unwrap();
                let mut raw = vec![0; 16384];
                let n = s.read(&mut raw).unwrap();
                assert!(String::from_utf8_lossy(&raw[..n]).contains(path));
                let body = body.to_string();
                write!(s,"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",body.len(),body).unwrap();
            }
        });
        let coordinator = Coordinator::test_new(
            &format!("http://{address}/"),
            "worker".into(),
            "miner".into(),
            runtime.clone(),
        )
        .unwrap();
        let worker = Worker {
            worker_id: "worker".into(),
            miner_id: "miner".into(),
            keys: runtime,
            attestation: attestation(),
            state_path: state_path.clone(),
            provider: RecoverOnly,
        };
        worker.recover_new_runtime(&coordinator).unwrap();
        assert!(worker.run_once(&coordinator).unwrap());
        assert_eq!(
            DurableState::load(&state_path).unwrap().accepted["job:attempt"].state,
            "completed"
        );
        server.join().unwrap();
    }

    #[test]
    #[ignore = "requires ATTELIER_API_MINER_NODE_RUNNER and the separately built offline Node fixture"]
    fn rust_worker_completes_real_node_gateway_encrypted_lifecycle() {
        use std::{
            io::BufRead,
            process::{Command, Stdio},
        };
        struct ChildGuard(std::process::Child);
        impl Drop for ChildGuard {
            fn drop(&mut self) {
                let _ = self.0.kill();
                let _ = self.0.wait();
            }
        }
        let runner = std::env::var("ATTELIER_API_MINER_NODE_RUNNER")
            .expect("set ATTELIER_API_MINER_NODE_RUNNER to api-miner-node-runner.mjs");
        let executable = |variable: &str, name: &str| -> std::path::PathBuf {
            std::env::var_os(variable)
                .map(std::path::PathBuf::from)
                .filter(|path| path.is_file())
                .or_else(|| {
                    std::env::var_os("PATH").and_then(|paths| {
                        std::env::split_paths(&paths)
                            .map(|dir| dir.join(name))
                            .find(|path| path.is_file())
                    })
                })
                .unwrap_or_else(|| panic!("set {variable} or put {name} on PATH"))
        };
        let ffmpeg = executable("FFMPEG_PATH", "ffmpeg");
        let ffprobe = executable("FFPROBE_PATH", "ffprobe");
        let d = tempdir().unwrap();
        let keys = Keys::load_or_create(&d.path().join("key")).unwrap();
        let control = ControlKey::load_or_create(&d.path().join("control.ed25519")).unwrap();
        let mut child = ChildGuard(
            Command::new("node")
                .arg(runner)
                .env("MINER_CONTROL_PUBLIC_KEY", control.public())
                .env("MINER_RUNTIME_SIGNING_PUBLIC_KEY", keys.public())
                .env(
                    "MINER_RUNTIME_ENCRYPTION_PUBLIC_KEY",
                    keys.encryption_public(),
                )
                .env("FFMPEG_PATH", ffmpeg)
                .env("FFPROBE_PATH", ffprobe)
                .stdout(Stdio::piped())
                .spawn()
                .unwrap(),
        );
        let mut lines = std::io::BufReader::new(child.0.stdout.take().unwrap()).lines();
        let ready: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
        let state = d.path().join("state.json");
        let coordinator = Coordinator::test_new(
            ready["url"].as_str().unwrap(),
            ready["workerId"].as_str().unwrap().into(),
            ready["minerId"].as_str().unwrap().into(),
            keys.clone(),
        )
        .unwrap();
        let nonce = coordinator.challenge(&control).unwrap();
        let quote = serde_json::json!({"quote":"A".repeat(1400),"event_log":"{}"});
        let report_data = attestation_report_data_v2(
            &nonce,
            &control.public(),
            &keys.public(),
            &keys.encryption_public(),
        )
        .unwrap();
        let (attestation, session) = coordinator
            .attest(&control, &nonce, &report_data, &quote)
            .unwrap();
        DurableState {
            session_id: Some(session),
            ..Default::default()
        }
        .save(&state)
        .unwrap();
        let coordinator =
            coordinator.with_session(DurableState::load(&state).unwrap().session_id.unwrap());
        coordinator
            .offer(serde_json::json!({
                "provider": ready["provider"], "model": ready["model"], "modality": "text-to-video",
                "quoteMicrousd": 90, "expiresAt": now_ms() + 30_000
            }))
            .unwrap();
        struct Media(Vec<u8>);
        impl Provider for Media {
            fn submit(&self, _: &Assignment) -> Result<ProviderSubmission> {
                Ok(ProviderSubmission {
                    reference: "fixture-provider".into(),
                    response: b"fixture-response".to_vec(),
                })
            }
            fn collect(&self, _: &Assignment, _: &ProviderSubmission) -> Result<Vec<u8>> {
                Ok(self.0.clone())
            }
        }
        let media = std::fs::read(ready["mediaPath"].as_str().unwrap()).unwrap();
        let worker = Worker {
            worker_id: ready["workerId"].as_str().unwrap().into(),
            miner_id: ready["minerId"].as_str().unwrap().into(),
            keys,
            attestation,
            state_path: state,
            provider: Media(media),
        };
        assert!(worker.run_once(&coordinator).unwrap());
        let line = lines.next().unwrap().unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(&line).unwrap()["event"],
            "custody_verified"
        );
    }
}
