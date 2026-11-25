use crate::{Error, Result};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64URL, Engine};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use rand::rngs::OsRng;
use serde::Deserialize;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};

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
pub(crate) fn domain(name: &str, value: &Value) -> Result<Vec<u8>> {
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
    pub(crate) encryption: StaticSecret,
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
