use crate::{sha256, Error, Result};
use serde_json::Value;
use std::process::Command;

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
