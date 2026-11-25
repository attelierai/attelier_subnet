//! Fail-closed worker protocol for API-backed Attelier miners.
//!
//! The gateway owns routing. This crate accepts only a fresh, one-use grant and
//! persists it before an upstream request, so an uncertain paid submission is
//! reconciled rather than replayed.

mod coordinator;
mod crypto;
mod phala;
mod protocol;
mod providers;
mod worker;

pub use coordinator::Coordinator;
pub use crypto::{
    canonical, now_ms, sha256, verify, verify_canonical, ControlKey, EncryptedAssignment, Keys,
};
pub use phala::{
    attestation_report_data_v2, enrollment_evidence, phala_quote, phala_vm_config,
};
pub use protocol::{
    pause_requested, AcceptedAttempt, Assignment, AssignmentInput, Attestation, DurableState,
    Receipt, TerminalFailure,
};
pub use providers::{DirectProvider, Provider, ProviderSubmission};
pub use worker::Worker;

use thiserror::Error;

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

#[cfg(test)]
mod tests {
    use super::*;
    use aes_gcm::{
        aead::{Aead, KeyInit},
        Aes256Gcm, Nonce,
    };
    use base64::{
        engine::general_purpose::{STANDARD as B64, URL_SAFE_NO_PAD as B64URL},
        Engine,
    };
    use hmac::{Hmac, Mac};
    use rand::rngs::OsRng;
    use serde_json::Value;
    use sha2::{Digest, Sha256};
    use std::fs;
    use std::sync::Mutex;
    use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret};
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

    #[test]
    fn worker_run_concurrent_executes_multiple_claimed_assignments_in_parallel() {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };
        let d = tempdir().unwrap();
        let state_path = d.path().join("state.json");
        let keys = Keys::fresh();
        let now = now_ms();
        let make_assignment = |idx: usize| {
            let mut a = assignment();
            a.job_id = format!("job-conc-{idx}");
            a.attempt_id = format!("att-conc-{idx}");
            a.miner_id = "m1".into();
            a.worker_id = "w1".into();
            a.session_id = "session".into();
            a.model = "dreamina-seedance-2-5-260628".into();
            a.grant_nonce = format!("nonce-conc-{idx}");
            a.expires_at = now + 60_000;
            a
        };
        let mut envelopes = vec![
            encrypted(&keys, "session", &make_assignment(1)),
            encrypted(&keys, "session", &make_assignment(2)),
        ];
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let completed = Arc::new(AtomicUsize::new(0));
        let completed_srv = completed.clone();
        let server = thread::spawn(move || {
            let mut served = 0usize;
            while served < 10 {
                let (mut stream, _) = listener.accept().unwrap();
                let mut buf = vec![0u8; 65_536];
                let n = stream.read(&mut buf).unwrap();
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let first = req.lines().next().unwrap_or("");
                let body = if first.contains("/v1/api-miners/claim") {
                    if let Some(env) = envelopes.pop() {
                        serde_json::json!({"state":"assigned","encryptedAssignment":env})
                    } else {
                        serde_json::json!({"state":"idle"})
                    }
                } else if first.contains("/v1/api-miners/start") {
                    serde_json::json!({"state":"started"})
                } else if first.contains("/v1/api-miners/complete") {
                    completed_srv.fetch_add(1, Ordering::SeqCst);
                    serde_json::json!({"status":"accepted"})
                } else {
                    serde_json::json!({"ok":true})
                };
                let bytes = body.to_string();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    bytes.len(),
                    bytes
                )
                .unwrap();
                served += 1;
                if completed_srv.load(Ordering::SeqCst) == 2 && envelopes.is_empty() {
                    break;
                }
            }
        });
        DurableState {
            session_id: Some("session".into()),
            ..Default::default()
        }
        .save(&state_path)
        .unwrap();
        let coordinator =
            Coordinator::test_new(&format!("http://{address}/"), "w1".into(), "m1".into(), keys.clone())
                .unwrap();
        struct ParallelProvider {
            in_flight: AtomicUsize,
            peak: AtomicUsize,
        }
        impl Provider for ParallelProvider {
            fn submit(&self, a: &Assignment) -> Result<ProviderSubmission> {
                let cur = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                self.peak.fetch_max(cur, Ordering::SeqCst);
                thread::sleep(std::time::Duration::from_millis(80));
                self.in_flight.fetch_sub(1, Ordering::SeqCst);
                Ok(ProviderSubmission {
                    reference: format!("ref-{}", a.job_id),
                    response: format!("resp-{}", a.job_id).into_bytes(),
                })
            }
            fn collect(&self, _: &Assignment, _: &ProviderSubmission) -> Result<Vec<u8>> {
                Ok(b"mp4-bytes".to_vec())
            }
        }
        let worker = Worker {
            worker_id: "w1".into(),
            miner_id: "m1".into(),
            keys,
            attestation: Attestation {
                measurement: "a".repeat(64),
                issued_at: now - 1000,
                expires_at: now + 60_000,
                status: "verified".into(),
            },
            state_path: state_path.clone(),
            provider: ParallelProvider {
                in_flight: AtomicUsize::new(0),
                peak: AtomicUsize::new(0),
            },
        };
        let count = worker.run_concurrent(&coordinator, 2).unwrap();
        server.join().unwrap();
        assert_eq!(count, 2);
        assert_eq!(completed.load(Ordering::SeqCst), 2);
        assert_eq!(worker.provider.peak.load(Ordering::SeqCst), 2);
        let final_state = DurableState::load(&state_path).unwrap();
        assert_eq!(final_state.accepted.len(), 2);
        assert!(final_state.accepted.values().all(|row| row.state == "completed"));
    }
}
