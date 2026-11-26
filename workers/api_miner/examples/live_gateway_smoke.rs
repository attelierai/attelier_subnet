//! Operator end-to-end smoke harness: exercises the real multi-provider
//! `DirectProvider::from_env_auto()` + `Worker` + `Coordinator` against a local
//! test gateway without requiring a live Phala dstack socket.
use attelier_api_miner::{
    attestation_report_data_v2, ControlKey, Coordinator, DirectProvider, DurableState, Keys, Worker,
};
use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from(env::var("ATTELIER_MINER_STATE_DIR")?);
    let state = root.join("state.json");
    let key_path = root.join("worker.ed25519");
    let coordinator_url = env::var("ATTELIER_COORDINATOR_URL")?;
    let worker_id = env::var("ATTELIER_WORKER_ID")?;
    let miner_id = env::var("ATTELIER_MINER_ID")?;
    let max_jobs: usize = env::var("ATTELIER_SMOKE_MAX_JOBS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let control_key = ControlKey::load_or_create(&key_path)?;
    if env::args().nth(1).as_deref() == Some("identity") {
        println!("{}", control_key.public());
        return Ok(());
    }

    let provider = DirectProvider::from_env_auto()?;
    println!("enabled_providers={}", provider.enabled_providers().join(","));

    let runtime = Keys::fresh();
    let control = if coordinator_url.starts_with("http://127.0.0.1") {
        Coordinator::new_loopback(&coordinator_url, worker_id.clone(), miner_id.clone(), runtime.clone())?
    } else {
        Coordinator::new(&coordinator_url, worker_id.clone(), miner_id.clone(), runtime.clone())?
    };
    let nonce = control.challenge(&control_key)?;
    let report_data = attestation_report_data_v2(
        &nonce,
        &control_key.public(),
        &runtime.public(),
        &runtime.encryption_public(),
    )?;
    let quote = serde_json::json!({
        "quote": "A".repeat(1400),
        "event_log": "{}"
    });
    let (attestation, session_id) = control.attest(&control_key, &nonce, &report_data, &quote)?;
    DurableState {
        session_id: Some(session_id.clone()),
        ..Default::default()
    }
    .save(&state)?;
    let control = control.with_session(session_id);

    let worker = Worker {
        worker_id,
        miner_id,
        keys: runtime,
        attestation,
        state_path: state,
        provider,
    };

    if env::var("ATTELIER_SMOKE_WAIT_STDIN").ok().as_deref() == Some("1") {
        println!("ATTESTED");
        let mut line = String::new();
        let _ = std::io::stdin().read_line(&mut line);
    }

    let mut processed = 0usize;
    while processed < max_jobs {
        match worker.run_once(&control) {
            Ok(true) => {
                processed += 1;
                println!("smoke_processed={processed}");
            }
            Ok(false) => break,
            Err(err) => {
                eprintln!("smoke_job_error={err}");
                processed += 1;
            }
        }
    }
    println!("smoke_done={processed}");
    Ok(())
}
