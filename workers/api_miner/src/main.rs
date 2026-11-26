use attelier_api_miner::{
    attestation_report_data_v2, enrollment_evidence, pause_requested, phala_quote, phala_vm_config,
    Attestation, ControlKey, Coordinator, DirectProvider, DurableState, Keys, Worker,
};
use fs2::FileExt;
use std::{
    env, fs,
    path::{Path, PathBuf},
    thread,
    time::Duration,
};

fn usage() {
    eprintln!("usage: attelier-api-miner doctor|identity|evidence|attest|pause|resume|status|run-once|run");
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command = env::args().nth(1).unwrap_or_else(|| "help".into());
    let root =
        PathBuf::from(env::var("ATTELIER_MINER_STATE_DIR").unwrap_or_else(|_| "state".into()));
    let state = root.join("state.json");
    let key = root.join("worker.ed25519");
    match command.as_str() {
        "doctor" => {
            let _ = ControlKey::load_or_create(&root.join("worker.ed25519"))?;
            for name in [
                "BYTEPLUS_API_KEY",
                "KLING_ACCESS_KEY",
                "KLING_SECRET_KEY",
                "GOOGLE_CLOUD_ACCESS_TOKEN",
                "FAL_KEY",
                "ATLASCLOUD_API_KEY",
            ] {
                let configured = env::var(name)
                    .map(|v| !v.trim().is_empty())
                    .unwrap_or(false);
                println!(
                    "{name}={}",
                    if configured { "configured" } else { "missing" }
                );
            }
            println!("state={}", root.display());
        }
        "identity" => {
            let keys = ControlKey::load_or_create(&key)?;
            println!(
                "workerId={}",
                env::var("ATTELIER_WORKER_ID").unwrap_or_else(|_| "unconfigured".into())
            );
            println!(
                "minerId={}",
                env::var("ATTELIER_MINER_ID").unwrap_or_else(|_| "unconfigured".into())
            );
            println!("signingPublicKey={}", keys.public());
        }
        "evidence" => {
            let _lock = ProcessLock::acquire(&root)?;
            enrollment(&key)?;
        }
        "pause" | "resume" => {
            // Do not rewrite state.json here: a running worker may be flushing
            // an accepted paid request.  The worker checks this atomically
            // created control file both before claiming and before submit.
            set_pause(&root, command == "pause")?;
            println!(
                "worker={}",
                if command == "pause" {
                    "paused"
                } else {
                    "ready"
                }
            );
        }
        "status" => {
            let current = DurableState::load(&state)?;
            println!(
                "paused={} accepted_attempts={}",
                pause_requested(&state, &current),
                current.accepted.len()
            );
        }
        "attest" => {
            let _lock = ProcessLock::acquire(&root)?;
            attest(&root, &state, &key, &Keys::fresh())?
        }
        "run-once" | "run" => {
            let _lock = ProcessLock::acquire(&root)?;
            let continuous = command == "run";
            let runtime = Keys::fresh();
            let offers =
                offers_from_config(&env::var("ATTELIER_MINER_OFFERS").unwrap_or_default())?;
            let max_concurrency = max_concurrency_from_env()?;
            let mut renew_offers_at = 0;
            let mut first_attestation = true;
            let mut recover_runtime = true;
            loop {
                let current = DurableState::load(&state)?;
                if pause_requested(&state, &current) {
                    if continuous {
                        thread::sleep(Duration::from_secs(2));
                        continue;
                    }
                    return Err("worker_paused".into());
                }
                let attestation_path = env::var("ATTELIER_TEE_ATTESTATION_FILE")
                    .unwrap_or_else(|_| root.join("attestation.json").display().to_string());
                let renew = first_attestation
                    || fs::read(&attestation_path)
                        .ok()
                        .and_then(|v| serde_json::from_slice::<Attestation>(&v).ok())
                        .is_none_or(|a| a.expires_at <= attelier_api_miner::now_ms() + 60_000);
                if renew {
                    attest(&root, &state, &key, &runtime)?;
                    first_attestation = false;
                    renew_offers_at = 0;
                }
                match run_once(
                    &root,
                    &state,
                    &runtime,
                    recover_runtime,
                    &offers,
                    &mut renew_offers_at,
                    if continuous { max_concurrency } else { 1 },
                ) {
                    Ok(()) => recover_runtime = false,
                    Err(error) if continuous => eprintln!("run_error={error}"),
                    Err(error) => return Err(error),
                }
                if !continuous {
                    break;
                }
                thread::sleep(Duration::from_secs(2));
            }
        }
        _ => usage(),
    }
    Ok(())
}

/// Advisory OS lock which is released if the process dies. The file is only a
/// diagnostic record, never a stale PID gate that blocks crash recovery.
struct ProcessLock(fs::File);
impl ProcessLock {
    fn acquire(root: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        fs::create_dir_all(root)?;
        let path = root.join("worker.lock");
        let mut file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)?;
        file.try_lock_exclusive()
            .map_err(|_| "worker_already_running")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
        }
        use std::io::Write;
        file.set_len(0)?;
        writeln!(file, "pid={}", std::process::id())?;
        file.sync_all()?;
        Ok(Self(file))
    }
}
impl Drop for ProcessLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
}

fn set_pause(root: &Path, paused: bool) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(root)?;
    let pause = root.join("pause.flag");
    let resume = root.join("resume.flag");
    if paused {
        let _ = fs::remove_file(&resume);
        if !pause.exists() {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&pause)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&pause, fs::Permissions::from_mode(0o600))?;
            }
            use std::io::Write;
            writeln!(file, "pausedAt={}", attelier_api_miner::now_ms())?;
            file.sync_all()?;
        }
    } else {
        let _ = fs::remove_file(&pause);
        if !resume.exists() {
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&resume)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&resume, fs::Permissions::from_mode(0o600))?;
            }
            use std::io::Write;
            writeln!(file, "resumedAt={}", attelier_api_miner::now_ms())?;
            file.sync_all()?;
        }
    }
    fs::File::open(root)?.sync_all()?;
    Ok(())
}

fn attest(
    root: &PathBuf,
    state: &PathBuf,
    key: &PathBuf,
    runtime: &Keys,
) -> Result<(), Box<dyn std::error::Error>> {
    let coordinator =
        env::var("ATTELIER_COORDINATOR_URL").map_err(|_| "ATTELIER_COORDINATOR_URL is required")?;
    let worker_id = env::var("ATTELIER_WORKER_ID").map_err(|_| "ATTELIER_WORKER_ID is required")?;
    let miner_id = env::var("ATTELIER_MINER_ID").map_err(|_| "ATTELIER_MINER_ID is required")?;
    let control_key = ControlKey::load_or_create(key)?;
    let control = Coordinator::new(&coordinator, worker_id, miner_id, runtime.clone())?;
    let nonce = control.challenge(&control_key)?;
    let report_data = attestation_report_data_v2(
        &nonce,
        &control_key.public(),
        &runtime.public(),
        &runtime.encryption_public(),
    )?;
    let quote = phala_quote(&report_data)?;
    let (attestation, session_id) = control.attest(&control_key, &nonce, &report_data, &quote)?;
    fs::write(
        root.join("attestation.json"),
        serde_json::to_vec(&attestation)?,
    )?;
    let mut current = DurableState::load(&state)?;
    current.session_id = Some(session_id);
    current.save(&state)?;
    println!("attestation=verified expiresAt={}", attestation.expires_at);
    Ok(())
}

/// Emits public evidence only. The CVM bootstrap transports it to the private
/// enrollment collector; the worker neither self-verifies nor self-admits.
fn enrollment(key: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let challenge = env::var("ATTELIER_ENROLLMENT_CHALLENGE")
        .map_err(|_| "ATTELIER_ENROLLMENT_CHALLENGE is required")?;
    let worker_id = env::var("ATTELIER_WORKER_ID").map_err(|_| "ATTELIER_WORKER_ID is required")?;
    let miner_id = env::var("ATTELIER_MINER_ID").map_err(|_| "ATTELIER_MINER_ID is required")?;
    let control = ControlKey::load_or_create(key)?;
    let runtime = Keys::fresh();
    let report_data = attestation_report_data_v2(
        &challenge,
        &control.public(),
        &runtime.public(),
        &runtime.encryption_public(),
    )?;
    let quote = phala_quote(&report_data)?;
    let vm_config = phala_vm_config()?;
    println!(
        "{}",
        serde_json::to_string(&enrollment_evidence(
            &miner_id,
            &worker_id,
            &control.public(),
            &runtime.public(),
            &runtime.encryption_public(),
            &challenge,
            &quote,
            &vm_config,
        )?)?
    );
    Ok(())
}

fn run_once(
    root: &PathBuf,
    state: &PathBuf,
    runtime: &Keys,
    recover_runtime: bool,
    offers: &[Offer],
    renew_offers_at: &mut i64,
    max_concurrency: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let coordinator =
        env::var("ATTELIER_COORDINATOR_URL").map_err(|_| "ATTELIER_COORDINATOR_URL is required")?;
    let worker_id = env::var("ATTELIER_WORKER_ID").map_err(|_| "ATTELIER_WORKER_ID is required")?;
    let miner_id = env::var("ATTELIER_MINER_ID").map_err(|_| "ATTELIER_MINER_ID is required")?;
    let provider_name =
        env::var("ATTELIER_PROVIDER").map_err(|_| "ATTELIER_PROVIDER is required")?;
    let attestation_path = env::var("ATTELIER_TEE_ATTESTATION_FILE")
        .unwrap_or_else(|_| root.join("attestation.json").display().to_string());
    let attestation: Attestation = serde_json::from_slice(&fs::read(attestation_path)?)?;
    let session_id = DurableState::load(state)?
        .session_id
        .ok_or("attestation_required")?;
    let control = Coordinator::new(
        &coordinator,
        worker_id.clone(),
        miner_id.clone(),
        runtime.clone(),
    )?
    .with_session(session_id);
    let provider = DirectProvider::from_env(&provider_name)?;
    let worker = Worker {
        worker_id,
        miner_id,
        keys: runtime.clone(),
        attestation,
        state_path: state.to_path_buf(),
        provider,
    };
    if recover_runtime {
        worker.recover_new_runtime(&control)?;
    }
    let now = attelier_api_miner::now_ms();
    if !offers.is_empty() && now >= *renew_offers_at {
        let expires_at = (now + 300_000).min(worker.attestation.expires_at);
        for offer in offers {
            let mut body = serde_json::Map::new();
            body.insert("provider".into(), serde_json::json!(provider_name));
            body.insert("model".into(), serde_json::json!(offer.model));
            body.insert("modality".into(), serde_json::json!(offer.modality));
            if let Some(flat) = offer.quote_microusd {
                body.insert("quoteMicrousd".into(), serde_json::json!(flat));
            }
            if let Some(per_sec) = offer.quote_microusd_per_second {
                body.insert("quoteMicrousdPerSecond".into(), serde_json::json!(per_sec));
            }
            if let Some(by_dur) = &offer.quotes_by_duration {
                body.insert("quotesByDuration".into(), serde_json::json!(by_dur));
            }
            body.insert("expiresAt".into(), serde_json::json!(expires_at));
            control.offer(serde_json::Value::Object(body))?;
        }
        *renew_offers_at = now + 120_000;
    }
    let claimed = worker.run_concurrent(&control, max_concurrency)?;
    println!("claimed={}", claimed > 0);
    if max_concurrency > 1 {
        println!("claimed_count={claimed}");
    }
    Ok(())
}

fn max_concurrency_from_env() -> Result<usize, Box<dyn std::error::Error>> {
    match env::var("ATTELIER_MAX_CONCURRENCY") {
        Ok(raw) if !raw.trim().is_empty() => {
            let value: usize = raw
                .trim()
                .parse()
                .map_err(|_| "invalid ATTELIER_MAX_CONCURRENCY")?;
            if !(1..=32).contains(&value) {
                return Err("ATTELIER_MAX_CONCURRENCY must be between 1 and 32".into());
            }
            Ok(value)
        }
        _ => Ok(1),
    }
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct Offer {
    model: String,
    modality: String,
    #[serde(default)]
    quote_microusd: Option<u64>,
    #[serde(default)]
    quote_microusd_per_second: Option<u64>,
    #[serde(default)]
    quotes_by_duration: Option<std::collections::BTreeMap<String, u64>>,
}

const MAX_JS_SAFE_INT: u64 = 9_007_199_254_740_991;

fn offers_from_config(raw: &str) -> Result<Vec<Offer>, Box<dyn std::error::Error>> {
    if raw.trim().is_empty() {
        return Ok(vec![]);
    }
    let offers: Vec<Offer> = serde_json::from_str(raw)?;
    let mut seen = std::collections::HashSet::new();
    for offer in &offers {
        let flat_ok = offer
            .quote_microusd
            .is_some_and(|v| (1..=MAX_JS_SAFE_INT).contains(&v));
        let per_sec_ok = offer
            .quote_microusd_per_second
            .is_some_and(|v| (1..=MAX_JS_SAFE_INT).contains(&v));
        let by_dur_ok = offer.quotes_by_duration.as_ref().is_some_and(|map| {
            !map.is_empty()
                && map.len() <= 30
                && map.iter().all(|(k, v)| {
                    k.parse::<u8>().is_ok_and(|d| (1..=30).contains(&d) && k == &d.to_string())
                        && (1..=MAX_JS_SAFE_INT).contains(v)
                })
        });
        if offer.model.is_empty()
            || offer.model.len() > 160
            || !matches!(
                offer.modality.as_str(),
                "text-to-video" | "image-to-video" | "text-to-image" | "image-to-image"
            )
            || (offer.quote_microusd.is_some() && !flat_ok)
            || (offer.quote_microusd_per_second.is_some() && !per_sec_ok)
            || (offer.quotes_by_duration.is_some() && !by_dur_ok)
            || !(flat_ok || per_sec_ok || by_dur_ok)
            || !seen.insert((&offer.model, &offer.modality))
        {
            return Err("invalid or duplicate miner offer".into());
        }
    }
    Ok(offers)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offers_from_config_accepts_duration_aware_and_flat_offers() {
        let parsed = offers_from_config(
            r#"[
              {"model":"dreamina-seedance-2-5-260628","modality":"text-to-video","quoteMicrousdPerSecond":20000,"quotesByDuration":{"5":95000,"10":180000}},
              {"model":"dreamina-seedance-2-5-260628","modality":"image-to-video","quoteMicrousd":100000},
              {"model":"fal-ai/nano-banana-pro","modality":"text-to-image","quoteMicrousd":90000},
              {"model":"fal-ai/nano-banana-pro/edit","modality":"image-to-image","quoteMicrousd":95000}
            ]"#,
        )
        .unwrap();
        assert_eq!(parsed.len(), 4);
        assert_eq!(parsed[0].quote_microusd_per_second, Some(20_000));
        assert_eq!(
            parsed[0]
                .quotes_by_duration
                .as_ref()
                .and_then(|m| m.get("5"))
                .copied(),
            Some(95_000)
        );
        assert_eq!(parsed[1].quote_microusd, Some(100_000));
    }

    #[test]
    fn offers_from_config_rejects_empty_or_invalid_duration_keys() {
        assert!(offers_from_config(
            r#"[{"model":"dreamina-seedance-2-5-260628","modality":"text-to-video"}]"#
        )
        .is_err());
        assert!(offers_from_config(
            r#"[{"model":"dreamina-seedance-2-5-260628","modality":"text-to-video","quotesByDuration":{"05":95000}}]"#
        )
        .is_err());
    }
}
