use crate::{
    pause_requested, sha256, AcceptedAttempt, Assignment, Attestation, Coordinator, DurableState,
    Error, Keys, Provider, ProviderSubmission, Receipt, Result, TerminalFailure,
};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD as B64URL, Engine};
use serde_json::Value;
use std::{path::PathBuf, sync::Mutex};

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

    fn execute_claimed(
        &self,
        coordinator: &Coordinator,
        assignment: Assignment,
        state_lock: &Mutex<()>,
    ) -> Result<()> {
        assignment.validate(&self.worker_id, &self.miner_id)?;
        self.attestation.validate()?;
        let id = format!("{}:{}", assignment.job_id, assignment.attempt_id);
        {
            let _guard = state_lock
                .lock()
                .map_err(|_| Error::Protocol("state_lock_poisoned"))?;
            let mut state = DurableState::load(&self.state_path)?;
            if pause_requested(&self.state_path, &state) {
                return Err(Error::Protocol("worker_paused"));
            }
            if state.accepted.contains_key(&id) {
                drop(_guard);
                coordinator.unknown(&assignment, "attempt_already_accepted_reconcile")?;
                return Ok(());
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
        }
        let submission = match self.provider.submit(&assignment) {
            Ok(value) => value,
            Err(Error::ProviderTerminal {
                reason,
                http_status,
                provider_code,
            }) => {
                let _guard = state_lock
                    .lock()
                    .map_err(|_| Error::Protocol("state_lock_poisoned"))?;
                let mut state = DurableState::load(&self.state_path)?;
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
                return Ok(());
            }
            Err(error) => {
                coordinator.unknown(&assignment, "provider_submission_uncertain")?;
                return Err(error);
            }
        };
        let encoded_response = B64URL.encode(&submission.response);
        let response_hash = sha256(&submission.response);
        {
            let _guard = state_lock
                .lock()
                .map_err(|_| Error::Protocol("state_lock_poisoned"))?;
            let mut state = DurableState::load(&self.state_path)?;
            state.accepted.insert(
                id.clone(),
                AcceptedAttempt {
                    assignment: assignment.clone(),
                    provider_ref: Some(submission.reference.clone()),
                    provider_response: encoded_response.clone(),
                    provider_response_hash: Some(response_hash.clone()),
                    state: "accepted".into(),
                    terminal_failure: None,
                },
            );
            state.save(&self.state_path)?;
        }
        coordinator.start(&assignment, &submission.reference, &response_hash)?;
        let media = match self.provider.collect(&assignment, &submission) {
            Ok(value) => value,
            Err(Error::ProviderTerminal {
                reason,
                http_status,
                provider_code,
            }) => {
                let _guard = state_lock
                    .lock()
                    .map_err(|_| Error::Protocol("state_lock_poisoned"))?;
                let mut state = DurableState::load(&self.state_path)?;
                self.record_terminal_failure(
                    coordinator,
                    &mut state,
                    id,
                    assignment,
                    Some(submission.reference),
                    encoded_response,
                    reason,
                    http_status,
                    provider_code,
                )?;
                return Ok(());
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
            response_hash,
            &media,
            &self.keys,
        )?;
        receipt.verify_with(&self.keys.public())?;
        coordinator.complete(&receipt, &media)?;
        {
            let _guard = state_lock
                .lock()
                .map_err(|_| Error::Protocol("state_lock_poisoned"))?;
            let mut state = DurableState::load(&self.state_path)?;
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
        }
        Ok(())
    }

    /// Claim and execute up to `max_concurrency` assignments concurrently, serializing
    /// durable state transitions while overlapping upstream provider network I/O.
    pub fn run_concurrent(&self, coordinator: &Coordinator, max_concurrency: usize) -> Result<usize>
    where
        P: Sync,
    {
        let limit = max_concurrency.clamp(1, 32);
        if limit == 1 {
            return Ok(usize::from(self.run_once(coordinator)?));
        }
        let mut state_before = DurableState::load(&self.state_path)?;
        if pause_requested(&self.state_path, &state_before) {
            return Err(Error::Protocol("worker_paused"));
        }
        let mut resumed = 0usize;
        while self.resume(coordinator, &mut state_before)? {
            resumed += 1;
            if resumed >= limit {
                return Ok(resumed);
            }
        }
        let _session_id = state_before
            .session_id
            .as_deref()
            .ok_or(Error::Protocol("attestation_required"))?
            .to_owned();
        let mut claimed: Vec<Assignment> = Vec::with_capacity(limit);
        let mut claimed_ids: Vec<String> = Vec::with_capacity(limit);
        for _ in 0..limit {
            match coordinator.claim_excluding(&claimed_ids)? {
                Some(assignment) => {
                    if claimed_ids.iter().any(|id| id == &assignment.job_id) {
                        break;
                    }
                    claimed_ids.push(assignment.job_id.clone());
                    claimed.push(assignment);
                }
                None => break,
            }
        }
        if claimed.is_empty() {
            return Ok(resumed);
        }
        let state_lock = Mutex::new(());
        let count = claimed.len();
        if count == 1 {
            self.execute_claimed(coordinator, claimed.into_iter().next().unwrap(), &state_lock)?;
            return Ok(resumed + 1);
        }
        let mut first_error: Option<Error> = None;
        std::thread::scope(|scope| {
            let handles: Vec<_> = claimed
                .into_iter()
                .map(|assignment| {
                    let lock_ref = &state_lock;
                    scope.spawn(move || self.execute_claimed(coordinator, assignment, lock_ref))
                })
                .collect();
            for handle in handles {
                match handle.join() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => {
                        if first_error.is_none() {
                            first_error = Some(err);
                        }
                    }
                    Err(_) => {
                        if first_error.is_none() {
                            first_error = Some(Error::Protocol("worker_thread_panicked"));
                        }
                    }
                }
            }
        });
        if let Some(err) = first_error {
            return Err(err);
        }
        Ok(resumed + count)
    }
}

