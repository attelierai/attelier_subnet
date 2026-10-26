"""A persisted simulated crown cannot enter the real MVP certificate verifier."""

import pytest

from forge.protocol.consensus_types import CertificateError, FinalizedCompetitionState
from forge.protocol.contract import ChallengeContract
from forge.protocol.koth import CommitmentPosition
from forge.protocol.mvp_certification import MVPCertificateVerifier
from forge.sim import SimNetwork, make_committee, profile


def test_simulated_certificate_never_enters_mvp_rewards(tmp_path):
    contract = ChallengeContract(screen_prompts=16, duel_prompts=64, confirmation_prompts=128)
    baseline, good = profile("baseline", contract, 0.7), profile("candidate", contract, 0.9)
    network = SimNetwork(contract, make_committee(str(tmp_path)), baseline)
    try:
        result = network.run_round(
            "r1", [("miner", good, CommitmentPosition(10, 0))], start_block=100, chain_time=1000
        )
        assert result.certificate is not None
        state = FinalizedCompetitionState.genesis(network.challenge_id, baseline.fingerprint())
        verifier = MVPCertificateVerifier(network.verifier, evidence={})
        with pytest.raises(CertificateError):
            verifier.verify(result.certificate, result.plan, state)
    finally:
        for validator in network.validators.values():
            validator.store.close()
