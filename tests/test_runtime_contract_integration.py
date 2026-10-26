from forge.models.artifact_manifest import ArtifactReference, ManifestFile, ModelManifest
from forge.models.generation import GenerationRecipe, paired_reference_scheduler, timing_scope
from forge.protocol.contract import ChallengeContract


def _pipeline(steps: int) -> ModelManifest:
    return ModelManifest(
        ArtifactReference("hf", "public/base", "a" * 40),
        "pipeline",
        (ManifestFile("model_index.json", "b" * 64, 2),),
        "apache-2.0",
        {"steps": steps, "guidance_scale": 5.0},
    )


def test_wan50_bootstrap_and_later_incumbent_have_explicit_reproducible_timing_recipes():
    contract = ChallengeContract(
        base_model_id="public/base", base_model_revision="a" * 40, reference_steps=50
    )
    base = _pipeline(50)
    assert (
        paired_reference_scheduler(base, base, GenerationRecipe(50, 5.0), contract)
        == "original_unipc"
    )
    assert (
        paired_reference_scheduler(base, base, GenerationRecipe(16, 5.0), contract)
        == "uniform_euler"
    )
    scope = timing_scope()
    assert scope["competitive_includes"] == ["text_encoding", "denoising", "vae_decoding"]
    assert "model_loading" in scope["competitive_excludes"]
