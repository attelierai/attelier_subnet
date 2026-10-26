"""Deterministic synthetic panels revealed only after candidate commitments close.

The generator is deliberately local and reproducible on every validator. Its
coverage needs independent calibration; synthetic does not mean ungameable.
"""

from forge.protocol.hashing import hash_canonical

GENERATOR_ID = "forge/synthetic-wan-panel/v1"
LAUNCH_V2_GENERATOR_ID = "forge/synthetic-wan-panel/v2.1"
SUBJECTS = {
    "human": (
        "a baker kneading dough on a wooden table",
        "a dancer turning with a blue scarf",
        "a cyclist riding along a gravel path",
        "a potter shaping a clay bowl",
        "a gardener watering flowers",
        "a violinist moving the bow across the strings",
        "a person folding a patterned blanket",
        "a painter brushing a canvas",
    ),
    "animal": (
        "a dog fetching a red ball",
        "a cat stretching beside a window",
        "a horse walking across a field",
        "a duck swimming across a pond",
        "a butterfly landing on a flower",
        "a squirrel opening a walnut",
        "a rabbit chewing a leaf",
        "a heron stepping through shallow water",
    ),
    "object": (
        "a brass watch resting motionless on velvet",
        "a paper boat floating in a stream",
        "a spinning wooden top",
        "a glass of water standing motionless on a table",
        "a toy train travelling around a curved track",
        "a windmill turning slowly",
        "a ceramic vase standing motionless on a shelf",
        "a pendulum swinging steadily",
    ),
    "scenery": (
        "a waterfall cascading over rocks",
        "gentle waves washing onto sand",
        "clouds passing over a mountain",
        "leaves moving in a gentle breeze",
        "an empty stone courtyard with still air",
        "snow falling beside a cabin",
        "a quiet library interior with no movement",
        "rain falling into a puddle",
    ),
}
SCREEN_SUBJECTS = {
    "human": ("a juggler tossing three balls", "a person tying a shoe"),
    "animal": ("an otter diving underwater", "a deer raising its head"),
    "object": ("an umbrella opening slowly", "a flag fluttering in the wind"),
    "scenery": ("mist drifting above a vineyard", "a fountain spraying in a city square"),
}
LIGHT = ("soft morning light", "overcast daylight", "warm afternoon light", "cool evening light")

# Whole-shot semantics are explicit; never assign motion by shuffled position.
# Keep the V1 catalog above unchanged so old evidence remains reproducible.
V2_STATIC_SCENES = {
    "human": (
        "one person sitting motionless beside a closed book",
        "two people standing motionless on opposite sides of a doorway",
    ),
    "animal": (
        "one cat lying motionless beside a closed window",
        "two dogs resting motionless on opposite sides of a basket",
    ),
    "object": (
        "one brass watch resting motionless beside two coins on velvet",
        "two ceramic vases standing motionless on opposite sides of a bowl",
    ),
    "scenery": (
        "an empty stone courtyard with two benches, everything motionless in still air",
        "a quiet library with two rows of shelves, everything motionless with no movement",
    ),
}
V2_MOTION_SCENES = {
    "human": (
        "a baker repeatedly kneading dough with both hands on a wooden table",
        "a dancer turning clockwise while holding a blue scarf",
        "a cyclist riding from left to right along a gravel path",
        "a potter shaping a clay bowl rotating on a wheel",
        "a gardener pouring water onto three flowers in a row",
        "a violinist repeatedly moving the bow across the strings",
        "a person folding a patterned blanket in half",
        "a painter brushing a canvas from left to right",
    ),
    "animal": (
        "a dog running from left to right to fetch a red ball",
        "a cat stretching its front legs beside a window",
        "a horse walking from right to left across a field",
        "two ducks swimming beside each other across a pond",
        "a butterfly flapping its wings while landing on a flower",
        "a squirrel turning a walnut between its paws",
        "a rabbit repeatedly chewing a leaf",
        "a heron stepping through shallow water toward a rock",
    ),
    "object": (
        "a paper boat floating downstream past two rocks",
        "a wooden top spinning clockwise on a table",
        "a toy train travelling clockwise around a curved track",
        "a windmill turning steadily above a field",
        "a pendulum swinging repeatedly from left to right",
        "two red balls rolling from left to right down a ramp",
        "a flag fluttering continuously in the wind",
        "water pouring from a jug into a glass",
    ),
    "scenery": (
        "a waterfall cascading over two large rocks",
        "gentle waves repeatedly washing onto sand",
        "clouds passing from left to right over a mountain",
        "leaves moving continuously in a gentle breeze",
        "snow falling beside a cabin with two windows",
        "rain falling into a puddle and making expanding ripples",
        "a river flowing beneath a stone bridge",
        "mist drifting from right to left above a vineyard",
    ),
}


def synthetic_panel(entropy: str, *, round_id: str, phase: str, per_cell: int = 8) -> list[dict]:
    if not entropy or not round_id or phase not in {"screening", "confirmation", "diagnostic"}:
        raise ValueError("frozen entropy, round and phase are required")
    if type(per_cell) is not int or not 1 <= per_cell <= 8:
        raise ValueError("per_cell must be in 1..8")
    prompts = []
    source = SCREEN_SUBJECTS if phase == "screening" else SUBJECTS
    if phase == "screening" and per_cell != 1:
        raise ValueError("screening has exactly one prompt per cell")
    for subject, scenes in source.items():
        for camera in ("static", "moving"):
            for style in ("natural", "stylized"):
                ordered = sorted(
                    scenes,
                    key=lambda scene: hash_canonical(
                        GENERATOR_ID, [entropy, round_id, phase, subject, camera, style, scene]
                    ),
                )
                for index, scene in enumerate(ordered[:per_cell]):
                    digest = hash_canonical(
                        GENERATOR_ID, [entropy, round_id, phase, subject, camera, style, index]
                    )
                    camera_text = (
                        "The camera remains fixed."
                        if camera == "static"
                        else "The camera slowly moves sideways, keeping the subject visible."
                    )
                    look = (
                        "Photorealistic footage" if style == "natural" else "Watercolor animation"
                    )
                    lighting = LIGHT[int(digest[:2], 16) % len(LIGHT)]
                    prompts.append(
                        {
                            "prompt_id": f"{phase}-{subject}-{camera}-{style}-{index}",
                            "text": f"{look} of {scene}, in {lighting}. "
                            f"{camera_text} A continuous shot without cuts or text overlays.",
                            "seed": int(digest[:15], 16),
                            "subject_class": subject,
                            "camera_type": camera,
                            "style_family": style,
                        }
                    )
    return prompts


def launch_panel(entropy: str, *, round_id: str) -> list[dict]:
    """Sixteen distinct scene families, including four genuinely static shots.

    Freeze entropy after submissions close. Public compositional templates do
    not imply an unseen distribution or immunity to reward-model overfitting.
    The static/motion labels describe the whole shot, not just its camera.
    """
    pool = synthetic_panel(entropy, round_id=round_id, phase="confirmation")
    ordered = sorted(pool, key=lambda p: hash_canonical("forge/launch-panel/v1", entropy, p))
    selected, families = [], set()
    static_markers = ("motionless", "still air", "no movement")
    for kind, count in (("static", 4), ("motion", 12)):
        for prompt in ordered:
            family = prompt["text"].split(" of ", 1)[1].split(", in ", 1)[0]
            is_static = any(marker in family for marker in static_markers)
            if family in families or is_static != (kind == "static"):
                continue
            if kind == "static" and prompt["camera_type"] != "static":
                continue
            families.add(family)
            selected.append(
                {
                    "prompt_id": prompt["prompt_id"],
                    "text": prompt["text"],
                    "seed": prompt["seed"],
                    "kind": kind,
                    "tags": ["steady_lighting"]
                    + (["motion_required", "continuous_motion"] if kind == "motion" else []),
                }
            )
            count -= 1
            if count == 0:
                break
        if count:
            raise ValueError("generator cannot supply the complete distinct V1 scene panel")
    return selected


def launch_panel_v2(entropy: str, *, round_id: str, split: str = "confirmation") -> list[dict]:
    """Return the public 32-prompt V2 panel.

    V1 remains frozen above.  V2 deliberately has a separate generator identity:
    its development and confirmation combinations are disjoint even though both
    are public, deterministic compositions.  A prompt (rather than one of its
    optional diversity seeds) is the independent scoring unit.
    """
    if not entropy or not round_id or split not in {"development", "confirmation"}:
        raise ValueError("entropy, round_id and split (development/confirmation) are required")
    prompts: list[dict] = []
    for family in V2_STATIC_SCENES:
        scenes = []
        for kind, catalog, count in (
            ("static", V2_STATIC_SCENES, 2),
            ("motion", V2_MOTION_SCENES, 6),
        ):
            selected = sorted(
                catalog[family],
                key=lambda value: hash_canonical(
                    LAUNCH_V2_GENERATOR_ID, [entropy, round_id, split, family, kind, value]
                ),
            )[:count]
            scenes.extend((kind, scene) for scene in selected)
        for position in range(8):
            digest = hash_canonical(
                LAUNCH_V2_GENERATOR_ID, [entropy, round_id, split, family, position]
            )
            kind, scene = scenes[position]
            camera = (
                "The camera remains fixed."
                if kind == "static"
                else ("The camera tracks the described action continuously without cuts.")
            )
            # Disjoint lighting conditions make the compositions differ in actual
            # content. Both catalogs are public: this is not an unseen holdout.
            lighting_options = LIGHT[:2] if split == "development" else LIGHT[2:]
            lighting = lighting_options[int(digest[-2:], 16) % len(lighting_options)]
            constraint = (
                "The entire scene stays motionless throughout one continuous shot."
                if kind == "static"
                else "Keep the stated relation visible throughout the described motion."
            )
            combination_id = hash_canonical(
                "forge/launch-v2-combination/v2",
                [family, scene, kind, camera, lighting, constraint],
            )
            prompts.append(
                {
                    "prompt_id": f"v2-{split}-{family}-{position}",
                    "text": (
                        f"Photorealistic footage of {scene}, in {lighting}. {camera} {constraint} "
                        "Lighting stays steady. No text overlays."
                    ),
                    "seed": int(digest[:15], 16),
                    "kind": kind,
                    "family": family,
                    "split": split,
                    "combination_id": combination_id,
                    "tags": ["steady_lighting"]
                    + (["motion_required"] if kind == "motion" else []),
                }
            )
    if len(prompts) != 32 or sum(p["kind"] == "static" for p in prompts) != 8:
        raise AssertionError("V2 panel geometry changed")
    return prompts
