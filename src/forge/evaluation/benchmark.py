"""Deterministic, candidate-independent procedural video benchmark grammars."""

from __future__ import annotations

import random
from dataclasses import dataclass

from forge.protocol.contract import ChallengeContract
from forge.protocol.hashing import hash_canonical

GRAMMAR_V1 = "forge/bench/grammar/v1"
GRAMMAR_V2 = "forge/bench/grammar/v2"
GRAMMAR_V3 = "forge/bench/grammar/v3"
SUPPORTED_GRAMMARS = frozenset((GRAMMAR_V1, GRAMMAR_V2, GRAMMAR_V3))

# v1 is kept behaviourally identical for replay of closed pilot receipts.
V1_SUBJECTS = {
    "rigid_object": (
        ("red ceramic sphere", "red ceramic spheres", "rolls", "roll"),
        ("blue wooden cube", "blue wooden cubes", "slides", "slide"),
        ("silver toy top", "silver toy tops", "spins", "spin"),
        ("yellow paper plane", "yellow paper planes", "glides", "glide"),
    ),
    "human": (
        ("person in a green jacket", "people in green jackets", "walks", "walk"),
        ("dancer in a blue outfit", "dancers in blue outfits", "dances", "dance"),
        ("runner in a red shirt", "runners in red shirts", "runs", "run"),
        ("hiker with a yellow backpack", "hikers with yellow backpacks", "hikes", "hike"),
    ),
    "animal": (
        ("golden retriever", "golden retrievers", "runs", "run"),
        ("black horse", "black horses", "gallops", "gallop"),
        ("white rabbit", "white rabbits", "hops", "hop"),
        ("orange butterfly", "orange butterflies", "flies", "fly"),
    ),
    "scenery": (
        ("waterfall", "", "flows", ""),
        ("bank of clouds", "", "drifts", ""),
        ("field of tall grass", "", "sways", ""),
        ("stretch of ocean waves", "", "rolls", ""),
    ),
}
V1_SPEEDS = ("slowly", "at a steady pace", "quickly")
V1_DIRECTIONS = ("left to right", "right to left", "toward the camera", "away from the camera")
CAMERA_PHRASES = {
    "static": "with a static camera",
    "pan": "while the camera pans slowly",
    "tilt": "while the camera tilts slowly",
    "dolly": "while the camera dollies in",
    "orbit": "while the camera orbits around the {target}",
    "tracking": "while the camera tracks the {target}",
}
V1_CAMERAS = tuple(CAMERA_PHRASES)
V1_ENVIRONMENTS = (
    "in a sunlit studio",
    "on a quiet city street",
    "beside a mountain lake",
    "in a misty forest clearing",
    "across a wide sandy plain",
)
V1_SCENERY_ENVIRONMENTS = {
    "waterfall": ("in a mountain valley", "in a misty forest"),
    "bank of clouds": ("above a mountain valley", "over a rocky coast"),
    "field of tall grass": ("in a mountain valley", "beside a misty forest"),
    "stretch of ocean waves": ("along a rocky coast", "under a stormy sky"),
}
V1_SCENERY_CAMERAS = ("static", "pan", "tilt", "dolly", "orbit")
V1_STYLES = ("cinematic realism", "documentary footage", "soft watercolor", "stop-motion")

# v2 uses visually checkable, plausible actions and has ample combinations for
# the 512-prompt confirmation panel.
V2_SUBJECTS = {
    "rigid_object": (
        ("red ceramic sphere", "red ceramic spheres", "rolls", "roll"),
        ("blue wooden cube", "blue wooden cubes", "slides", "slide"),
        ("silver toy top", "silver toy tops", "spins", "spin"),
        ("yellow paper plane", "yellow paper planes", "glides", "glide"),
    ),
    "human": (
        ("person in a green jacket", "people in green jackets", "walks", "walk"),
        ("dancer in a blue outfit", "dancers in blue outfits", "turns", "turn"),
        ("runner in a red shirt", "runners in red shirts", "runs", "run"),
        ("hiker with a yellow backpack", "hikers with yellow backpacks", "walks", "walk"),
        ("skateboarder", "skateboarders", "rides", "ride"),
        ("cyclist", "cyclists", "pedals", "pedal"),
    ),
    "animal": (
        ("golden retriever", "golden retrievers", "runs", "run"),
        ("black horse", "black horses", "gallops", "gallop"),
        ("white rabbit", "white rabbits", "hops", "hop"),
        ("orange butterfly", "orange butterflies", "flies", "fly"),
        ("mallard duck", "mallard ducks", "swims", "swim"),
        ("deer", "deer", "walks", "walk"),
    ),
    "scenery": (
        ("waterfall", "", "flows", ""),
        ("bank of clouds", "", "drifts", ""),
        ("field of tall grass", "", "sways", ""),
        ("ocean waves", "", "break", ""),
        ("autumn leaves", "", "fall", ""),
        ("river", "", "flows", ""),
    ),
}
V2_ENVIRONMENTS = {
    "rigid_object": (
        "across a sunlit wooden table",
        "on a clean white studio floor",
        "along a quiet park path",
        "across a sandy beach",
    ),
    "human": (
        "through a sunlit studio",
        "along a quiet city street",
        "across a wide sandy plain",
        "through a misty forest clearing",
    ),
    "animal": (
        "across a grassy field",
        "beside a mountain lake",
        "through a misty forest clearing",
        "along a sandy beach",
    ),
}
V2_SUBJECT_ENVIRONMENTS = {
    "red ceramic sphere": ("across a sunlit wooden table", "on a clean studio floor"),
    "blue wooden cube": ("across a sunlit wooden table", "on a clean studio floor"),
    "silver toy top": ("on a polished wooden tabletop", "on a clean studio floor"),
    "yellow paper plane": ("through a sunlit studio", "along a quiet park path"),
    "mallard duck": ("across a calm pond", "across a quiet lake", "along a slow river"),
}
V2_IN_PLACE_SUBJECTS = frozenset(("silver toy top",))
V2_SCENERY_ENVIRONMENTS = {
    "waterfall": ("in a mountain valley", "in a misty forest"),
    "bank of clouds": ("above a mountain valley", "over a rocky coast"),
    "field of tall grass": ("beside a misty forest", "in a mountain valley"),
    "ocean waves": ("along a rocky coast", "under a stormy sky"),
    "autumn leaves": ("through a quiet forest", "across a park path"),
    "river": ("through a mountain valley", "beside a forest"),
}
V2_SCENERY_HEADS = {
    "waterfall": "A waterfall",
    "bank of clouds": "A bank of clouds",
    "field of tall grass": "A field of tall grass",
    "ocean waves": "Ocean waves",
    "autumn leaves": "Autumn leaves",
    "river": "A river",
}
V2_STYLES = {
    "realistic": ("cinematic realism", "documentary footage"),
    "stylized": ("soft watercolor", "stop-motion animation"),
}
V2_SPEEDS = {"slow": ("slowly", "at a steady pace"), "fast": ("quickly",)}
V2_DIRECTIONS = ("left to right", "right to left", "toward the camera", "away from the camera")
V2_MOVING_CAMERAS = ("pan", "dolly", "orbit", "tracking")
V2_SCENERY_MOVING_CAMERAS = ("pan", "dolly", "orbit")

# v3 expands genuine visual conditions, instead of appending opaque identifiers,
# so a large public confirmation panel has enough semantically distinct prompts.
V3_LIGHTING = (
    "in soft morning light",
    "in bright midday light",
    "under warm evening light",
    "under cool overcast light",
    "at golden hour",
    "in gentle diffused light",
    "in clear crisp light",
    "in low winter sunlight",
    "in pale dawn light",
    "in rich late-afternoon light",
    "with subtle side lighting",
    "in even ambient light",
)
V3_FRAMING = (
    "framed in a wide composition",
    "framed in a medium composition",
    "framed in a close composition",
    "viewed from eye level",
    "viewed from a low angle",
    "viewed from a high angle",
    "framed with a centered composition",
    "framed with an off-center composition",
)
V3_ENVIRONMENTS = {
    "rigid_object": (
        "across a wooden table",
        "on a clean white studio floor",
        "along a quiet park path",
        "across a sandy beach",
    ),
    "human": (
        "through a quiet studio",
        "along a quiet city street",
        "across a wide sandy plain",
        "through a forest clearing",
    ),
    "animal": (
        "across a grassy field",
        "beside a mountain lake",
        "through a forest clearing",
        "along a sandy beach",
    ),
}
V3_SUBJECT_ENVIRONMENTS = {
    "red ceramic sphere": ("across a wooden table", "on a clean studio floor"),
    "blue wooden cube": ("across a wooden table", "on a clean studio floor"),
    "silver toy top": ("on a polished wooden tabletop", "on a clean studio floor"),
    "yellow paper plane": ("through a quiet studio", "along a quiet park path"),
    "mallard duck": ("across a calm pond", "across a quiet lake", "along a slow river"),
}
V3_SCENERY_ENVIRONMENTS = {
    "waterfall": ("in a mountain valley", "in a forest"),
    "bank of clouds": ("above a mountain valley", "over a rocky coast"),
    "field of tall grass": ("beside a forest", "in a mountain valley"),
    "ocean waves": ("along a rocky coast", "beneath an open sky"),
    "autumn leaves": ("through a quiet forest", "across a park path"),
    "river": ("through a mountain valley", "beside a forest"),
}


@dataclass(frozen=True)
class PromptCase:
    prompt_id: str
    text: str
    properties: dict[str, str | int]
    rubric: tuple[str, ...]
    video_seeds: tuple[int, ...]


def benchmark_seed(
    contract_hash: str,
    round_id: str,
    stage: str,
    block_hash: str,
    grammar_version: str = GRAMMAR_V1,
) -> str:
    if stage not in {"screen", "duel", "confirmation"}:
        raise ValueError(f"unknown benchmark stage: {stage}")
    if grammar_version not in SUPPORTED_GRAMMARS:
        raise ValueError(f"unsupported benchmark grammar: {grammar_version}")
    return hash_canonical(
        "forge/bench/v1", grammar_version, contract_hash, round_id, stage, block_hash
    )


def _generate_v1(root: str, count: int, seed_count: int) -> tuple[PromptCase, ...]:
    """Original v1 implementation, retained for historical panel replay."""
    rng = random.Random(int(root, 16))
    classes = tuple(V1_SUBJECTS)
    cases: list[PromptCase] = []
    class_order: list[str] = []
    for index in range(count):
        if index % len(classes) == 0:
            class_order = list(classes)
            rng.shuffle(class_order)
        subject_class = class_order[index % len(classes)]
        singular, plural, verb_one, verb_many = rng.choice(V1_SUBJECTS[subject_class])
        speed = rng.choice(V1_SPEEDS)
        scenery = subject_class == "scenery"
        camera = rng.choice(V1_SCENERY_CAMERAS if scenery else V1_CAMERAS)
        environment = rng.choice(V1_SCENERY_ENVIRONMENTS[singular] if scenery else V1_ENVIRONMENTS)
        style = rng.choice(V1_STYLES)
        subject_count = 1 if scenery else rng.randint(1, 3)
        direction = "none" if scenery else rng.choice(V1_DIRECTIONS)
        target = "scene" if scenery else ("subject" if subject_count == 1 else "group")
        if scenery:
            head = f"A {singular} {verb_one} {speed}"
        elif subject_count == 1:
            head = f"One {singular} {verb_one} {speed} {direction}"
        else:
            count_word = {2: "Two", 3: "Three"}[subject_count]
            head = f"{count_word} {plural} {verb_many} {speed} {direction}"
        camera_phrase = CAMERA_PHRASES[camera].format(target=target)
        text = f"{head} {environment}, {camera_phrase}, rendered as {style}."
        properties: dict[str, str | int] = {
            "subject_class": subject_class,
            "subject": singular,
            "action": verb_one,
            "speed": speed,
            "direction": direction,
            "camera": camera,
            "environment": environment,
            "subject_count": subject_count,
            "style": style,
        }
        prompt_id = hash_canonical("forge/prompt/v1", root, index, properties, text)
        video_seeds = tuple(rng.getrandbits(63) for _ in range(seed_count))
        rubric = (
            "A single continuous scene is shown."
            if scenery
            else f"Exactly {subject_count} principal subject(s) remain visible and countable.",
            f"The main subject is recognizably a {singular}.",
            f"The motion ({verb_one}) happens {speed}.",
            "The scene has no traveling subject."
            if scenery
            else f"Net subject motion follows {direction}.",
            f"Camera behavior is {camera}, distinct from subject motion.",
            "Motion is temporally coherent and physically plausible.",
        )
        cases.append(PromptCase(prompt_id, text, properties, rubric, video_seeds))
    return tuple(cases)


def _balanced_value(root: str, dimension: str, index: int, values: tuple[str, ...]) -> str:
    """A shuffled cycle gives every category coverage within one case."""
    cycle, offset = divmod(index, len(values))
    rng = random.Random(int(hash_canonical("forge/bench/cycle/v2", root, dimension, cycle), 16))
    order = list(values)
    rng.shuffle(order)
    return order[offset]


def _v2_case(root: str, index: int, seed_count: int, used_texts: set[str]) -> PromptCase:
    subject_class = _balanced_value(root, "subject_class", index, tuple(V2_SUBJECTS))
    camera_mode = _balanced_value(root, "camera_mode", index, ("static", "moving"))
    style_family = _balanced_value(root, "style_family", index, ("realistic", "stylized"))
    speed_band = _balanced_value(root, "speed_band", index, ("slow", "fast"))
    scenery = subject_class == "scenery"
    content_rng = random.Random(int(hash_canonical("forge/bench/content/v2", root, index), 16))
    applicable_rank = sum(
        _balanced_value(root, "subject_class", prior, tuple(V2_SUBJECTS)) != "scenery"
        for prior in range(index)
    )
    for _ in range(10_000):
        singular, plural, verb_one, verb_many = content_rng.choice(V2_SUBJECTS[subject_class])
        style = content_rng.choice(V2_STYLES[style_family])
        speed = content_rng.choice(V2_SPEEDS[speed_band])
        if scenery:
            subject_count, count_group, direction = 1, "not_applicable", "none"
            environment = content_rng.choice(V2_SCENERY_ENVIRONMENTS[singular])
            cameras = ("static",) if camera_mode == "static" else V2_SCENERY_MOVING_CAMERAS
            camera, target = content_rng.choice(cameras), "scene"
            head = f"{V2_SCENERY_HEADS[singular]} {verb_one} {speed}"
        else:
            count_group = _balanced_value(
                root, "subject_count", applicable_rank, ("single", "multiple")
            )
            subject_count = 1 if count_group == "single" else content_rng.choice((2, 3))
            in_place = singular in V2_IN_PLACE_SUBJECTS
            direction = "none" if in_place else content_rng.choice(V2_DIRECTIONS)
            environment = content_rng.choice(
                V2_SUBJECT_ENVIRONMENTS.get(singular, V2_ENVIRONMENTS[subject_class])
            )
            cameras = ("static",) if camera_mode == "static" else V2_MOVING_CAMERAS
            camera = content_rng.choice(cameras)
            target = "subject" if subject_count == 1 else "group"
            travel = "in place" if in_place else direction
            if subject_count == 1:
                head = f"One {singular} {verb_one} {speed} {travel}"
            else:
                count_word = {2: "Two", 3: "Three"}[subject_count]
                head = f"{count_word} {plural} {verb_many} {speed} {travel}"
        camera_phrase = CAMERA_PHRASES[camera].format(target=target)
        text = f"{head} {environment}, {camera_phrase}, rendered as {style}."
        if text not in used_texts:
            break
    else:  # pragma: no cover - protocol safeguard.
        raise RuntimeError("v2 grammar exhausted its unique prompt combinations")
    used_texts.add(text)
    properties: dict[str, str | int] = {
        "grammar_version": GRAMMAR_V2,
        "subject_class": subject_class,
        "subject": singular,
        "action": verb_one,
        "speed": speed,
        "speed_band": speed_band,
        "direction": direction,
        "camera": camera,
        "camera_mode": camera_mode,
        "environment": environment,
        "subject_count": subject_count,
        "count_group": count_group,
        "style": style,
        "style_family": style_family,
        "evaluation_slices": "|".join(
            (
                f"subject_class:{subject_class}",
                f"camera_mode:{camera_mode}",
                f"style_family:{style_family}",
                f"speed_band:{speed_band}",
                f"count_group:{count_group}",
            )
        ),
    }
    prompt_id = hash_canonical("forge/prompt/v2", root, index, properties, text)
    seed_rng = random.Random(int(hash_canonical("forge/bench/seeds/v2", root, index), 16))
    video_seeds = tuple(seed_rng.getrandbits(63) for _ in range(seed_count))
    rubric = (
        "A single continuous scene is shown.",
        f"The main subject is recognizably {singular}.",
        f"The motion ({verb_one}) happens {speed}.",
        "The scene has no traveling subject."
        if scenery
        else "The subject spins in place without traveling."
        if direction == "none"
        else f"Net subject motion follows {direction}.",
        f"Camera behavior is {camera}, distinct from subject motion.",
        "Motion is temporally coherent and physically plausible.",
    )
    return PromptCase(prompt_id, text, properties, rubric, video_seeds)


def _generate_v2(root: str, count: int, seed_count: int) -> tuple[PromptCase, ...]:
    used_texts: set[str] = set()
    return tuple(_v2_case(root, index, seed_count, used_texts) for index in range(count))


def _v3_case(
    root: str,
    index: int,
    seed_count: int,
    applicable_rank: int,
    used_texts: set[str],
) -> PromptCase:
    subject_class = _balanced_value(root, "subject_class", index, tuple(V2_SUBJECTS))
    camera_mode = _balanced_value(root, "camera_mode", index, ("static", "moving"))
    style_family = _balanced_value(root, "style_family", index, ("realistic", "stylized"))
    speed_band = _balanced_value(root, "speed_band", index, ("slow", "fast"))
    lighting = _balanced_value(root, "lighting", index, V3_LIGHTING)
    framing = _balanced_value(root, "framing", index, V3_FRAMING)
    scenery = subject_class == "scenery"
    content_rng = random.Random(int(hash_canonical("forge/bench/content/v3", root, index), 16))
    for _ in range(10_000):
        singular, plural, verb_one, verb_many = content_rng.choice(V2_SUBJECTS[subject_class])
        style = content_rng.choice(V2_STYLES[style_family])
        speed = content_rng.choice(V2_SPEEDS[speed_band])
        if scenery:
            subject_count, count_group, direction = 1, "not_applicable", "none"
            environment = content_rng.choice(V3_SCENERY_ENVIRONMENTS[singular])
            cameras = ("static",) if camera_mode == "static" else V2_SCENERY_MOVING_CAMERAS
            camera, target = content_rng.choice(cameras), "scene"
            head = f"{V2_SCENERY_HEADS[singular]} {verb_one} {speed}"
        else:
            count_group = _balanced_value(
                root, "subject_count", applicable_rank, ("single", "multiple")
            )
            subject_count = 1 if count_group == "single" else content_rng.choice((2, 3))
            in_place = singular in V2_IN_PLACE_SUBJECTS
            direction = "none" if in_place else content_rng.choice(V2_DIRECTIONS)
            environment = content_rng.choice(
                V3_SUBJECT_ENVIRONMENTS.get(singular, V3_ENVIRONMENTS[subject_class])
            )
            cameras = ("static",) if camera_mode == "static" else V2_MOVING_CAMERAS
            camera = content_rng.choice(cameras)
            target = "subject" if subject_count == 1 else "group"
            travel = "in place" if in_place else direction
            if subject_count == 1:
                head = f"One {singular} {verb_one} {speed} {travel}"
            else:
                count_word = {2: "Two", 3: "Three"}[subject_count]
                head = f"{count_word} {plural} {verb_many} {speed} {travel}"
        camera_phrase = CAMERA_PHRASES[camera].format(target=target)
        text = f"{head} {environment} {lighting}, {framing}, {camera_phrase}, rendered as {style}."
        if text not in used_texts:
            break
    else:  # pragma: no cover - protocol safeguard.
        raise RuntimeError("v3 grammar exhausted its unique prompt combinations")
    used_texts.add(text)
    properties: dict[str, str | int] = {
        "grammar_version": GRAMMAR_V3,
        "subject_class": subject_class,
        "subject": singular,
        "action": verb_one,
        "speed": speed,
        "speed_band": speed_band,
        "direction": direction,
        "camera": camera,
        "camera_mode": camera_mode,
        "environment": environment,
        "lighting": lighting,
        "framing": framing,
        "subject_count": subject_count,
        "count_group": count_group,
        "style": style,
        "style_family": style_family,
        "evaluation_slices": "|".join(
            (
                f"subject_class:{subject_class}",
                f"camera_mode:{camera_mode}",
                f"style_family:{style_family}",
                f"speed_band:{speed_band}",
                f"count_group:{count_group}",
                f"lighting:{lighting}",
                f"framing:{framing}",
            )
        ),
    }
    prompt_id = hash_canonical("forge/prompt/v3", root, index, properties, text)
    seed_rng = random.Random(int(hash_canonical("forge/bench/seeds/v3", root, index), 16))
    video_seeds = tuple(seed_rng.getrandbits(63) for _ in range(seed_count))
    rubric = (
        "A single continuous scene is shown.",
        f"The main subject is recognizably {singular}.",
        f"The motion ({verb_one}) happens {speed}.",
        "The scene has no traveling subject."
        if scenery
        else "The subject spins in place without traveling."
        if direction == "none"
        else f"Net subject motion follows {direction}.",
        f"Camera behavior is {camera}, distinct from subject motion.",
        "Motion is temporally coherent and physically plausible.",
    )
    return PromptCase(prompt_id, text, properties, rubric, video_seeds)


def _generate_v3(root: str, count: int, seed_count: int) -> tuple[PromptCase, ...]:
    used_texts: set[str] = set()
    cases: list[PromptCase] = []
    applicable_rank = 0
    for index in range(count):
        subject_class = _balanced_value(root, "subject_class", index, tuple(V2_SUBJECTS))
        cases.append(_v3_case(root, index, seed_count, applicable_rank, used_texts))
        if subject_class != "scenery":
            applicable_rank += 1
    return tuple(cases)


def generate_prompts(
    contract: ChallengeContract,
    round_id: str,
    stage: str,
    block_hash: str,
    *,
    prompt_count: int | None = None,
    seeds_per_prompt: int | None = None,
) -> tuple[PromptCase, ...]:
    """Generate a public deterministic panel from finalized entropy inputs."""
    default_count, default_seeds = contract.stage_sizes()[stage]
    count = default_count if prompt_count is None else prompt_count
    seed_count = default_seeds if seeds_per_prompt is None else seeds_per_prompt
    if count <= 0 or seed_count <= 0:
        raise ValueError("panel dimensions must be positive")
    root = benchmark_seed(
        contract.contract_hash(), round_id, stage, block_hash, contract.grammar_version
    )
    if contract.grammar_version == GRAMMAR_V1:
        return _generate_v1(root, count, seed_count)
    if contract.grammar_version == GRAMMAR_V2:
        return _generate_v2(root, count, seed_count)
    if contract.grammar_version == GRAMMAR_V3:
        return _generate_v3(root, count, seed_count)
    raise ValueError(f"unsupported benchmark grammar: {contract.grammar_version}")
