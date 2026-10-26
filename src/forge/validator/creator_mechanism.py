"""Creator mechanism: rewards Attelier template creators who verified a coldkey in the app.

This is a separate incentive mechanism (its own weights and consensus) beside the model
competition on mechanism 0, so the two never dilute each other's scoring. The app publishes a
signed feed of credits that *other* people spent on each creator's templates (paid plans only,
daily caps, a hold period and operator voids already applied). A validator:

1. fetches the feed for a window and checks Attelier's sr25519 signature over the exact bytes;
2. keeps only entries whose hotkey is registered on the subnet *and* owned by the coldkey the
   creator proved in the app (one metagraph read at a pinned block);
3. weights those UIDs in proportion to credits, or burns when nobody qualifies.

Weights for this mechanism are a pure function of the signed feed and the chain, so every
honest validator computes the same vector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from forge.chain import Sr25519Verifier

CREATOR_MECHANISM_ID = 1
FEED_KIND = "attelier.creator-usage"
MAX_FEED_BYTES = 4 * 1024 * 1024


class OwnershipChain(Protocol):
    def owner_snapshot(self, height: int) -> dict[int, tuple[str, str]]:
        """UID -> (hotkey, owning coldkey) at ``height``."""


@dataclass(frozen=True)
class CreatorUsage:
    hotkey: str
    coldkey: str
    credits: float
    runs: int
    people: int


@dataclass(frozen=True)
class CreatorFeed:
    issuer: str
    window_from: int
    window_to: int
    entries: tuple[CreatorUsage, ...]


def verify_creator_feed(
    document: dict, *, issuer: str, window_from: int, window_to: int
) -> CreatorFeed:
    """Accept a feed only if Attelier's key signed these exact bytes for this exact window."""

    if document.get("issuer") != issuer:
        raise ValueError("creator feed is not signed by the configured Attelier issuer")
    payload, signature = document.get("payload"), document.get("signature")
    if (
        not isinstance(payload, str)
        or not isinstance(signature, str)
        or not signature.startswith("0x")
    ):
        raise ValueError("creator feed is malformed")
    if not Sr25519Verifier().verify(issuer, payload.encode(), bytes.fromhex(signature[2:])):
        raise ValueError("creator feed signature is invalid")
    body = json.loads(payload)
    if body.get("version") != 1 or body.get("kind") != FEED_KIND:
        raise ValueError("creator feed has an unsupported version")
    if body.get("from") != window_from or body.get("to") != window_to:
        raise ValueError("creator feed covers a different window")
    entries = []
    for raw in body.get("entries", []):
        credits = float(raw["credits"])
        if credits <= 0:
            continue
        entries.append(
            CreatorUsage(
                str(raw["hotkey"]),
                str(raw["coldkey"]),
                credits,
                int(raw["runs"]),
                int(raw["people"]),
            )
        )
    return CreatorFeed(issuer, window_from, window_to, tuple(entries))


def fetch_creator_feed(
    base_url: str, *, issuer: str, window_from: int, window_to: int, timeout: float = 20
) -> CreatorFeed:
    query = urlencode({"from": window_from, "to": window_to})
    request = Request(
        f"{base_url.rstrip('/')}/v1/mining/usage-feed?{query}",
        headers={"accept": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-configured HTTPS endpoint
        raw = response.read(MAX_FEED_BYTES + 1)
    if len(raw) > MAX_FEED_BYTES:
        raise ValueError("creator feed is too large")
    return verify_creator_feed(
        json.loads(raw), issuer=issuer, window_from=window_from, window_to=window_to
    )


@dataclass(frozen=True)
class CreatorWeights:
    height: int
    uid_weights: dict[int, float]
    burned: bool
    excluded: dict[str, str]


def creator_weights(
    feed: CreatorFeed, chain: OwnershipChain, height: int, *, burn_uid: int
) -> CreatorWeights:
    """UID weights in proportion to credits, for entries the chain confirms; burn otherwise."""

    owners = chain.owner_snapshot(height)
    uid_of = {hotkey: uid for uid, (hotkey, _) in owners.items()}
    credits: dict[int, float] = {}
    excluded: dict[str, str] = {}
    for entry in feed.entries:
        uid = uid_of.get(entry.hotkey)
        if uid is None:
            excluded[entry.hotkey] = "hotkey is not registered on the subnet"
        elif owners[uid][1] != entry.coldkey:
            excluded[entry.hotkey] = "hotkey is not owned by the verified coldkey"
        else:
            credits[uid] = credits.get(uid, 0.0) + entry.credits
    total = sum(credits.values())
    if total <= 0:
        return CreatorWeights(height, {burn_uid: 1.0}, True, excluded)
    return CreatorWeights(
        height, {uid: value / total for uid, value in sorted(credits.items())}, False, excluded
    )
