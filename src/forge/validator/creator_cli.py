"""`forge creator-weights`: score the creator mechanism from Attelier's signed usage feed."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

from forge.validator.creator_mechanism import (
    CREATOR_MECHANISM_ID,
    creator_weights,
    fetch_creator_feed,
)

HOUR_MS = 3_600_000


def add_commands(commands: argparse._SubParsersAction) -> None:
    parser = commands.add_parser(
        "creator-weights",
        help="creator mechanism weights from the signed Attelier feed (dry run by default)",
    )
    parser.add_argument("--network", default="test")
    parser.add_argument("--netuid", type=int, required=True)
    parser.add_argument(
        "--feed-url", required=True, help="Attelier API origin, e.g. https://api.attelier.ai"
    )
    parser.add_argument(
        "--issuer", required=True, help="SS58 address of Attelier's feed signing key"
    )
    parser.add_argument("--window-hours", type=int, default=24)
    parser.add_argument(
        "--burn-uid",
        type=int,
        required=True,
        help="UID that receives the mechanism's weight when no creator qualifies",
    )
    parser.add_argument("--wallet")
    parser.add_argument("--hotkey", default="default")
    parser.add_argument(
        "--submit",
        action="store_true",
        help="submit on chain (testnet only until the emission split is decided)",
    )


def handle_command(args: argparse.Namespace) -> int | None:
    if args.command != "creator-weights":
        return None
    if args.submit and (args.network != "test" or not args.wallet):
        raise SystemExit("--submit needs --wallet and is limited to --network test for now")
    # Whole hours, so every validator asks for the same window and gets the same signed bytes.
    window_to = int(time.time() * 1000) // HOUR_MS * HOUR_MS
    window_from = window_to - args.window_hours * HOUR_MS
    feed = fetch_creator_feed(
        args.feed_url, issuer=args.issuer, window_from=window_from, window_to=window_to
    )

    from forge.chain import BittensorChain

    chain = BittensorChain(args.network, args.netuid)
    try:
        height = chain.finalized_height()
        weights = creator_weights(feed, chain, height, burn_uid=args.burn_uid)
        wallet = None
        if args.submit:
            import bittensor as bt

            wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey)
        submission = chain.set_weights(
            wallet, weights.uid_weights, dry_run=not args.submit, mechid=CREATOR_MECHANISM_ID
        )
    finally:
        chain.close()
    print(
        json.dumps(
            {
                "window": {"from": window_from, "to": window_to},
                "finalized_block": weights.height,
                "creators": len(feed.entries),
                "burned": weights.burned,
                "excluded": weights.excluded,
                "submission": asdict(submission),
            },
            indent=2,
        )
    )
    return 0
