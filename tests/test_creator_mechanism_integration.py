"""The creator mechanism reads Attelier's signed usage feed over HTTP and weights only creators
whose hotkey is registered and owned by the coldkey they verified in the app.

`signed_feed.json` was produced by the app's own feed code (Node, @polkadot sr25519) with fixed
test seeds, so this also proves the two implementations agree on the signed bytes.
"""

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from forge.validator.creator_mechanism import creator_weights, fetch_creator_feed

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "creator_feed" / "signed_feed.json").read_text()
)
DOCUMENT, WINDOW, ADDRESS = FIXTURE["document"], FIXTURE["window"], FIXTURE["addresses"]
ISSUER = ADDRESS["7"]


class RecordedChain:
    """A finalized metagraph port: UID -> (hotkey, owning coldkey)."""

    def __init__(self, owners):
        self.owners = owners
        self.heights = []

    def owner_snapshot(self, height):
        self.heights.append(height)
        return dict(self.owners)


def serve(document):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server API
            url = urlsplit(self.path)
            requests.append(
                (url.path, {key: values[0] for key, values in parse_qs(url.query).items()})
            )
            body = json.dumps(document).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}", requests


def fetch(document, **overrides):
    server, url, requests = serve(document)
    try:
        options = {
            "issuer": ISSUER,
            "window_from": WINDOW["from"],
            "window_to": WINDOW["to"],
            **overrides,
        }
        return fetch_creator_feed(url, **options), requests
    finally:
        server.shutdown()


def test_signed_feed_weights_only_registered_creators_owned_by_their_verified_coldkey():
    feed, requests = fetch(DOCUMENT)
    assert requests == [
        ("/v1/mining/usage-feed", {"from": str(WINDOW["from"]), "to": str(WINDOW["to"])})
    ]
    assert [entry.credits for entry in feed.entries] == [300, 100, 50, 25]
    chain = RecordedChain(
        {
            0: (ADDRESS["7"], ADDRESS["7"]),  # the owner's burn UID
            3: (
                ADDRESS["12"],
                ADDRESS["11"],
            ),  # creator A: registered, owned by the verified coldkey
            5: (ADDRESS["14"], ADDRESS["13"]),  # creator B: registered, owned
            8: (
                ADDRESS["16"],
                ADDRESS["18"],
            ),  # creator C's hotkey, but another coldkey owns it now
            # creator D's hotkey is not registered at all
        }
    )
    weights = creator_weights(feed, chain, 4_200, burn_uid=0)
    assert chain.heights == [4_200]
    assert weights.burned is False
    assert weights.uid_weights == {3: 0.75, 5: 0.25}
    assert weights.excluded == {
        ADDRESS["16"]: "hotkey is not owned by the verified coldkey",
        ADDRESS["18"]: "hotkey is not registered on the subnet",
    }


def test_nobody_qualifying_burns_the_mechanism():
    feed, _ = fetch(DOCUMENT)
    weights = creator_weights(
        feed, RecordedChain({0: (ADDRESS["7"], ADDRESS["7"])}), 10, burn_uid=0
    )
    assert weights.burned is True
    assert weights.uid_weights == {0: 1.0}


@pytest.mark.parametrize(
    ("change", "overrides", "error"),
    [
        (
            lambda doc: doc.update(
                payload=doc["payload"].replace('"credits":300', '"credits":3000')
            ),
            {},
            "signature is invalid",
        ),
        (
            lambda doc: None,
            {"issuer": ADDRESS["11"]},
            "not signed by the configured Attelier issuer",
        ),
        (lambda doc: None, {"window_from": WINDOW["from"] - 3_600_000}, "different window"),
    ],
)
def test_a_tampered_foreign_or_stale_feed_is_refused(change, overrides, error):
    document = copy.deepcopy(DOCUMENT)
    change(document)
    with pytest.raises(ValueError, match=error):
        fetch(document, **overrides)
