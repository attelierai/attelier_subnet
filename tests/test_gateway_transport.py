import json

import pytest

from forge.inference.gateway_transport import MAX_MASTER_BYTES, parse_gateway_master_bytes


def test_gateway_master_transport_is_https_json_object_and_bounded():
    payload = json.dumps({"schema": "attelier/gateway-master-window/v1"}).encode()
    assert parse_gateway_master_bytes(
        "https://example.test/data/master.json", payload, str(len(payload))
    ) == {"schema": "attelier/gateway-master-window/v1"}
    with pytest.raises(ValueError, match="gateway_master_url_https"):
        parse_gateway_master_bytes("http://127.0.0.1/master.json", payload)
    with pytest.raises(ValueError, match="gateway_master_json_object"):
        parse_gateway_master_bytes("https://example.test/master.json", b"[]")
    with pytest.raises(ValueError, match="gateway_master_too_large"):
        parse_gateway_master_bytes(
            "https://example.test/master.json", b"{}", str(MAX_MASTER_BYTES + 1)
        )
    with pytest.raises(ValueError, match="gateway_master_too_large"):
        parse_gateway_master_bytes(
            "https://example.test/master.json", b"x" * (MAX_MASTER_BYTES + 1)
        )
