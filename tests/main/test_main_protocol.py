import io
import json
import struct
import sys

import pytest


def _pack_msg(payload: bytes, target: int = 0) -> bytes:
    return struct.pack(">Ii", len(payload), target) + payload


class _FakeIn:
    def __init__(self, data: bytes):
        self.buffer = io.BytesIO(data)


class _FakeOut:
    def __init__(self):
        self.buffer = io.BytesIO()
    def flush(self):
        # match sys.stdout API
        return None


def test_send_to_judger_header_and_body(monkeypatch):
    import main as m

    fake_out = _FakeOut()
    monkeypatch.setattr(sys, "stdout", fake_out, raising=False)

    payload = b"{\"hello\":1}"
    m.send_to_judger(payload, target=7)

    data = fake_out.buffer.getvalue()
    assert len(data) == 8 + len(payload)
    length, target = struct.unpack(">Ii", data[:8])
    assert length == len(payload)
    assert target == 7
    assert data[8:] == payload


def test_receive_from_judger_reads_full_header(monkeypatch):
    import main as m

    p = b"{\"a\":1}"
    fake_in = _FakeIn(_pack_msg(p, target=3))
    monkeypatch.setattr(sys, "stdin", fake_in, raising=False)

    out = m.receive_from_judger()
    assert out == p


def test_receive_from_judger_alignment_two_messages(monkeypatch):
    import main as m

    p1 = b"{\"x\":1}"
    p2 = b"{\"y\":2}"
    data = _pack_msg(p1, target=1) + _pack_msg(p2, target=2)
    fake_in = _FakeIn(data)
    monkeypatch.setattr(sys, "stdin", fake_in, raising=False)

    out1 = m.receive_from_judger()
    out2 = m.receive_from_judger()
    assert out1 == p1
    assert out2 == p2


def test_receive_ai_info_and_request_end_state(monkeypatch):
    import main as m

    ai_info = {"player": 0, "content": "8\n"}
    fake_in = _FakeIn(_pack_msg(json.dumps(ai_info).encode("utf-8"), target=0))
    fake_out = _FakeOut()
    monkeypatch.setattr(sys, "stdin", fake_in, raising=False)
    monkeypatch.setattr(sys, "stdout", fake_out, raising=False)

    got = m.receive_ai_info()
    assert got == ai_info

    # Also verify request_ai_end_state() emits the right action payload
    m.request_ai_end_state()
    buf = fake_out.buffer.getvalue()
    # Parse only the last message after two headers; take the last header
    # We can scan by reading the stream using the same receive logic
    bio = io.BytesIO(buf)
    # first message (from previous writes in other tests in same process is unlikely here)
    # but to be robust, we just consume one message and keep the last
    def read_one(b):
        hdr = b.read(8)
        if not hdr:
            return None
        L, _ = struct.unpack(">Ii", hdr)
        return b.read(L)

    first = read_one(bio)
    second = read_one(bio)
    last = second if second is not None else first
    assert last is not None
    msg = json.loads(last.decode("utf-8"))
    assert msg.get("action") == "request_end_state"
