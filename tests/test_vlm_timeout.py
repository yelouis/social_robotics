import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


# ---------------------------------------------------------------------------
# vlm_client.ollama_chat — enforced-timeout HTTP call
# ---------------------------------------------------------------------------
def test_ollama_chat_posts_to_api_chat_with_enforced_timeout(monkeypatch, tmp_path):
    import shared.vlm_client as vc
    captured = {}

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "YES"}}

    def fake_post(url, json=None, timeout=None):
        captured.update(url=url, payload=json, timeout=timeout)
        return FakeResp()

    monkeypatch.setattr(vc.httpx, "post", fake_post)
    img = tmp_path / "f.jpg"; img.write_bytes(b"\xff\xd8\xffabc")
    out = vc.ollama_chat("qwen", "hi", image_paths=[str(img)],
                         options={"num_ctx": 4096}, fmt="json", timeout=180)
    assert out == "YES"
    assert captured["url"].endswith("/api/chat")
    assert captured["timeout"] == 180            # the timeout is actually passed to httpx
    assert captured["payload"]["stream"] is False
    assert captured["payload"]["format"] == "json"
    assert captured["payload"]["options"] == {"num_ctx": 4096}
    assert captured["payload"]["messages"][0]["images"]   # image base64-encoded


def test_ollama_chat_raises_on_timeout(monkeypatch):
    import shared.vlm_client as vc

    def fake_post(*a, **k):
        raise vc.httpx.TimeoutException("timed out")

    monkeypatch.setattr(vc.httpx, "post", fake_post)
    with pytest.raises(vc.httpx.TimeoutException):
        vc.ollama_chat("m", "hi", timeout=1)


# ---------------------------------------------------------------------------
# The social-presence gate degrades gracefully on a VLM timeout / honors result
# ---------------------------------------------------------------------------
def test_gate_times_out_to_default_and_passes_timeout(monkeypatch):
    import shared.social_presence as sp
    import shared.vlm_client as vc
    det = sp.SocialPresenceDetector(vlm_model="qwen2.5vl:7b")
    frame = np.zeros((48, 48, 3), dtype=np.uint8)

    # A timeout must not propagate out of the gate — it returns the default.
    captured = {}

    def boom(*a, **k):
        captured.update(k)
        raise vc.httpx.TimeoutException("wedged")

    monkeypatch.setattr(sp, "ollama_chat", boom)
    assert det._vlm_confirms_multiple_people(frame) is True   # default_on_error=True
    assert captured.get("timeout") == sp.VLM_REQUEST_TIMEOUT   # generous bound passed through

    # Normal YES / NO answers are honored.
    monkeypatch.setattr(sp, "ollama_chat", lambda *a, **k: "YES")
    assert det._vlm_confirms_multiple_people(frame) is True
    monkeypatch.setattr(sp, "ollama_chat", lambda *a, **k: "NO")
    assert det._vlm_confirms_multiple_people(frame) is False


def test_vlm_timeout_constants_are_generous():
    import shared.social_presence as sp
    import shared.climax_extraction as ce
    assert sp.VLM_REQUEST_TIMEOUT >= 60       # generous vs ~6s normal — only true hangs fire
    assert ce.CLIMAX_VLM_REQUEST_TIMEOUT >= 60
