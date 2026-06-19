import os
import sys
import types

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_social_presence_ollama_client_uses_request_timeout(monkeypatch):
    """The VLM client must be constructed with an explicit request timeout, so a
    stuck ollama call raises (-> _vlm_ask_yes_no default) instead of hanging the
    whole 02 batch forever."""
    import shared.social_presence as sp
    captured = {}

    class FakeClient:
        def __init__(self, **kw):
            captured.update(kw)

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=FakeClient))
    det = sp.SocialPresenceDetector()  # lazy __init__ — no model load
    client = det.ollama_client
    assert isinstance(client, FakeClient)
    assert captured.get("timeout") == sp.VLM_REQUEST_TIMEOUT
    assert sp.VLM_REQUEST_TIMEOUT and sp.VLM_REQUEST_TIMEOUT > 0
    # property is cached (client built once)
    assert det.ollama_client is client


def test_climax_extraction_defines_vlm_timeout():
    """The climax-refinement VLM call is also bounded by a positive timeout."""
    import shared.climax_extraction as ce
    assert isinstance(ce.CLIMAX_VLM_REQUEST_TIMEOUT, (int, float))
    assert ce.CLIMAX_VLM_REQUEST_TIMEOUT > 0
