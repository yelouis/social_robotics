"""
Constants and validation for the Layer 05 Overlay Bundle schema.
"""

SCHEMA_VERSION = "05.2.0"

def validate_bundle(bundle: dict):
    """
    Validate the top-level structure of an Overlay Bundle.
    """
    if bundle.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected bundle schema {SCHEMA_VERSION}, got {bundle.get('schema_version')}")
    
    assert "video_id" in bundle, "Missing video_id"
    assert "source_dataset" in bundle, "Missing source_dataset"
    assert "clip" in bundle, "Missing clip"
    assert "video_path" in bundle["clip"], "Missing clip.video_path"
    assert "native_width" in bundle["clip"], "Missing clip.native_width"
    assert "native_height" in bundle["clip"], "Missing clip.native_height"
    assert "fps" in bundle["clip"], "Missing clip.fps"
    assert "layers_present" in bundle, "Missing layers_present"
    assert "people" in bundle, "Missing people"
    assert "tracks" in bundle, "Missing tracks"
    assert "tasks" in bundle, "Missing tasks"
