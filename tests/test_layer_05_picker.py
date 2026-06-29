"""
Tests for Layer 05 Picker.
"""
import pytest
from src.layer_05_visualizer.picker import build_picker_rows, filter_rows, sort_rows, PickerRow
from src.layer_05_visualizer.catalog import CatalogEntry

def test_headless_picker_logic():
    # P1 - Row model
    catalog = {
        "vid1": CatalogEntry(
            video_id="vid1",
            manifest_entry={},
            results_by_layer={},
            video_path=None,
            findings={"03a": "finding"},
            num_layers_with_findings=1,
            summary_text="vid1 summary",
            sources={},
            has_audio=None
        ),
        "vid2": CatalogEntry(
            video_id="vid2",
            manifest_entry={"has_audio": False, "identified_tasks": [{"task_label": "cooking"}]},
            results_by_layer={},
            video_path=None,
            findings={"03a": "finding", "03e": "finding"},
            num_layers_with_findings=2,
            summary_text="vid2 summary",
            sources={},
            has_audio=False
        )
    }
    
    rows = build_picker_rows(catalog)
    assert len(rows) == 2
    # Should be sorted by stars desc
    assert rows[0].video_id == "vid2"
    assert rows[1].video_id == "vid1"
    
    # P2 - Filtering
    filtered = filter_rows(rows, "cooking")
    assert len(filtered) == 1
    assert filtered[0].video_id == "vid2"
    
    filtered = filter_rows(rows, "noaudio")
    assert len(filtered) == 1
    assert filtered[0].video_id == "vid2"
    
    # P3 - Sorting
    sorted_rows = sort_rows(rows, "star", descending=False)
    assert sorted_rows[0].video_id == "vid1"
    
    sorted_rows = sort_rows(rows, "task", descending=False)
    assert sorted_rows[0].video_id == "vid2" # 'cooking' < 'none'

@pytest.mark.gui
def test_picker_gui():
    # Skip actual tk mainloop in headless tests, just ensure it imports
    from src.layer_05_visualizer.picker import pick_video
    pass
