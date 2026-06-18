import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from layer_04_dehydrated_export import per_layer
from layer_04_dehydrated_export.per_layer import (
    drop_unmeasured_rows,
    dataset_card,
    build_layer_dataset,
    published_layer_name,
    repo_slug,
)

MANIFEST = list(per_layer.MANIFEST_COLUMNS)


def _row(vid, **layer_vals):
    base = {"video_id": vid, "source_dataset": "ego4d", "task_labels": "x",
            "duration_sec": 1.0, "fps": 30.0}
    base.update(layer_vals)
    return base


class TestDropUnmeasuredRows(unittest.TestCase):
    def test_drops_absent_and_sentinel_keeps_scored(self):
        df = pd.DataFrame([
            _row("A", L_metric=0.5, L_skipped_reason=None, L_raw='[{"a":1}]'),   # scored -> keep
            _row("B", L_metric=None, L_skipped_reason="no_data", L_raw="[]"),     # sentinel -> drop
            _row("C", L_metric=None, L_skipped_reason=None, L_raw=None),          # absent -> drop
        ])
        out = drop_unmeasured_rows(df)
        self.assertEqual(list(out["video_id"]), ["A"])
        # the now-vestigial sentinel column is pruned; real signal columns stay
        self.assertNotIn("L_skipped_reason", out.columns)
        self.assertIn("L_metric", out.columns)
        self.assertIn("L_raw", out.columns)
        # manifest columns always preserved
        for c in MANIFEST:
            self.assertIn(c, out.columns)

    def test_empty_json_raw_is_not_signal(self):
        """A row whose only non-manifest content is an empty-JSON raw string
        ("[]") must be treated as unmeasured."""
        df = pd.DataFrame([
            _row("A", L_raw="[]", L_skipped_reason="span_capped"),
            _row("B", L_raw='[{"task":1}]', L_skipped_reason=None),
        ])
        out = drop_unmeasured_rows(df)
        self.assertEqual(list(out["video_id"]), ["B"])

    def test_no_signal_columns_returns_all_rows(self):
        df = pd.DataFrame([_row("A"), _row("B")])  # manifest only
        out = drop_unmeasured_rows(df)
        self.assertEqual(len(out), 2)


class TestNaming(unittest.TestCase):
    def test_published_name_and_slug(self):
        self.assertEqual(published_layer_name("03f_motor_resonance"), "motor_resonance")
        self.assertEqual(repo_slug("03f_motor_resonance"), "social-robotics-motor-resonance")
        self.assertEqual(repo_slug("03a_attention"), "social-robotics-attention")


class TestBuildLayerDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        manifest = [
            {"video_id": "A", "source_dataset": "ego4d", "duration_sec": 10.0, "fps": 30.0,
             "identified_tasks": [{"task_label": "stacking"}]},
            {"video_id": "B", "source_dataset": "ego4d", "duration_sec": 12.0, "fps": 30.0,
             "identified_tasks": [{"task_label": "pouring"}]},
        ]
        (self.dir / "filtered_manifest.json").write_text(json.dumps(manifest))
        # One scored video (A) + one sentinel (B).
        result = [
            {"video_id": "A", "layer": "03f_motor_resonance", "tasks_analyzed": [
                {"task_id": "t1", "ego_kinetic_chaos_score": 0.8, "per_person": [
                    {"person_id": 0, "bystander_pose_velocity_peak": 4.5, "resonance_delay_sec": 0.2,
                     "motor_resonance_detected": True, "empathy_scalar": 0.9,
                     "mirroring_detected": False, "mirroring_scalar": 0.0}]}]},
            {"video_id": "B", "layer": "03f_motor_resonance", "tasks_analyzed": [],
             "skipped_reason": "no_pose_data"},
        ]
        self.result_path = self.dir / "03f_motor_resonance_result.json"
        self.result_path.write_text(json.dumps(result))

    def tearDown(self):
        self.tmp.cleanup()

    def test_sentinel_row_dropped_and_prefix_stripped(self):
        out_dir = self.dir / "out"
        summary = build_layer_dataset(
            "03f_motor_resonance", self.result_path, self.dir / "filtered_manifest.json",
            out_dir, owner="louisye", git_sha="testsha")
        self.assertEqual(summary["repo_id"], "louisye/social-robotics-motor-resonance")
        self.assertEqual(summary["n_rows"], 1)  # B dropped

        df = pd.read_parquet(summary["parquet"])
        self.assertEqual(list(df["video_id"]), ["A"])
        # published headers carry no 03x_ prefix
        self.assertFalse(any(c[:4] in ("03a_", "03b_", "03c_", "03d_", "03e_", "03f_") for c in df.columns))
        self.assertIn("motor_resonance_max_ego_chaos_score", df.columns)
        self.assertIn("motor_resonance_any_motor_resonance", df.columns)
        # the vestigial sentinel column is pruned (B is gone, A has no skip reason)
        self.assertNotIn("motor_resonance_skipped_reason", df.columns)
        # manifest metadata joined
        self.assertEqual(df.iloc[0]["task_labels"], "stacking")

        # card written, matches columns, valid frontmatter
        card = (out_dir / "README.md").read_text()
        self.assertTrue(card.startswith("---"))
        self.assertIn("Motor Resonance", card)
        self.assertIn("motor_resonance_max_ego_chaos_score", card)
        self.assertIn("louisye/social-robotics-motor-resonance", card)
        self.assertIn("Rows:** 1", card)
        # export metadata present
        self.assertTrue((out_dir / "export_metadata.json").exists())


if __name__ == "__main__":
    unittest.main()
