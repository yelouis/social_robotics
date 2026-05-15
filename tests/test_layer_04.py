import unittest
import json
import logging
import tempfile
from pathlib import Path
import pandas as pd
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import patch
from layer_04_dehydrated_export.aggregator import (
    DataAggregator,
    PANDAS_INFLATION_FACTOR,
    MEM_HEADROOM_FACTOR_DEFAULT,
    MEM_HEADROOM_FACTOR_LARGE,
    LARGE_HOST_BYTES_THRESHOLD,
    _compute_inflation_factor,
)
from layer_04_dehydrated_export.export import DehydratedExporter

class TestDehydratedExport(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        
        # Create dummy manifest matching the REAL production schema
        # (with nested identified_tasks, not a flat task_label)
        self.manifest_path = self.temp_dir_path / "filtered_manifest.json"
        manifest_data = [
            {
                "video_id": "vid1",
                "source_dataset": "Ego4D",
                "duration_sec": 10.0,
                "fps": 30,
                "identified_tasks": [
                    {"task_id": "t_01", "task_label": "Cooking", "task_confidence": 0.9},
                    {"task_id": "t_02", "task_label": "Stirring", "task_confidence": 0.85}
                ],
                "bystander_detections": []
            },
            {
                "video_id": "vid2",
                "source_dataset": "Ego4D",
                "duration_sec": 5.0,
                "fps": 30,
                "identified_tasks": [
                    {"task_id": "t_01", "task_label": "Cleaning", "task_confidence": 0.8}
                ],
                "bystander_detections": []
            }
        ]
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest_data, f)
            
        # Create dummy 03a result (Schema A: aggregate + per_person)
        self.result_03a_path = self.temp_dir_path / "03a_attention_result.json"
        result_03a_data = [
            {
                "video_id": "vid1",
                "layer": "03a_attention",
                "aggregate": {"num_bystanders_tracked": 2, "any_person_engaged": True},
                "per_person": [{"person_id": 0, "average_attention_score": 0.9}]
            }
        ]
        with open(self.result_03a_path, 'w') as f:
            json.dump(result_03a_data, f)
            
        # Create dummy 03b result (Schema B: tasks_analyzed, NOT aggregate/per_person)
        self.result_03b_path = self.temp_dir_path / "03b_reasonable_emotion_result.json"
        result_03b_data = [
            {
                "video_id": "vid1",
                "layer": "03b_reasonable_emotion",
                "tasks_analyzed": [
                    {"task_id": "t_01", "task_aggregate_score": 0.75, "per_person": [{"person_id": 0}]},
                    {"task_id": "t_02", "task_aggregate_score": 0.50, "per_person": [{"person_id": 0}]}
                ]
            },
            {
                "video_id": "vid2",
                "layer": "03b_reasonable_emotion",
                "tasks_analyzed": [
                    {"task_id": "t_01", "task_aggregate_score": -0.25, "per_person": []}
                ]
            }
        ]
        with open(self.result_03b_path, 'w') as f:
            json.dump(result_03b_data, f)

        # Create dummy 03c result (Schema B: tasks_analyzed with prosody_scalar)
        self.result_03c_path = self.temp_dir_path / "03c_acoustic_prosody_result.json"
        result_03c_data = [
            {
                "video_id": "vid1",
                "layer": "03c_acoustic_prosody",
                "tasks_analyzed": [
                    {"task_id": "t_01", "prosody_scalar": 1.0, "classified_acoustic_tone": "Soothing"}
                ]
            }
        ]
        with open(self.result_03c_path, 'w') as f:
            json.dump(result_03c_data, f)

    def tearDown(self):
        self.temp_dir.cleanup()

    # ------------------------------------------------------------------
    #  Aggregation tests
    # ------------------------------------------------------------------
    def test_aggregation_row_count(self):
        """Outer join should produce one row per video across all sources."""
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        self.assertEqual(len(df), 2)

    def test_manifest_task_labels_extracted(self):
        """Real manifest uses nested identified_tasks — verify comma-separated extraction."""
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        vid1_row = df[df['video_id'] == 'vid1'].iloc[0]
        self.assertEqual(vid1_row['task_labels'], "Cooking, Stirring")

    def test_schema_a_aggregate_flattening(self):
        """Layers like 03a with aggregate/per_person should be properly flattened."""
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        vid1_row = df[df['video_id'] == 'vid1'].iloc[0]
        self.assertEqual(vid1_row['03a_attention_num_bystanders_tracked'], 2)
        self.assertTrue(vid1_row['03a_attention_any_person_engaged'])
        # per_person should be JSON-stringified
        self.assertIsInstance(vid1_row['03a_attention_per_person_raw'], str)

    def test_schema_b_tasks_analyzed_flattening(self):
        """Layers like 03b with tasks_analyzed should get JSON-stringified + summary scalar."""
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        vid1_row = df[df['video_id'] == 'vid1'].iloc[0]
        
        # tasks_analyzed should be JSON-stringified
        self.assertIn('03b_reasonable_emotion_tasks_analyzed_raw', df.columns)
        raw = json.loads(vid1_row['03b_reasonable_emotion_tasks_analyzed_raw'])
        self.assertEqual(len(raw), 2)
        
        # Average task score should be computed: (0.75 + 0.50) / 2 = 0.625
        self.assertAlmostEqual(vid1_row['03b_reasonable_emotion_avg_task_score'], 0.625, places=3)

    def test_schema_b_prosody_scalar_extraction(self):
        """03c prosody_scalar should be averaged across tasks."""
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        vid1_row = df[df['video_id'] == 'vid1'].iloc[0]
        self.assertAlmostEqual(vid1_row['03c_acoustic_prosody_avg_prosody_scalar'], 1.0, places=3)

    def test_outer_join_nan_propagation(self):
        """vid2 has no 03a data — those columns should be NaN."""
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        vid2_row = df[df['video_id'] == 'vid2'].iloc[0]
        self.assertTrue(pd.isna(vid2_row['03a_attention_num_bystanders_tracked']))
        # But 03b should be present
        self.assertAlmostEqual(vid2_row['03b_reasonable_emotion_avg_task_score'], -0.25, places=3)

    # ------------------------------------------------------------------
    #  Export metadata tests
    # ------------------------------------------------------------------
    def test_export_metadata_injection(self):
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        df = aggregator.add_export_metadata(df, active_layers=["03a", "03b", "03c"], git_sha="abc1234")
        
        self.assertEqual(df.attrs['pipeline_git_sha'], "abc1234")
        self.assertEqual(df.attrs['active_layers'], ["03a", "03b", "03c"])
        self.assertIn("T", df.attrs['export_timestamp'])  # ISO format check

    # ------------------------------------------------------------------
    #  Dehydration validation tests
    # ------------------------------------------------------------------
    def test_dehydration_rejects_bytes(self):
        """Injecting raw bytes into a column must raise ValueError."""
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        df['invalid_bytes'] = [b"raw_image_data", b""]
        
        exporter = DehydratedExporter(str(self.temp_dir_path))
        with self.assertRaises(ValueError) as context:
            exporter.export_parquet(df)
        self.assertIn("Dehydration validation failed", str(context.exception))

    # ------------------------------------------------------------------
    #  End-to-end export test
    # ------------------------------------------------------------------
    def test_successful_export(self):
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        df = aggregator.add_export_metadata(df, active_layers=["03a", "03b", "03c"])
        
        exporter = DehydratedExporter(str(self.temp_dir_path))
        parquet_path, metadata_path = exporter.export_parquet(df)
        
        self.assertTrue(parquet_path.exists())
        self.assertTrue(metadata_path.exists())
        
        # Verify parquet loads cleanly
        loaded_df = pd.read_parquet(parquet_path)
        self.assertEqual(len(loaded_df), 2)
        
        # Verify metadata
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        self.assertEqual(meta['active_layers'], ["03a", "03b", "03c"])
        self.assertEqual(meta['total_videos'], 2)

    def test_no_corrupt_json_in_layer_files(self):
        """Corrupt JSON in a layer file should be logged and skipped, not crash."""
        corrupt_path = self.temp_dir_path / "03z_fake_result.json"
        with open(corrupt_path, 'w') as f:
            f.write("{invalid json!!! ")

        aggregator = DataAggregator(str(self.temp_dir_path))
        # Should not raise
        df = aggregator.aggregate()
        self.assertEqual(len(df), 2)

    # ------------------------------------------------------------------
    # Resolved Issue 10 — per-layer summary registry
    # ------------------------------------------------------------------
    def test_layer_03f_summary_scalars_surface_as_columns(self):
        """03f motor resonance summary scalars (empathy, mirroring, motor_resonance)
        must appear as Parquet-queryable columns alongside the raw JSON."""
        path = self.temp_dir_path / "03f_motor_resonance_result.json"
        with open(path, 'w') as f:
            json.dump([
                {
                    "video_id": "vid1",
                    "layer": "03f_motor_resonance",
                    "tasks_analyzed": [
                        {
                            "task_id": "t_01",
                            "ego_kinetic_chaos_score": 0.42,
                            "per_person": [
                                {"person_id": 0, "empathy_scalar": 0.8,
                                 "motor_resonance_detected": True,
                                 "mirroring_scalar": 0.6, "mirroring_detected": True},
                                {"person_id": 1, "empathy_scalar": 0.3,
                                 "motor_resonance_detected": False,
                                 "mirroring_scalar": 0.1, "mirroring_detected": False},
                            ],
                        }
                    ],
                }
            ], f)

        df = DataAggregator(str(self.temp_dir_path)).aggregate()
        row = df[df['video_id'] == 'vid1'].iloc[0]
        self.assertAlmostEqual(row['03f_motor_resonance_max_empathy_scalar'], 0.8, places=3)
        self.assertTrue(row['03f_motor_resonance_any_motor_resonance'])
        self.assertAlmostEqual(row['03f_motor_resonance_max_mirroring_scalar'], 0.6, places=3)
        self.assertTrue(row['03f_motor_resonance_any_mirroring'])
        self.assertAlmostEqual(row['03f_motor_resonance_max_ego_chaos_score'], 0.42, places=3)

    def test_layer_03e_summary_scalars_surface_as_columns(self):
        path = self.temp_dir_path / "03e_affirmation_gesture_result.json"
        with open(path, 'w') as f:
            json.dump([
                {
                    "video_id": "vid1",
                    "layer": "03e_affirmation_gesture",
                    "tasks_analyzed": [
                        {
                            "task_id": "t_01",
                            "per_person": [
                                {"person_id": 0, "gesture_detected": "affirming_nod", "confidence": 0.9},
                                {"person_id": 1, "gesture_detected": "none", "confidence": 0.1},
                            ],
                        }
                    ],
                }
            ], f)

        df = DataAggregator(str(self.temp_dir_path)).aggregate()
        row = df[df['video_id'] == 'vid1'].iloc[0]
        self.assertAlmostEqual(row['03e_affirmation_gesture_max_gesture_confidence'], 0.9, places=3)
        self.assertTrue(row['03e_affirmation_gesture_any_nod_detected'])
        self.assertFalse(row['03e_affirmation_gesture_any_shake_detected'])

    def test_layer_03g_summary_scalars_surface_as_columns(self):
        path = self.temp_dir_path / "03g_shared_reality_result.json"
        with open(path, 'w') as f:
            json.dump([
                {
                    "video_id": "vid1",
                    "layer": "03g_shared_reality",
                    "tasks_analyzed": [
                        {"task_id": "t_01",
                         "post_climax_camera_shift_vector": [12.0, 3.0],
                         "bystander_centered_in_fov": True,
                         "social_reference_sought": True}
                    ],
                }
            ], f)

        df = DataAggregator(str(self.temp_dir_path)).aggregate()
        row = df[df['video_id'] == 'vid1'].iloc[0]
        self.assertTrue(row['03g_shared_reality_any_social_reference'])
        self.assertTrue(row['03g_shared_reality_any_bystander_centered'])

    # ------------------------------------------------------------------
    # Resolved Issue 11 — top-level dict flattening
    # ------------------------------------------------------------------
    def test_03a_processing_meta_dict_is_flattened(self):
        """Top-level provenance dicts (e.g. 03a's processing_meta) must surface
        as flattened columns rather than being silently dropped."""
        # Overwrite the 03a result with one that includes processing_meta
        with open(self.result_03a_path, 'w') as f:
            json.dump([{
                "video_id": "vid1",
                "layer": "03a_attention",
                "aggregate": {"num_bystanders_tracked": 2, "any_person_engaged": True},
                "per_person": [{"person_id": 0, "average_attention_score": 0.9}],
                "processing_meta": {
                    "model_used": "l2cs_net_3d_gaze",
                    "sampling_fps_effective": "fixed_8fps",
                },
            }], f)

        df = DataAggregator(str(self.temp_dir_path)).aggregate()
        row = df[df['video_id'] == 'vid1'].iloc[0]
        self.assertEqual(
            row['03a_attention_processing_meta_model_used'], 'l2cs_net_3d_gaze',
        )
        self.assertEqual(
            row['03a_attention_processing_meta_sampling_fps_effective'], 'fixed_8fps',
        )

    # ------------------------------------------------------------------
    # Resolved Issue 14 — schema hash versioning
    # ------------------------------------------------------------------
    def test_schema_version_includes_column_hash(self):
        """add_export_metadata must emit a SemVer+hash schema_version that
        diffs mechanically when columns change."""
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        df = aggregator.add_export_metadata(df, active_layers=["03a", "03b", "03c"])
        self.assertRegex(df.attrs['schema_version'], r'^1\.0\.0\+[0-9a-f]{6}$')

        # Adding a column changes the hash suffix
        df2 = df.copy()
        df2['new_column'] = 0
        df2 = aggregator.add_export_metadata(df2, active_layers=["03a", "03b", "03c"])
        self.assertNotEqual(df.attrs['schema_version'], df2.attrs['schema_version'])

    # ------------------------------------------------------------------
    # Resolved Issue 15 — graceful handling of records without video_id
    # ------------------------------------------------------------------
    def test_record_missing_video_id_is_skipped_not_crash(self):
        """A malformed layer record without video_id must be skipped with a
        warning rather than aborting the entire aggregation run."""
        path = self.temp_dir_path / "03y_partial_result.json"
        with open(path, 'w') as f:
            json.dump([
                {"video_id": "vid1", "layer": "03y", "aggregate": {"score": 0.5}},
                {"layer": "03y", "aggregate": {"score": 0.0}},  # malformed: no video_id
            ], f)

        # Must not raise
        df = DataAggregator(str(self.temp_dir_path)).aggregate()
        # vid1's column is still present despite the malformed sibling record
        row = df[df['video_id'] == 'vid1'].iloc[0]
        self.assertAlmostEqual(row['03y_partial_score'], 0.5, places=3)


    # ------------------------------------------------------------------
    # Resolved Issue 16 — host-adaptive Dask headroom factor
    # ------------------------------------------------------------------
    def test_headroom_factor_uses_large_value_on_64gb_host(self):
        """A 64 GB Mac Studio crosses the 48 GB threshold; with available
        memory just below the 0.7-factor ceiling but above the old 0.5
        ceiling, the aggregator must stay on the Pandas path under the new
        constant (i.e., NOT trigger Dask)."""
        # Constants are wired correctly.
        self.assertGreater(MEM_HEADROOM_FACTOR_LARGE, MEM_HEADROOM_FACTOR_DEFAULT)
        self.assertEqual(LARGE_HOST_BYTES_THRESHOLD, 48 * 1024 ** 3)

        # Craft fake memory so effective_size sits between 0.5*avail and
        # 0.7*avail — under the old factor this would have fired Dask; under
        # the new factor on a 64 GB host it must NOT.
        layer_files = list(self.temp_dir_path.glob("03*_result.json"))
        total_size = self.manifest_path.stat().st_size + sum(f.stat().st_size for f in layer_files)
        effective_size = total_size * PANDAS_INFLATION_FACTOR
        fake_available = int(effective_size / 0.6)  # 0.5 < ratio < 0.7
        fake_vm = type('VM', (), {
            'total': 64 * 1024 ** 3,
            'available': fake_available,
        })()
        with patch('psutil.virtual_memory', return_value=fake_vm):
            # Capture all logs (not just WARNING) so the context manager
            # doesn't fail when no warnings are emitted.
            with self.assertLogs('layer_04_dehydrated_export.aggregator', level='DEBUG') as cm:
                logger = logging.getLogger('layer_04_dehydrated_export.aggregator')
                logger.debug("test marker")  # ensures assertLogs has something
                DataAggregator(str(self.temp_dir_path)).aggregate()
                for msg in cm.output:
                    self.assertNotIn("falling back to Dask", msg)

    def test_headroom_factor_falls_back_to_default_on_small_host(self):
        """A 24 GB Mac mini stays under the 48 GB threshold and must keep the
        original 0.5 factor to preserve the calibration of Resolved Issue 7."""
        # 24 GB host, available memory pushed below effective_size so Dask
        # triggers; this exercises the default-factor branch end-to-end.
        fake_vm = type('VM', (), {
            'total': 24 * 1024 ** 3,
            'available': 100,  # tiny — guarantees the Dask path
        })()
        with patch('psutil.virtual_memory', return_value=fake_vm):
            with self.assertLogs('layer_04_dehydrated_export.aggregator', level='WARNING') as cm:
                try:
                    DataAggregator(str(self.temp_dir_path)).aggregate()
                except Exception:
                    pass  # Dask path may not have it installed; the warning is the contract.
                joined = "\n".join(cm.output)
                self.assertIn("exceeds 50%", joined)

    # ------------------------------------------------------------------
    # Resolved Issue 17 — measurable inflation factor + override hook
    # ------------------------------------------------------------------
    def test_inflation_factor_override_propagates_to_memory_check(self):
        """An explicit ``inflation_factor`` on the aggregator must replace the
        static ``PANDAS_INFLATION_FACTOR`` in the effective_size calculation."""
        agg = DataAggregator(str(self.temp_dir_path), inflation_factor=12.5)
        self.assertEqual(agg.inflation_factor, 12.5)
        # Default constructor still uses the documented constant.
        self.assertEqual(
            DataAggregator(str(self.temp_dir_path)).inflation_factor,
            PANDAS_INFLATION_FACTOR,
        )

    def test_compute_inflation_factor_emits_ratio_and_schema_hash(self):
        """Running the helper on an exported parquet must produce a positive
        ratio paired with the same column-hash shape used by Resolved Issue 14."""
        aggregator = DataAggregator(str(self.temp_dir_path))
        df = aggregator.aggregate()
        df = aggregator.add_export_metadata(df, active_layers=["03a", "03b", "03c"])
        exporter = DehydratedExporter(str(self.temp_dir_path))
        parquet_path, _ = exporter.export_parquet(df)

        result = _compute_inflation_factor(parquet_path)
        self.assertIsNotNone(result)
        factor, schema_hash = result
        self.assertGreater(factor, 0.0)
        self.assertRegex(schema_hash, r'^[0-9a-f]{6}$')

    def test_compute_inflation_factor_returns_none_for_missing_parquet(self):
        result = _compute_inflation_factor(self.temp_dir_path / "does_not_exist.parquet")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
