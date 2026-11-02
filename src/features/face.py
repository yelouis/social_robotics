from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


OPENFACE_COLUMNS = (
    # Gaze (8)
    "gaze_0_x", "gaze_0_y", "gaze_0_z",
    "gaze_1_x", "gaze_1_y", "gaze_1_z",
    "gaze_angle_x", "gaze_angle_y",
    # Head rotation (3)
    "pose_Rx", "pose_Ry", "pose_Rz",
    # Continuous AU intensity (17)
    "AU01_r", "AU02_r", "AU04_r", "AU05_r", "AU06_r", "AU07_r",
    "AU09_r", "AU10_r", "AU12_r", "AU14_r", "AU15_r", "AU17_r",
    "AU20_r", "AU23_r", "AU25_r", "AU26_r", "AU45_r",
    # Binary AU presence (18)
    "AU01_c", "AU02_c", "AU04_c", "AU05_c", "AU06_c", "AU07_c",
    "AU09_c", "AU10_c", "AU12_c", "AU14_c", "AU15_c", "AU17_c",
    "AU20_c", "AU23_c", "AU25_c", "AU26_c", "AU28_c", "AU45_c",
)

FACETORCH_COLUMNS = ("Valence", "Arousal")

BODY_COLUMNS = ("Crossed Arms", "Arms behind back")


def get_reflex_face_feature_names() -> List[str]:
    """Return the ordered list of 153 feature names for reflex-face-v1."""
    names: List[str] = []
    # 1. OpenFace 46 columns: mean, std, max
    for c in OPENFACE_COLUMNS:
        names.append(f"{c}_mean")
    for c in OPENFACE_COLUMNS:
        names.append(f"{c}_std")
    for c in OPENFACE_COLUMNS:
        names.append(f"{c}_max")
    # Angular velocity variability (3 dims)
    names.extend(["pose_Rx_vel_std", "pose_Ry_vel_std", "pose_Rz_vel_std"])
    # 2. Facetorch 2 columns: mean, std, max
    for c in FACETORCH_COLUMNS:
        names.append(f"{c}_mean")
    for c in FACETORCH_COLUMNS:
        names.append(f"{c}_std")
    for c in FACETORCH_COLUMNS:
        names.append(f"{c}_max")
    # 3. Body 2 columns: mean, std, max
    for c in BODY_COLUMNS:
        c_clean = c.replace(" ", "_")
        names.append(f"{c_clean}_mean")
    for c in BODY_COLUMNS:
        c_clean = c.replace(" ", "_")
        names.append(f"{c_clean}_std")
    for c in BODY_COLUMNS:
        c_clean = c.replace(" ", "_")
        names.append(f"{c_clean}_max")
    if len(names) != 153:
        raise ValueError(f"Expected 153 feature names, got {len(names)}")
    return names


class ReflexFaceEncoder:
    encoder_id: str = "reflex-face-v1"

    def __init__(self) -> None:
        # Cache per participant: participant_dir -> { "openface": ..., "facetorch": ..., "body": ... }
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _load_participant_data(self, participant_dir: Path) -> Dict[str, Any]:
        p_key = str(participant_dir.resolve())
        if p_key in self._cache:
            return self._cache[p_key]

        # 1. Load openface.csv (1-indexed frame)
        openface_path = participant_dir / "openface.csv"
        of_records: List[Dict[str, Any]] = []
        if openface_path.is_file():
            with open(openface_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned = {k.strip(): v.strip() for k, v in row.items() if k is not None and v is not None}
                    try:
                        f_idx = int(float(cleaned.get("frame", "-1")))
                        success = int(float(cleaned.get("success", "0")))
                        conf = float(cleaned.get("confidence", "0.0"))
                    except ValueError:
                        continue

                    # Only store valid rows
                    if success == 1 and conf >= 0.7:
                        try:
                            vals = [float(cleaned[col]) for col in OPENFACE_COLUMNS]
                            of_records.append({
                                "frame": f_idx,
                                "vals": vals,
                                "pose": [float(cleaned["pose_Rx"]), float(cleaned["pose_Ry"]), float(cleaned["pose_Rz"])],
                            })
                        except (KeyError, ValueError):
                            continue

        # 2. Load facetorch.csv (0-indexed Frame ID)
        facetorch_path = participant_dir / "facetorch.csv"
        ft_records: List[Dict[str, Any]] = []
        if facetorch_path.is_file():
            with open(facetorch_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned = {k.strip(): v.strip() for k, v in row.items() if k is not None and v is not None}
                    try:
                        f_0idx = int(float(cleaned.get("Frame ID", "-1")))
                        val_str = cleaned.get("Valence", "")
                        aro_str = cleaned.get("Arousal", "")
                        if val_str and aro_str and val_str.upper() != "NA" and aro_str.upper() != "NA":
                            v_val = float(val_str)
                            a_val = float(aro_str)
                            if np.isfinite(v_val) and np.isfinite(a_val):
                                ft_records.append({"frame_0idx": f_0idx, "vals": [v_val, a_val]})
                    except ValueError:
                        continue

        # 3. Load body.csv (0-indexed Frame)
        body_path = participant_dir / "body.csv"
        body_records: List[Dict[str, Any]] = []
        if body_path.is_file():
            with open(body_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cleaned = {k.strip(): v.strip() for k, v in row.items() if k is not None and v is not None}
                    try:
                        f_0idx = int(float(cleaned.get("Frame", "-1")))
                        c_arms = cleaned.get("Crossed Arms", "").upper()
                        b_arms = cleaned.get("Arms behind back", "").upper()
                        if c_arms in ("TRUE", "FALSE") and b_arms in ("TRUE", "FALSE"):
                            v1 = 1.0 if c_arms == "TRUE" else 0.0
                            v2 = 1.0 if b_arms == "TRUE" else 0.0
                            body_records.append({"frame_0idx": f_0idx, "vals": [v1, v2]})
                    except ValueError:
                        continue

        data = {
            "openface": of_records,
            "facetorch": ft_records,
            "body": body_records,
        }
        self._cache[p_key] = data
        return data

    def encode_item(self, item: Any) -> np.ndarray:
        p_dir = Path(item.video_path).parent
        rsf, ref = item.meta["reaction_frames"]
        return self.encode(p_dir, rsf, ref)

    def encode_window(
        self,
        video_path: Union[str, Path],
        window: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        p_dir = Path(video_path).parent
        if meta and "reaction_frames" in meta:
            rsf, ref = meta["reaction_frames"]
        else:
            raise ValueError("ReflexFaceEncoder requires meta['reaction_frames']")
        return self.encode(p_dir, rsf, ref)

    def encode(self, participant_dir: Union[str, Path], rsf: int, ref: int) -> np.ndarray:
        p_dir = Path(participant_dir)
        data = self._load_participant_data(p_dir)

        # 1. OpenFace (141 dims)
        of_matches = [
            r for r in data["openface"]
            if rsf <= r["frame"] <= ref
        ]
        if not of_matches:
            raise RuntimeError(f"no valid OpenFace frames in [{rsf}, {ref}]")

        of_mat = np.array([r["vals"] for r in of_matches], dtype=np.float32)  # (M, 46)
        of_mean = np.mean(of_mat, axis=0)
        of_std = np.std(of_mat, axis=0, ddof=0)
        of_max = np.max(of_mat, axis=0)

        # Angular velocity variability (3 dims)
        if len(of_matches) >= 2:
            pose_mat = np.array([r["pose"] for r in of_matches], dtype=np.float32)  # (M, 3)
            vel_std = np.std(np.diff(pose_mat, axis=0), axis=0, ddof=0)
        else:
            vel_std = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        openface_141d = np.concatenate([of_mean, of_std, of_max, vel_std])

        # 2. Facetorch (6 dims) - 0-indexed: [rsf - 1, ref - 1]
        ft_matches = [
            r for r in data["facetorch"]
            if (rsf - 1) <= r["frame_0idx"] <= (ref - 1)
        ]
        if not ft_matches:
            raise RuntimeError(f"no valid Facetorch frames in [{rsf - 1}, {ref - 1}]")

        ft_mat = np.array([r["vals"] for r in ft_matches], dtype=np.float32)  # (K, 2)
        ft_mean = np.mean(ft_mat, axis=0)
        ft_std = np.std(ft_mat, axis=0, ddof=0)
        ft_max = np.max(ft_mat, axis=0)
        facetorch_6d = np.concatenate([ft_mean, ft_std, ft_max])

        # 3. Body posture (6 dims) - 0-indexed: [rsf - 1, ref - 1]
        body_matches = [
            r for r in data["body"]
            if (rsf - 1) <= r["frame_0idx"] <= (ref - 1)
        ]
        if not body_matches:
            raise RuntimeError(f"no valid Body frames in [{rsf - 1}, {ref - 1}]")

        body_mat = np.array([r["vals"] for r in body_matches], dtype=np.float32)  # (L, 2)
        body_mean = np.mean(body_mat, axis=0)
        body_std = np.std(body_mat, axis=0, ddof=0)
        body_max = np.max(body_mat, axis=0)
        body_6d = np.concatenate([body_mean, body_std, body_max])

        # Total 153 dims
        feat = np.concatenate([openface_141d, facetorch_6d, body_6d]).astype(np.float32)

        if not np.all(np.isfinite(feat)):
            raise RuntimeError("non-finite values in reflex-face-v1 vector")

        if feat.shape != (153,):
            raise RuntimeError(f"Unexpected shape for reflex-face-v1: {feat.shape} != (153,)")

        return feat


class ReflexSpeechEncoder:
    encoder_id: str = "reflex-speech-v1"

    def __init__(self) -> None:
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def _load_participant_speech(self, participant_dir: Path) -> List[Dict[str, Any]]:
        p_key = str(participant_dir.resolve())
        if p_key in self._cache:
            return self._cache[p_key]

        speech_path = participant_dir / "speech.csv"
        records: List[Dict[str, Any]] = []
        if speech_path.is_file():
            with open(speech_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    header = [h.strip() for h in header]
                    id_idx = 0
                    b_idx = 1
                    e_idx = 2
                    for idx, h in enumerate(header):
                        hl = h.lower()
                        if hl == "id":
                            id_idx = idx
                        elif hl in ("begintime", "begin time"):
                            b_idx = idx
                        elif hl in ("endtime", "end time"):
                            e_idx = idx

                    # Columns 6 to 53 inclusive are Hume prosody dimensions (48 dims)
                    for row in reader:
                        if len(row) < 54:
                            continue
                        speaker = row[id_idx].strip().lower()
                        if speaker not in {"participant", "robot/participant"}:
                            continue
                        try:
                            b_time = float(row[b_idx].strip())
                            e_time = float(row[e_idx].strip())
                            prosody = [float(val.strip()) for val in row[6:54]]
                        except ValueError:
                            continue

                        if len(prosody) == 48 and np.all(np.isfinite(prosody)):
                            records.append({
                                "speaker": speaker,
                                "begin_time": b_time,
                                "end_time": e_time,
                                "prosody": prosody,
                            })

        self._cache[p_key] = records
        return records

    def encode_item(self, item: Any) -> np.ndarray:
        p_dir = Path(item.video_path).parent
        wr_s, wr_e = item.meta["wall_reaction_window_sec"]
        return self.encode(p_dir, wr_s, wr_e)

    def encode_window(
        self,
        video_path: Union[str, Path],
        window: List[float],
        meta: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        p_dir = Path(video_path).parent
        if meta and "wall_reaction_window_sec" in meta:
            wr_s, wr_e = meta["wall_reaction_window_sec"]
        else:
            raise ValueError("ReflexSpeechEncoder requires meta['wall_reaction_window_sec']")
        return self.encode(p_dir, wr_s, wr_e)

    def encode(self, participant_dir: Union[str, Path], wr_s: float, wr_e: float) -> np.ndarray:
        p_dir = Path(participant_dir)
        records = self._load_participant_speech(p_dir)

        matches = [
            r for r in records
            if r["begin_time"] <= wr_e and r["end_time"] >= wr_s
        ]
        if not matches:
            raise RuntimeError("no participant speech in reaction window")

        prosody_mat = np.array([r["prosody"] for r in matches], dtype=np.float32)  # (U, 48)
        p_mean = np.mean(prosody_mat, axis=0)
        p_max = np.max(prosody_mat, axis=0)

        feat = np.concatenate([p_mean, p_max]).astype(np.float32)  # (96,)

        if not np.all(np.isfinite(feat)):
            raise RuntimeError("non-finite values in reflex-speech-v1 vector")

        if feat.shape != (96,):
            raise RuntimeError(f"Unexpected shape for reflex-speech-v1: {feat.shape} != (96,)")

        return feat
