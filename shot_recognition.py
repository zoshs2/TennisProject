import os
import sys
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np


@dataclass
class _ShotCounter:
    min_frames_between_shots: int = 60
    confidence_threshold: float = 0.98
    nb_forehands: int = 0
    nb_backhands: int = 0
    nb_serves: int = 0
    last_shot: str = "neutral"
    frames_since_last_shot: int = 60

    def update(self, probs: np.ndarray) -> None:
        if probs[0] > self.confidence_threshold and self.frames_since_last_shot > self.min_frames_between_shots:
            self.nb_backhands += 1
            self.last_shot = "backhand"
            self.frames_since_last_shot = 0
        elif probs[1] > self.confidence_threshold and self.frames_since_last_shot > self.min_frames_between_shots:
            self.nb_forehands += 1
            self.last_shot = "forehand"
            self.frames_since_last_shot = 0
        elif len(probs) > 3 and probs[3] > self.confidence_threshold and self.frames_since_last_shot > self.min_frames_between_shots:
            self.nb_serves += 1
            self.last_shot = "serve"
            self.frames_since_last_shot = 0

        self.frames_since_last_shot += 1


class RNNShotRecognizer:
    """
    Adapter that reuses `tennis_shot_recognition` RNN inference logic inside TennisProject.
    """

    def __init__(
        self,
        shot_project_dir: str,
        model_path: str,
        left_handed: bool = False,
        window_size: int = 30,
        confidence_threshold: float = 0.98,
        min_frames_between_shots: int = 60,
        active_shot_window: int = 30,
    ):
        self.shot_project_dir = os.path.abspath(shot_project_dir)
        self.model_path = model_path
        self.left_handed = left_handed
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        self.min_frames_between_shots = min_frames_between_shots
        self.active_shot_window = active_shot_window

        if self.shot_project_dir not in sys.path:
            sys.path.insert(0, self.shot_project_dir)

        from tensorflow import keras  # noqa: WPS433 (lazy import by design)
        from extract_human_pose import HumanPoseExtractor  # noqa: WPS433 (dynamic import by design)

        self._keras = keras
        self._HumanPoseExtractor = HumanPoseExtractor
        self._model = self._keras.models.load_model(self.model_path)
        self.window_size = self._resolve_window_size(self.window_size)

    def _resolve_window_size(self, requested_window_size: int) -> int:
        """
        Ensure runtime window size matches the loaded model's expected sequence length.
        """
        try:
            input_shape = getattr(self._model, "input_shape", None)
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            expected_window = None
            if input_shape and len(input_shape) >= 2:
                expected_window = input_shape[1]
            if isinstance(expected_window, int) and expected_window > 0:
                if expected_window != requested_window_size:
                    print(
                        f"Warning: overriding shot_window_size={requested_window_size} "
                        f"with model-required value {expected_window}",
                    )
                return expected_window
        except Exception:
            pass
        return requested_window_size

    def infer(self, frames: List[np.ndarray]) -> List[Dict[str, Any]]:
        if not frames:
            return []

        old_cwd = os.getcwd()
        os.chdir(self.shot_project_dir)
        try:
            human_pose_extractor = self._HumanPoseExtractor(frames[0].shape)
        finally:
            os.chdir(old_cwd)

        shot_counter = _ShotCounter(
            min_frames_between_shots=self.min_frames_between_shots,
            confidence_threshold=self.confidence_threshold,
        )
        features_pool = []
        results: List[Dict[str, Any]] = []

        for frame in frames:
            human_pose_extractor.extract(frame)
            human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

            features = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
            if self.left_handed:
                features[:, 1] = 1 - features[:, 1]
            features = features[features[:, 2] > 0][:, 0:2]

            probs = np.zeros(4, dtype=np.float32)
            if features.shape[0] == 13:
                features_pool.append(features.reshape(1, 26))
                if len(features_pool) == self.window_size:
                    features_seq = np.array(features_pool).reshape(1, self.window_size, 26)
                    probs = np.array(self._model.__call__(features_seq)[0], dtype=np.float32)
                    shot_counter.update(probs)
                    features_pool = features_pool[1:]

            if hasattr(human_pose_extractor, "roi") and hasattr(human_pose_extractor, "keypoints_pixels_frame"):
                human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

            active_shot = (
                shot_counter.last_shot
                if shot_counter.frames_since_last_shot < self.active_shot_window and shot_counter.last_shot != "neutral"
                else "neutral"
            )
            results.append(
                {
                    "shot": active_shot,
                    "probs": probs.tolist(),
                    "counts": {
                        "backhand": shot_counter.nb_backhands,
                        "forehand": shot_counter.nb_forehands,
                        "serve": shot_counter.nb_serves,
                    },
                }
            )

        return results
