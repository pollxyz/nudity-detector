"""
Combined moderation analyzer.

Stacks two complementary models:
  - Detector (NudeNet)            : where in the image are body parts
  - NSFWClassifier (Falconsai)    : is the whole image NSFW yes/no

The classifier is trusted as the primary verdict signal because it is not
gender-biased the way NudeNet is. The detector contributes:
  - granular labels (annotations / explanations)
  - a backup explicit signal at high confidence

A single image goes through both models and a final verdict is computed
from a small set of rules you can tune.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from detector import (
    Detection,
    DetectionResult,
    Detector,
    EXPLICIT_CLASSES,
)
from nsfw_classifier import NSFWClassifier, NSFWScore


@dataclass
class VerdictPolicy:
    """Tunable thresholds. Defaults are conservative for an 18+ profile-pic gate."""
    # Whole-image classifier
    nsfw_block_threshold: float = 0.70   # >= this -> BLOCK from classifier alone
    nsfw_review_threshold: float = 0.30  # >= this -> REVIEW
    # NudeNet
    explicit_block_threshold: float = 0.50   # any explicit class >= this -> BLOCK
    explicit_review_threshold: float = 0.20  # any explicit class >= this -> REVIEW
    # Compensate for NudeNet's female-skewed training data.
    # Male class raw scores typically peak around 0.2-0.4 vs female 0.7-0.9.
    # Multiplying male scores by ~2.5 brings them into the same range so a
    # single threshold can be applied fairly across both genders.
    male_score_boost: float = 2.5
    # Cross-class conflict resolution
    conflict_iou_threshold: float = 0.5  # boxes overlapping >= this conflict
    drop_male_when_female_face: bool = True
    drop_female_when_male_face: bool = True
    # NudeNet's FACE_MALE/FACE_FEMALE classifier is noisy. Only trust a face-gender
    # label above this score — below it the face is treated as gender-unknown.
    face_gender_min_score: float = 0.75


@dataclass
class AnalysisResult:
    verdict: str                       # ALLOW | REVIEW | BLOCK
    reason: str                        # human-readable explanation
    nsfw_score: NSFWScore
    detection_result: DetectionResult
    kept_detections: list[Detection]   # after conflict-resolution
    conflicts: list[str] = field(default_factory=list)  # human-readable conflict notes


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


_CONTRADICTORY_PAIRS = [
    # (class A, class B) -- biologically can't be the same body part
    ("FEMALE_BREAST_EXPOSED", "MALE_GENITALIA_EXPOSED"),
    ("FEMALE_BREAST_COVERED", "MALE_GENITALIA_EXPOSED"),
    ("FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"),
    ("FEMALE_GENITALIA_COVERED", "MALE_GENITALIA_EXPOSED"),
    ("FEMALE_BREAST_EXPOSED", "MALE_BREAST_EXPOSED"),
    ("FEMALE_BREAST_COVERED", "MALE_BREAST_EXPOSED"),
]


def _resolve_conflicts(
    detections: list[Detection],
    policy: VerdictPolicy,
) -> tuple[list[Detection], list[str]]:
    """
    Drop detections that biologically contradict each other in the same region.
    For each contradictory pair with high IoU, keep the higher-scoring one
    UNLESS one is a male class and the other a female class — in that case
    we have no good way to choose so we drop both and flag the conflict.
    """
    conflicts: list[str] = []
    drop_indices: set[int] = set()

    for i, a in enumerate(detections):
        if i in drop_indices:
            continue
        for j in range(i + 1, len(detections)):
            if j in drop_indices:
                continue
            b = detections[j]
            pair = (a.class_name, b.class_name)
            rev = (b.class_name, a.class_name)
            if pair not in _CONTRADICTORY_PAIRS and rev not in _CONTRADICTORY_PAIRS:
                continue
            iou = _iou(a.box, b.box)
            if iou < policy.conflict_iou_threshold:
                continue
            # Same region, contradictory classes — drop both, flag the conflict.
            # The whole-image classifier will determine the actual verdict.
            drop_indices.add(i)
            drop_indices.add(j)
            conflicts.append(
                f"Conflicting detections in same region "
                f"(IoU={iou:.2f}): {a.class_name}@{a.score:.2f} vs "
                f"{b.class_name}@{b.score:.2f} — both dropped, deferring to classifier."
            )

    kept = [d for i, d in enumerate(detections) if i not in drop_indices]
    return kept, conflicts


def _boost_male_scores(
    detections: list[Detection],
    policy: VerdictPolicy,
) -> tuple[list[Detection], list[str]]:
    """
    Multiply MALE_* class raw scores by `policy.male_score_boost` to compensate
    for NudeNet's training-data bias. Returns new Detection list (originals
    untouched) and a one-line note describing what happened.
    """
    if policy.male_score_boost == 1.0:
        return detections, []

    boosted: list[Detection] = []
    n_boosted = 0
    for d in detections:
        if d.class_name.startswith("MALE_"):
            new_score = min(1.0, d.score * policy.male_score_boost)
            boosted.append(Detection(d.class_name, new_score, d.box))
            n_boosted += 1
        else:
            boosted.append(d)

    notes: list[str] = []
    if n_boosted > 0:
        notes.append(
            f"Boosted {n_boosted} male-class score(s) by "
            f"{policy.male_score_boost:.2f}x to compensate for NudeNet bias"
        )
    return boosted, notes


def _apply_face_gender_filter(
    detections: list[Detection],
    policy: VerdictPolicy,
) -> tuple[list[Detection], list[str]]:
    """If a confident face-gender label exists, suppress opposite-gender body parts.

    NudeNet's face-gender classifier is unreliable below high confidence, so we
    only act on FACE_MALE / FACE_FEMALE detections at or above
    `face_gender_min_score`. We also only emit a pipeline note when something
    was actually filtered out — otherwise the note is just noise to the user.
    """
    notes: list[str] = []
    min_score = policy.face_gender_min_score
    has_male_face = any(
        d.class_name == "FACE_MALE" and d.score >= min_score for d in detections
    )
    has_female_face = any(
        d.class_name == "FACE_FEMALE" and d.score >= min_score for d in detections
    )

    # Don't filter if BOTH genders are present (multi-person image).
    if has_male_face and has_female_face:
        return detections, notes

    suppress: set[str] = set()
    label = ""
    if has_male_face and policy.drop_female_when_male_face:
        suppress.update({
            "FEMALE_BREAST_EXPOSED", "FEMALE_BREAST_COVERED",
            "FEMALE_GENITALIA_EXPOSED", "FEMALE_GENITALIA_COVERED",
        })
        label = "MALE"
    elif has_female_face and policy.drop_male_when_female_face:
        suppress.update({"MALE_GENITALIA_EXPOSED", "MALE_BREAST_EXPOSED"})
        label = "FEMALE"

    if not suppress:
        return detections, notes

    kept = [d for d in detections if d.class_name not in suppress]
    suppressed_count = len(detections) - len(kept)
    if suppressed_count > 0:
        opposite = "female" if label == "MALE" else "male"
        notes.append(
            f"Detected {label} face — suppressed {suppressed_count} "
            f"{opposite}-anatomy detection(s)"
        )
    return kept, notes


def _compute_verdict(
    nsfw: NSFWScore,
    detections: list[Detection],
    policy: VerdictPolicy,
) -> tuple[str, str]:
    """Returns (verdict, reason)."""
    explicit_dets = [d for d in detections if d.class_name in EXPLICIT_CLASSES]
    top_explicit = max(explicit_dets, key=lambda d: d.score, default=None)

    # Hard block from either signal
    if nsfw.nsfw >= policy.nsfw_block_threshold:
        return "BLOCK", f"Whole-image NSFW score {nsfw.nsfw:.2f} >= {policy.nsfw_block_threshold}"
    if top_explicit and top_explicit.score >= policy.explicit_block_threshold:
        return "BLOCK", (
            f"NudeNet detected {top_explicit.class_name} at {top_explicit.score:.2f} "
            f">= {policy.explicit_block_threshold}"
        )

    # Review band
    if nsfw.nsfw >= policy.nsfw_review_threshold:
        return "REVIEW", f"Whole-image NSFW score {nsfw.nsfw:.2f} in review band"
    if top_explicit and top_explicit.score >= policy.explicit_review_threshold:
        return "REVIEW", (
            f"NudeNet detected {top_explicit.class_name} at {top_explicit.score:.2f} "
            f"in review band"
        )

    return "ALLOW", f"Whole-image NSFW {nsfw.nsfw:.2f}, no explicit body parts above threshold"


class CombinedAnalyzer:
    def __init__(
        self,
        *,
        detector: Detector | None = None,
        classifier: NSFWClassifier | None = None,
        policy: VerdictPolicy | None = None,
    ) -> None:
        self.detector = detector or Detector(inference_resolution=640)
        self.classifier = classifier or NSFWClassifier()
        self.policy = policy or VerdictPolicy()

    def analyze(self, image: str | Path | np.ndarray) -> AnalysisResult:
        detection_result = self.detector.detect(image)
        nsfw_score = self.classifier.classify(image)

        # Pipeline order:
        #   1) Boost male-class scores (compensate for NudeNet bias)
        #   2) Face-based opposite-gender suppression
        #   3) Cross-class conflict resolution (drop both on contradiction)
        boosted, boost_notes = _boost_male_scores(
            detection_result.detections, self.policy
        )
        face_filtered, face_notes = _apply_face_gender_filter(
            boosted, self.policy
        )
        deduped, conflict_notes = _resolve_conflicts(face_filtered, self.policy)

        verdict, reason = _compute_verdict(nsfw_score, deduped, self.policy)

        return AnalysisResult(
            verdict=verdict,
            reason=reason,
            nsfw_score=nsfw_score,
            detection_result=detection_result,
            kept_detections=deduped,
            conflicts=boost_notes + face_notes + conflict_notes,
        )
