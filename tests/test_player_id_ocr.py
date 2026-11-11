import numpy as np
import supervision as sv

from player_id_ocr import (
    DEFAULT_ROSTER_MAP,
    PlayerIdentityEllipseAnnotator,
    STRICT_VLM_PROMPT,
    TemporalJerseyVoter,
    VLMJerseyOCR,
    crop_upper_torso,
    parse_jersey_digits,
)


def test_crop_upper_torso_extracts_valid_subregion():
    frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    box = [400.0, 300.0, 500.0, 500.0]  # 100x200 player box
    crop = crop_upper_torso(frame, box)
    assert crop is not None
    # Height should be (0.58 - 0.12) * 200 = 92 px; width should be (1 - 0.20) * 100 = 80 px
    assert crop.shape == (92, 80, 3)
    assert crop_upper_torso(frame, [10.0, 10.0, 15.0, 20.0]) is None


def test_vlm_strict_prompt_and_digit_parsing():
    assert STRICT_VLM_PROMPT == "Read the jersey number on this player. Return only the digits."
    seen_prompts = []

    def fake_vlm(crop, prompt):
        seen_prompts.append(prompt)
        return "Assistant: 10"

    ocr = VLMJerseyOCR(predict_fn=fake_vlm)
    dummy_crop = np.zeros((48, 40, 3), dtype=np.uint8)
    assert ocr.read_number(dummy_crop) == 10
    assert seen_prompts == ["Read the jersey number on this player. Return only the digits."]
    assert parse_jersey_digits("The number is 11.") == 11
    assert parse_jersey_digits("blurry") is None


def test_temporal_voter_filters_blur_misreads_and_maps_roster_names():
    roster = {
        "Team A": {10: "Pito", 11: "Ferrão"},
        "Team B": {9: "Borruto"},
    }
    voter = TemporalJerseyVoter(roster_map=roster, buffer_size=10, min_votes=3, min_agreement=0.5, ocr_stride=1)

    # Simulate VLM predictions for tracker_id=7 with a single blur misread (18 instead of 10)
    assert voter.update(7, 10, frame_index=0) is None  # 1 vote < min_votes(3)
    assert voter.update(7, 18, frame_index=1) is None  # 2 votes < min_votes(3)
    assert voter.update(7, 10, frame_index=2) == 10    # 2/3 votes = 66.7% >= 50% -> mode is 10!
    assert voter.update(7, 10, frame_index=3) == 10

    assert voter.resolve_player_name(7, "team_a") == "Pito"
    assert voter.format_label(7, "team_a", role="player", speed_kmh=12.4) == "#10 Pito | 12.4 km/h"


def test_player_identity_ellipse_annotator_displays_player_name():
    predictions = iter(["11", "11", "11"])
    ocr = VLMJerseyOCR(predict_fn=lambda crop, prompt: next(predictions))
    voter = TemporalJerseyVoter(
        roster_map=DEFAULT_ROSTER_MAP,
        buffer_size=10,
        min_votes=3,
        ocr_stride=1,
    )
    annotator = PlayerIdentityEllipseAnnotator(ocr=ocr, voter=voter)

    frame = np.full((360, 640, 3), 40, dtype=np.uint8)
    detections = sv.Detections(
        xyxy=np.array([[200.0, 100.0, 260.0, 240.0]], dtype=np.float32),
        confidence=np.array([0.95], dtype=np.float32),
        class_id=np.array([0], dtype=int),
        tracker_id=np.array([42], dtype=int),
        data={"role": np.array(["player"])},
    )

    for f_idx in range(3):
        out = annotator.annotate(frame, detections, teams=["team_a"], frame_index=f_idx, speeds_kmh=[14.2])

    assert out.shape == frame.shape
    assert not np.array_equal(out, frame)
    # After 3 frames of voting "11" on Team A, tracker_id=42 resolves to Ferrão (#11)
    assert voter.get_jersey_number(42) == 11
    assert voter.resolve_player_name(42, "team_a") == "Ferrão"
    assert voter.format_label(42, "team_a", speed_kmh=14.2) == "#11 Ferrão | 14.2 km/h"
