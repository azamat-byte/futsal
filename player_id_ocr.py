"""Top-level convenience re-export of `futsal.player_id_ocr`."""

from futsal.player_id_ocr import (
    DEFAULT_ROSTER_MAP,
    PlayerIdentityEllipseAnnotator,
    STRICT_VLM_PROMPT,
    TemporalJerseyVoter,
    VLMJerseyOCR,
    crop_upper_torso,
    normalize_team_key,
    parse_jersey_digits,
)

__all__ = [
    "DEFAULT_ROSTER_MAP",
    "PlayerIdentityEllipseAnnotator",
    "STRICT_VLM_PROMPT",
    "TemporalJerseyVoter",
    "VLMJerseyOCR",
    "crop_upper_torso",
    "normalize_team_key",
    "parse_jersey_digits",
]
