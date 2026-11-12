"""Top-level convenience re-export of Epics 2 & 3 (`reid_boundary`)."""

from futsal.reid_boundary import (
    InPlayBoundaryFilter,
    PlayerIdentityManager,
    PlayerProfile,
    SigLIPEmbedder,
    SubstitutionZone,
)

__all__ = [
    "InPlayBoundaryFilter",
    "PlayerIdentityManager",
    "PlayerProfile",
    "SigLIPEmbedder",
    "SubstitutionZone",
]
