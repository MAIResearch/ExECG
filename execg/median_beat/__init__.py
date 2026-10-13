"""Median beat: summarize a cohort of ECGs into a single representative beat.

Algorithms live in sibling modules, all implement :class:`MedianBeatExtractor`
and return a :class:`MedianBeat`, so they are interchangeable.

Currently available:
    - :class:`~execg.median_beat.rlign.RlignMedianBeat` -- R-peak aligned
      median beat, with or without heart-rate correction (``scale_method``).

Use :func:`harmonize` to bring raw signals onto a common sampling rate / length
before feeding an extractor.
"""

from execg.median_beat.base import MedianBeat, MedianBeatExtractor
from execg.median_beat.harmonize import (
    LEAD_NAMES_12,
    assemble_leads,
    harmonize,
)
from execg.median_beat.rlign import RlignMedianBeat

__all__ = [
    "MedianBeat",
    "MedianBeatExtractor",
    "RlignMedianBeat",
    "harmonize",
    "assemble_leads",
    "LEAD_NAMES_12",
]
