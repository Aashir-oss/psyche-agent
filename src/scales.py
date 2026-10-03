"""GAD-7 and PHQ-9 screening scales — deterministic, no LLM, no network.

Both scales are free for all uses (public domain / freely licensed).
Scores are SCREENING SIGNALS only — never diagnoses.
"""
from __future__ import annotations

# ---------------------------------------------------------------- items ---
GAD7_STEM = ("Over the last 2 weeks, how often have you been bothered "
             "by the following problems?")

GAD7_ITEMS = [
    "Feeling nervous, anxious or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen",
]

# PHQ-9 uses the same stem as GAD-7.
PHQ9_STEM = GAD7_STEM

PHQ9_ITEMS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let "
    "yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or "
    "watching television",
    "Moving or speaking so slowly that other people could have noticed? Or "
    "the opposite — being so fidgety or restless that you have been moving "
    "around a lot more than usual",
    "Thoughts that you would be better off dead, or of hurting yourself "
    "in some way",
]

# 0-indexed position of the self-harm item inside PHQ9_ITEMS.
PHQ9_ITEM9_INDEX = 8

# ---------------------------------------------------------------- choices --
CHOICES = [
    "Not at all",
    "Several days",
    "More than half the days",
    "Nearly every day",
]

IMPAIRMENT_QUESTION = (
    "If you checked off any problems, how difficult have these problems made "
    "it for you to do your work, take care of things at home, or get along "
    "with other people?"
)

IMPAIRMENT_CHOICES = [
    "Not difficult at all",
    "Somewhat difficult",
    "Very difficult",
    "Extremely difficult",
]

_DISCLAIMER_SENTENCE = "This is a screening signal, not a diagnosis."

# ---------------------------------------------------------------- scoring --
def score_scale(answers: list[int]) -> int:
    """Sum of the 0-3 item answers. Pure arithmetic — no interpretation."""
    if len(answers) not in (7, 9):
        raise ValueError(
            f"score_scale needs 7 (GAD-7) or 9 (PHQ-9) answers, got {len(answers)}")
    return sum(int(a) for a in answers)


def _band_for(score: int, bands: list[tuple[int, int, str, str]]) -> tuple[str, str]:
    """bands: (lo, hi, label, note). Returns (label, note)."""
    for lo, hi, label, note in bands:
        if lo <= score <= hi:
            return label, note + " " + _DISCLAIMER_SENTENCE
    # score outside every band: clamp to the nearest one
    if score < bands[0][0]:
        label, note = bands[0][2], bands[0][3]
    else:
        label, note = bands[-1][2], bands[-1][3]
    return label, note + " " + _DISCLAIMER_SENTENCE


_GAD7_BANDS = [
    (0, 4, "Minimal",
     "Your anxiety symptoms look minimal right now. Keep doing what helps you stay steady."),
    (5, 9, "Mild",
     "Your anxiety symptoms look mild. Small daily habits — regular sleep, "
     "movement, talking to someone you trust — can make a real difference."),
    (10, 14, "Moderate",
     "Your anxiety symptoms look moderate. Consider working through the coping "
     "tools and talking to someone you trust or a professional."),
    (15, 21, "Severe",
     "Your anxiety symptoms look severe. Please consider reaching out to a "
     "qualified mental-health professional soon — support helps."),
]

_PHQ9_BANDS = [
    (0, 4, "Minimal",
     "Your mood symptoms look minimal right now. Keep doing what helps you stay steady."),
    (5, 9, "Mild",
     "Your mood symptoms look mild. Small daily habits — daylight, movement, "
     "connection with people you trust — can make a real difference."),
    (10, 14, "Moderate",
     "Your mood symptoms look moderate. Consider working through the coping "
     "tools and talking to someone you trust or a professional."),
    (15, 19, "Moderately severe",
     "Your mood symptoms look moderately severe. Please consider talking to a "
     "qualified mental-health professional — support helps."),
    (20, 27, "Severe",
     "Your mood symptoms look severe. Please reach out to a qualified "
     "mental-health professional soon — you deserve support."),
]


def gad7_band(score: int) -> tuple[str, str]:
    """(label, note) for a GAD-7 total (0-21)."""
    return _band_for(int(score), _GAD7_BANDS)


def phq9_band(score: int) -> tuple[str, str]:
    """(label, note) for a PHQ-9 total (0-27)."""
    return _band_for(int(score), _PHQ9_BANDS)


# ---------------------------------------------------------------- safety ---
def item9_positive(scale: str, answers: list[int]) -> bool:
    """True only when this is a PHQ-9 AND item 9 (self-harm) scored above 0.

    Deterministic triage: callers must handle a positive result with the
    crisis path BEFORE anything else.
    """
    if scale != "PHQ9":
        return False
    if not answers or len(answers) <= PHQ9_ITEM9_INDEX:
        return False
    return int(answers[PHQ9_ITEM9_INDEX]) > 0


# ------------------------------------------------------------ graph bands --
# (lo, hi, label, color) — passed to graphs.scale_trend for band shading.
GAD7_GRAPH_BANDS = [
    (0, 4, "Minimal", "#c8e6c9"),
    (5, 9, "Mild", "#fff9c4"),
    (10, 14, "Moderate", "#ffe0b2"),
    (15, 21, "Severe", "#f5b7b1"),
]

PHQ9_GRAPH_BANDS = [
    (0, 4, "Minimal", "#c8e6c9"),
    (5, 9, "Mild", "#e8f5c8"),
    (10, 14, "Moderate", "#fff9c4"),
    (15, 19, "Moderately severe", "#ffe0b2"),
    (20, 27, "Severe", "#f5b7b1"),
]
