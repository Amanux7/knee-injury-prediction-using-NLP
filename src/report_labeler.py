"""
RSNA Knee Abnormality Detection — Radiology Report Labeler
===========================================================
Extracts binary / probabilistic labels for the 12 competition targets from
free‑text radiology reports.

Two extraction backends
-----------------------
1. **LLM‑based** (``method="llm"``): Uses a HuggingFace zero‑shot
   classification pipeline (e.g. ``facebook/bart-large-mnli``,
   ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract``, or any
   compatible checkpoint).  Requires a model download on first use.
2. **Regex / keyword fallback** (``method="regex"``): A deterministic,
   CPU‑only extractor built on curated clinical keyword dictionaries with
   negation awareness.  Runs offline on Kaggle kernels with zero dependencies
   beyond the standard library + ``re``.

Primary API
-----------
>>> from src.report_labeler import extract_labels_from_report
>>> scores = extract_labels_from_report(
...     "Complete ACL tear with moderate joint effusion.",
...     method="regex",
... )
>>> scores["ACL"]
0.95

Design notes
~~~~~~~~~~~~
* **Negation detection**: phrases like *"no evidence of"*, *"without"*,
  *"intact"*, *"rule out"* suppress the score for a target that would
  otherwise match, reducing it to a low‑confidence value (0.1) rather than
  hard zero so downstream models can still learn from ambiguity.
* **Multilingual support** (regex path): a curated set of French, Spanish,
  German, and Portuguese clinical synonyms is included.  This is deliberately
  not exhaustive — the LLM path handles unseen languages far better.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Module‑level constants & logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

TARGET_COLUMNS: List[str] = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

# Confidence values emitted by the regex extractor
_HIGH_CONF: float = 0.95
_MED_CONF: float = 0.70
_NEGATED_CONF: float = 0.10
_ABSENT_CONF: float = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# 1. Keyword / Regex Dictionaries
# ═══════════════════════════════════════════════════════════════════════════

# Each entry maps a target column to a list of regex patterns (case‑insensitive).
# Patterns are ordered roughly by specificity — most specific first.
# Multilingual synonyms (FR / ES / DE / PT) are appended at the end of each list.

_KEYWORD_PATTERNS: Dict[str, List[str]] = {
    "ACL": [
        # English
        r"\bacl\b",
        r"\banterior\s+cruciate\s+ligament\b",
        r"\banterior\s+cruciate\b",
        # French
        r"\blca\b",                          # ligament croisé antérieur
        r"\bligament\s+crois[ée]\s+ant[ée]rieur\b",
        # Spanish
        r"\bligamento\s+cruzado\s+anterior\b",
        # German
        r"\bvorderes?\s+kreuzband\b",
    ],
    "MCL": [
        r"\bmcl\b",
        r"\bmedial\s+collateral\s+ligament\b",
        r"\bmedial\s+collateral\b",
        r"\btibial\s+collateral\s+ligament\b",
        # French
        r"\bligament\s+collat[ée]ral\s+m[ée]dial\b",
        r"\blcm\b",
        # Spanish
        r"\bligamento\s+colateral\s+medial\b",
    ],
    "Medial Meniscus": [
        r"\bmedial\s+menisc(?:us|al|i)\b",
        r"\bmm\s+tear\b",
        r"\bmedial\s+meniscal\s+tear\b",
        r"\bmedial\s+meniscal\b",
        # French
        r"\bm[ée]nisque\s+m[ée]dial\b",
        r"\bm[ée]nisque\s+interne\b",
        # Spanish
        r"\bmenisco\s+medial\b",
        # German
        r"\binnenmeniskus\b",
    ],
    "Lateral Meniscus": [
        r"\blateral\s+menisc(?:us|al|i)\b",
        r"\blm\s+tear\b",
        r"\blateral\s+meniscal\s+tear\b",
        r"\blateral\s+meniscal\b",
        # French
        r"\bm[ée]nisque\s+lat[ée]ral\b",
        r"\bm[ée]nisque\s+externe\b",
        # Spanish
        r"\bmenisco\s+lateral\b",
        # German
        r"\bau[sß]enmeniskus\b",
    ],
    "Medial OA": [
        r"\bmedial\s+(?:compartment\s+)?osteoarthr(?:itis|osis|opathy)\b",
        r"\bmedial\s+compartment\s+(?:oa|narrowing|degenerat(?:ion|ive))\b",
        r"\bmedial\s+oa\b",
        r"\bmedial\s+joint\s+space\s+(?:narrowing|loss)\b",
        r"\bmedial\s+tibiofemoral\s+(?:oa|degenerat)\b",
        # French
        r"\barthrose\s+m[ée]diale\b",
        r"\bgonarthrose\s+m[ée]diale\b",
        # Spanish
        r"\bartrosis\s+medial\b",
    ],
    "Lateral OA": [
        r"\blateral\s+(?:compartment\s+)?osteoarthr(?:itis|osis|opathy)\b",
        r"\blateral\s+compartment\s+(?:oa|narrowing|degenerat(?:ion|ive))\b",
        r"\blateral\s+oa\b",
        r"\blateral\s+joint\s+space\s+(?:narrowing|loss)\b",
        r"\blateral\s+tibiofemoral\s+(?:oa|degenerat)\b",
        # French
        r"\barthrose\s+lat[ée]rale\b",
        # Spanish
        r"\bartrosis\s+lateral\b",
    ],
    "PF OA": [
        r"\bpatellofemoral\s+(?:oa|osteoarthr(?:itis|osis)|degenerat(?:ion|ive)|narrowing)\b",
        r"\bpf\s+oa\b",
        r"\bpf\s+osteoarthr\b",
        r"\bpf\s+joint\s+(?:narrowing|degenerat)\b",
        r"\bpf\s+compartment\b",
        r"\bretropatellar\s+(?:cartilage|chondro|degenerat)\b",
        # French
        r"\barthrose\s+f[ée]moro\s*-?\s*patellaire\b",
        # Spanish
        r"\bartrosis\s+(?:patelofemoral|femoropatelar)\b",
    ],
    "Effusion": [
        r"\beffusion\b",
        r"\bjoint\s+fluid\b",
        r"\bhydrarthrosis\b",
        r"\bsuprapatellar\s+(?:fluid|effusion|pouch\s+fluid)\b",
        # French
        r"\b[ée]panchement\b",
        r"\bhydarthrose\b",
        # Spanish
        r"\bderrame\s+articular\b",
        r"\bderrame\b",
        # German
        r"\bgelenkerguss\b",
        r"\berguss\b",
        # Portuguese
        r"\bderrame\b",
    ],
    "Synovitis": [
        r"\bsynovitis\b",
        r"\bsynovial\s+(?:thickening|inflammation|hypertroph|proliferat|enhancement)\b",
        # French
        r"\bsynovite\b",
        # Spanish
        r"\bsinovitis\b",
        # German
        r"\bsynovialitis\b",
    ],
    "Baker's": [
        r"\bbaker'?s?\s+cyst\b",
        r"\bpopliteal\s+cyst\b",
        r"\bpopliteal\s+(?:fluid|collection)\b",
        # French
        r"\bkyste\s+(?:de\s+)?baker\b",
        r"\bkyste\s+poplit[ée]\b",
        # Spanish
        r"\bquiste\s+(?:de\s+)?baker\b",
        r"\bquiste\s+popl[ií]teo\b",
        # German
        r"\bbakerzyste\b",
        r"\bpoplitealzyste\b",
    ],
    "Contusion": [
        r"\bcontusion\b",
        r"\bbone\s+(?:bruise|contusion|marrow\s+(?:edema|oedema|contusion))\b",
        r"\bsubchondral\s+(?:edema|oedema|bruise)\b",
        r"\bmarrow\s+(?:edema|oedema)\b",
        # French
        r"\bcontusion\s+osseuse\b",
        r"\b[oœ]d[èe]me\s+(?:osseux|m[ée]dullaire)\b",
        # Spanish
        r"\bcontusi[oó]n\s+[oó]sea\b",
        r"\bedema\s+[oó]seo\b",
        # German
        r"\bknochenkontusion\b",
        r"\bknochenmarksödem\b",
    ],
    "Fracture": [
        r"\bfracture\b",
        r"\bfx\b",
        r"\bfractur(?:ed|ing)\b",
        r"\bstress\s+fracture\b",
        r"\binsufficiency\s+fracture\b",
        r"\bavulsion\b",
        r"\btibial\s+plateau\s+fracture\b",
        # French
        r"\bfracture\b",
        # Spanish
        r"\bfractura\b",
        # German
        r"\bfraktur\b",
        r"\bbruch\b",
        # Portuguese
        r"\bfratura\b",
    ],
}

# ---------------------------------------------------------------------------
# Negation cues — matched in a window *before* the keyword hit
# ---------------------------------------------------------------------------
_NEGATION_PATTERNS: List[str] = [
    r"\bno\s+(?:evidence\s+(?:of|for)\s+)?",
    r"\bno\b",
    r"\bnot?\s+(?:seen|identified|demonstrated|present|noted)\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\brule[sd]?\s+out\b",
    r"\bunlikely\b",
    r"\bnegative\s+for\b",
    r"\bintact\b",
    r"\bnormal\b",
    r"\bpreserved\b",
    r"\bdenies\b",
    # French
    r"\bpas\s+(?:de|d')\b",
    r"\babsence\s+(?:de|d')\b",
    r"\bsans\b",
    # Spanish
    r"\bsin\b",
    r"\bno\s+se\s+(?:observa|evidencia|identifica)\b",
    r"\bausencia\s+de\b",
    # German
    r"\bkein(?:e|er|en|em)?\b",
    r"\bohne\b",
]

# Pre‑compile for performance
_NEGATION_RE: re.Pattern[str] = re.compile(
    "|".join(f"(?:{p})" for p in _NEGATION_PATTERNS),
    flags=re.IGNORECASE,
)

# The negation window: how many characters before/after a keyword match to inspect
_NEG_WINDOW: int = 60
_NEG_WINDOW_AFTER: int = 40

# Post‑keyword negation cues (e.g. "ACL is intact", "meniscus appears normal")
_POST_NEGATION_PATTERNS: List[str] = [
    r"\b(?:is|are|appears?|remains?|was|were)\s+(?:intact|normal|preserved|unremarkable|stable)\b",
    r"\bintact\b",
    r"\bnormal\b",
    r"\bpreserved\b",
    r"\bunremarkable\b",
]
_POST_NEGATION_RE: re.Pattern[str] = re.compile(
    "|".join(f"(?:{p})" for p in _POST_NEGATION_PATTERNS),
    flags=re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Regex / Keyword Extractor
# ═══════════════════════════════════════════════════════════════════════════

def _is_negated(text: str, match_start: int, match_end: int) -> bool:
    """Check whether a keyword hit is negated by surrounding context.

    Inspects both a *preceding* window (e.g. "no evidence of **ACL** tear")
    and a *trailing* window (e.g. "**ACL** is intact").

    Parameters
    ----------
    text : str
        The full (lowered) report text.
    match_start : int
        Character index where the keyword pattern match begins.
    match_end : int
        Character index where the keyword pattern match ends.

    Returns
    -------
    bool
        ``True`` if a negation cue is found in either surrounding window.
    """
    # --- Preceding window ("no ACL tear") ---
    window_start: int = max(0, match_start - _NEG_WINDOW)
    preceding: str = text[window_start:match_start]
    if _NEGATION_RE.search(preceding) is not None:
        return True

    # --- Trailing window ("ACL is intact") ---
    window_end: int = min(len(text), match_end + _NEG_WINDOW_AFTER)
    trailing: str = text[match_end:window_end]
    if _POST_NEGATION_RE.search(trailing) is not None:
        return True

    return False


def regex_extract_labels(report_text: str) -> Dict[str, float]:
    """Deterministic, offline label extraction via keyword / regex matching.

    For every target, we scan the report for any matching pattern.  If a
    match is found *and* is **not** negated, the target receives a high
    confidence score.  If negated, it receives a low (but non‑zero) score.
    Targets with no keyword match at all receive a near‑zero baseline.

    Parameters
    ----------
    report_text : str
        Free‑text radiology report (any language).

    Returns
    -------
    Dict[str, float]
        Mapping from target name → confidence ∈ [0.0, 1.0].
    """
    text_lower: str = report_text.lower()
    scores: Dict[str, float] = {}

    for target, patterns in _KEYWORD_PATTERNS.items():
        best_score: float = _ABSENT_CONF  # default when nothing matches

        for pat in patterns:
            for m in re.finditer(pat, text_lower):
                if _is_negated(text_lower, m.start(), m.end()):
                    # Negated mention — low confidence, but keep searching
                    # for a potential non‑negated mention elsewhere.
                    best_score = max(best_score, _NEGATED_CONF)
                else:
                    # Positive (non‑negated) mention — high confidence.
                    best_score = _HIGH_CONF
                    break  # no need to check more patterns for this target

            if best_score >= _HIGH_CONF:
                break  # early exit — already at ceiling

        scores[target] = best_score

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# 3. LLM‑based Extractor (HuggingFace zero‑shot classification)
# ═══════════════════════════════════════════════════════════════════════════

def _build_candidate_labels() -> List[str]:
    """Create descriptive hypothesis sentences for zero‑shot classification.

    The zero‑shot pipeline classifies the report against each hypothesis
    independently.  Using full clinical sentences as candidate labels
    dramatically improves recall over bare noun phrases.
    """
    return [
        "anterior cruciate ligament (ACL) tear or injury",
        "medial collateral ligament (MCL) tear or injury",
        "medial meniscus tear or degeneration",
        "lateral meniscus tear or degeneration",
        "medial compartment osteoarthritis",
        "lateral compartment osteoarthritis",
        "patellofemoral osteoarthritis",
        "joint effusion or fluid accumulation",
        "synovitis or synovial inflammation",
        "Baker's cyst or popliteal cyst",
        "bone contusion or bone marrow edema",
        "fracture or stress fracture",
    ]


def llm_extract_labels(
    report_text: str,
    model_name: str = "facebook/bart-large-mnli",
    device: int = -1,
) -> Dict[str, float]:
    """Extract labels using a HuggingFace zero‑shot classification pipeline.

    Parameters
    ----------
    report_text : str
        Free‑text radiology report.
    model_name : str
        HuggingFace model identifier.  Recommended options:
        - ``facebook/bart-large-mnli`` (general‑purpose, ~1.6 GB)
        - ``microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract``
        - Any NLI‑finetuned checkpoint compatible with
          ``pipeline("zero-shot-classification")``.
    device : int
        PyTorch device ordinal (``-1`` = CPU, ``0`` = first GPU).

    Returns
    -------
    Dict[str, float]
        Mapping from target name → confidence ∈ [0.0, 1.0].

    Raises
    ------
    RuntimeError
        If the ``transformers`` library is not installed or the model cannot
        be loaded.  The caller should fall back to :func:`regex_extract_labels`.
    """
    try:
        from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "HuggingFace `transformers` is required for LLM extraction.  "
            "Install it with: pip install transformers"
        ) from exc

    logger.info(
        "Loading zero‑shot classification pipeline ('%s', device=%d)…",
        model_name,
        device,
    )

    classifier: Any = hf_pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )

    candidate_labels: List[str] = _build_candidate_labels()

    # multi_label=True means every hypothesis is scored independently
    result: Dict[str, Any] = classifier(
        report_text,
        candidate_labels,
        multi_label=True,
    )

    # Map hypothesis labels back to our canonical target column names
    label_score_pairs: List[Tuple[str, float]] = list(
        zip(result["labels"], result["scores"])
    )

    # Build ordered output keyed by TARGET_COLUMNS
    scores: Dict[str, float] = {}
    for target, hypothesis in zip(TARGET_COLUMNS, candidate_labels):
        for label, score in label_score_pairs:
            if label == hypothesis:
                scores[target] = float(score)
                break
        else:
            # Hypothesis not found in results (should not happen)
            scores[target] = 0.05

    return scores


# ═══════════════════════════════════════════════════════════════════════════
# 4. Unified Public API
# ═══════════════════════════════════════════════════════════════════════════

def extract_labels_from_report(
    report_text: str,
    method: str = "regex",
    model_name: str = "facebook/bart-large-mnli",
    device: int = -1,
) -> Dict[str, float]:
    """Extract abnormality probabilities from a radiology report.

    This is the **primary entry point** for the module.  It dispatches to
    either the LLM or regex backend and always returns a dictionary with
    exactly 12 keys matching :data:`TARGET_COLUMNS`.

    Parameters
    ----------
    report_text : str
        Raw radiology report text.
    method : {"regex", "llm"}
        Extraction backend.
    model_name : str
        HuggingFace model identifier (only used when ``method="llm"``).
    device : int
        PyTorch device ordinal (only used when ``method="llm"``).

    Returns
    -------
    Dict[str, float]
        ``{target_name: confidence}`` with values in ``[0.0, 1.0]``.
    """
    if not report_text or not report_text.strip():
        logger.warning("Empty report received — returning baseline scores.")
        return {t: _ABSENT_CONF for t in TARGET_COLUMNS}

    if method == "llm":
        try:
            return llm_extract_labels(
                report_text,
                model_name=model_name,
                device=device,
            )
        except RuntimeError:
            logger.warning(
                "LLM extraction failed — falling back to regex extractor."
            )
            return regex_extract_labels(report_text)

    elif method == "regex":
        return regex_extract_labels(report_text)

    else:
        raise ValueError(
            f"Unknown extraction method '{method}'. Use 'regex' or 'llm'."
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. CLI Test Runner
# ═══════════════════════════════════════════════════════════════════════════

def _print_scores_table(
    title: str,
    scores: Dict[str, float],
    width: int = 62,
) -> None:
    """Pretty-print a scores dictionary as an aligned table."""
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")
    print(f"  {'Target':<26s}  {'Confidence':>10s}  {'Bar'}")
    print(f"  {'-' * 24}  {'-' * 10}  {'-' * 20}")
    for target, score in scores.items():
        bar: str = "#" * int(score * 20)
        print(f"  {target:<26s}  {score:>10.3f}  {bar}")
    print(f"{'-' * width}")


def main() -> None:
    """Run the report labeler on a set of test cases and print results."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    # ------------------------------------------------------------------
    # Test reports -- English, French, Spanish, and an edge-case negation
    # ------------------------------------------------------------------
    test_reports: List[Tuple[str, str]] = [
        (
            "English - positive multi-finding",
            "Complete tear of the ACL with moderate joint effusion and "
            "complex medial meniscal tear.  Mild patellofemoral "
            "osteoarthritis.  Small Baker's cyst noted.",
        ),
        (
            "English - negated findings",
            "The ACL is intact.  No evidence of meniscal tear.  No "
            "effusion.  Normal medial and lateral compartments without "
            "osteoarthritis.  No fracture identified.",
        ),
        (
            "English - mixed positive / negative",
            "Grade II MCL sprain.  Bone contusion of the lateral tibial "
            "plateau with associated marrow edema.  No fracture.  Mild "
            "synovitis.  The ACL and lateral meniscus are normal.",
        ),
        (
            "French - positive report",
            "Rupture compl\u00e8te du ligament crois\u00e9 ant\u00e9rieur.  \u00c9panchement "
            "articulaire mod\u00e9r\u00e9.  L\u00e9sion du m\u00e9nisque m\u00e9dial.",
        ),
        (
            "Spanish - positive report",
            "Fractura del platillo tibial lateral con edema \u00f3seo.  "
            "Derrame articular moderado.  Quiste de Baker.",
        ),
        (
            "Empty report (edge case)",
            "",
        ),
    ]

    print("\n" + "=" * 62)
    print("  REPORT LABELER -- REGEX BACKEND TEST SUITE")
    print("=" * 62)

    for title, report in test_reports:
        display: str = report[:80] + ("..." if len(report) > 80 else "")
        print(f"\n  [Report]: \"{display}\"")
        scores: Dict[str, float] = extract_labels_from_report(
            report, method="regex"
        )
        _print_scores_table(f"[REGEX] {title}", scores)

    # ------------------------------------------------------------------
    # Optional: LLM backend (only if transformers is installed)
    # ------------------------------------------------------------------
    try:
        from transformers import pipeline as _  # noqa: F401

        llm_available: bool = True
    except ImportError:
        llm_available = False

    if llm_available and "--llm" in sys.argv:
        print("\n" + "=" * 62)
        print("  REPORT LABELER -- LLM BACKEND TEST")
        print("=" * 62)
        # Run just the first report through the LLM path
        llm_report: str = test_reports[0][1]
        print(f"\n  [Report]: \"{llm_report[:80]}...\"")
        llm_scores: Dict[str, float] = extract_labels_from_report(
            llm_report, method="llm"
        )
        _print_scores_table("[LLM] English - positive multi-finding", llm_scores)
    elif not llm_available:
        print(
            "\n  [INFO] Skipping LLM test -- `transformers` not installed.  "
            "Run with `pip install transformers` and `--llm` flag to enable."
        )


if __name__ == "__main__":
    main()
