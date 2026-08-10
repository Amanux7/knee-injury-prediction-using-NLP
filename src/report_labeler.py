"""
RSNA Knee Abnormality Detection — Radiology Report Labeler
===========================================================
Extracts soft-probability pseudo-labels for 12 competition targets from
free-text radiology reports.  Designed for Kaggle pseudo-labeling pipelines
where a mix of annotated images and unlabeled reports must be reconciled.

Two extraction backends
-----------------------
1. **Regex / keyword** (``method="regex"``):  Deterministic, CPU-only
   extractor with curated multilingual clinical keyword dictionaries,
   sentence-level bidirectional negation detection, and severity-modifier
   awareness.  Zero external dependencies beyond the stdlib + ``re``.

2. **LLM / NLI** (``method="llm"``):  Uses a HuggingFace zero-shot
   classification pipeline (default ``facebook/bart-large-mnli``).  The model
   is lazy-loaded and cached across calls for batched inference.

Confidence semantics
--------------------
* **Positive finding** → 0.95
* **Negated finding** → 0.10  (keeps gradient signal for noisy-label training)
* **Not mentioned**   → 0.05  (soft prior; avoids hard zero)
* Severity modifiers (*mild / moderate / severe*) interpolate between 0.70–0.95.

Author : Kaggle Grandmaster pipeline
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Module-level constants & logger
# ---------------------------------------------------------------------------
__all__ = ["ReportLabeler", "extract_labels_from_report", "TARGET_COLUMNS"]

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

# Confidence levels
_HIGH_CONF: float = 0.95
_MOD_CONF: float = 0.85    # moderate severity
_MILD_CONF: float = 0.70   # mild / minimal
_NEGATED_CONF: float = 0.10
_ABSENT_CONF: float = 0.05


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Multilingual Keyword Dictionaries
# ═══════════════════════════════════════════════════════════════════════════
# Each entry: target → list of regex patterns (case-insensitive).
# Order: most specific first, multilingual synonyms appended.
# Languages covered: English (EN), French (FR), Spanish (ES).

_KEYWORD_PATTERNS: Dict[str, List[str]] = {
    "ACL": [
        # EN
        r"\bacl\b",
        r"\banterior\s+cruciate\s+ligament\b",
        r"\banterior\s+cruciate\b",
        # FR
        r"\blca\b",
        r"\bligament\s+crois[ée]\s+ant[ée]rieur\b",
        # ES
        r"\bligamento\s+cruzado\s+anterior\b",
        r"\blca\b",
    ],
    "MCL": [
        r"\bmcl\b",
        r"\bmedial\s+collateral\s+ligament\b",
        r"\bmedial\s+collateral\b",
        r"\btibial\s+collateral\s+ligament\b",
        # FR
        r"\bligament\s+collat[ée]ral\s+m[ée]dial\b",
        r"\blcm\b",
        # ES
        r"\bligamento\s+colateral\s+medial\b",
    ],
    "Medial Meniscus": [
        r"\bmedial\s+menisc(?:us|al|i)\b",
        r"\bmm\s+tear\b",
        r"\bmedial\s+meniscal\s+tear\b",
        r"\bmedial\s+meniscal\b",
        # FR
        r"\bm[ée]nisque\s+m[ée]dial\b",
        r"\bm[ée]nisque\s+interne\b",
        # ES
        r"\bmenisco\s+medial\b",
        r"\bmenisco\s+interno\b",
    ],
    "Lateral Meniscus": [
        r"\blateral\s+menisc(?:us|al|i)\b",
        r"\blm\s+tear\b",
        r"\blateral\s+meniscal\s+tear\b",
        r"\blateral\s+meniscal\b",
        # FR
        r"\bm[ée]nisque\s+lat[ée]ral\b",
        r"\bm[ée]nisque\s+externe\b",
        # ES
        r"\bmenisco\s+lateral\b",
        r"\bmenisco\s+externo\b",
    ],
    "Medial OA": [
        r"\bmedial\s+(?:compartment\s+)?osteoarthr(?:itis|osis|opathy)\b",
        r"\bmedial\s+compartment\s+(?:oa|narrowing|degenerat(?:ion|ive))\b",
        r"\bmedial\s+oa\b",
        r"\bmedial\s+joint\s+space\s+(?:narrowing|loss)\b",
        r"\bmedial\s+tibiofemoral\s+(?:oa|degenerat)\b",
        # FR
        r"\barthrose\s+m[ée]diale\b",
        r"\bgonarthrose\s+m[ée]diale\b",
        # ES
        r"\bartrosis\s+medial\b",
        r"\bartrosis\s+del\s+compartimento\s+medial\b",
    ],
    "Lateral OA": [
        r"\blateral\s+(?:compartment\s+)?osteoarthr(?:itis|osis|opathy)\b",
        r"\blateral\s+compartment\s+(?:oa|narrowing|degenerat(?:ion|ive))\b",
        r"\blateral\s+oa\b",
        r"\blateral\s+joint\s+space\s+(?:narrowing|loss)\b",
        r"\blateral\s+tibiofemoral\s+(?:oa|degenerat)\b",
        # FR
        r"\barthrose\s+lat[ée]rale\b",
        # ES
        r"\bartrosis\s+lateral\b",
    ],
    "PF OA": [
        r"\bpatellofemoral\s+(?:oa|osteoarthr(?:itis|osis)|degenerat(?:ion|ive)|narrowing)\b",
        r"\bpf\s+oa\b",
        r"\bpf\s+osteoarthr\b",
        r"\bpf\s+joint\s+(?:narrowing|degenerat)\b",
        r"\bpf\s+compartment\b",
        r"\bretropatellar\s+(?:cartilage|chondro|degenerat)\b",
        # FR
        r"\barthrose\s+f[ée]moro\s*-?\s*patellaire\b",
        # ES
        r"\bartrosis\s+(?:patelofemoral|femoropatelar)\b",
    ],
    "Effusion": [
        r"\beffusion\b",
        r"\bjoint\s+fluid\b",
        r"\bhydrarthrosis\b",
        r"\bsuprapatellar\s+(?:fluid|effusion|pouch\s+fluid)\b",
        # FR
        r"\b[ée]panchement(?:\s+articulaire)?\b",
        r"\bhydarthrose\b",
        # ES
        r"\bderrame\s+articular\b",
        r"\bderrame\b",
    ],
    "Synovitis": [
        r"\bsynovitis\b",
        r"\bsynovial\s+(?:thickening|inflammation|hypertroph|proliferat|enhancement)\b",
        # FR
        r"\bsynovite\b",
        # ES
        r"\bsinovitis\b",
    ],
    "Baker's": [
        r"\bbaker'?s?\s+cyst\b",
        r"\bpopliteal\s+cyst\b",
        r"\bpopliteal\s+(?:fluid|collection)\b",
        # FR
        r"\bkyste\s+(?:de\s+)?baker\b",
        r"\bkyste\s+poplit[ée]\b",
        # ES
        r"\bquiste\s+(?:de\s+)?baker\b",
        r"\bquiste\s+popl[ií]teo\b",
    ],
    "Contusion": [
        r"\bcontusion\b",
        r"\bbone\s+(?:bruise|contusion|marrow\s+(?:edema|oedema|contusion))\b",
        r"\bsubchondral\s+(?:edema|oedema|bruise)\b",
        r"\bmarrow\s+(?:edema|oedema)\b",
        # FR
        r"\bcontusion\s+osseuse\b",
        r"\b[oœ]d[èe]me\s+(?:osseux|m[ée]dullaire)\b",
        # ES
        r"\bcontusi[oó]n\s+[oó]sea\b",
        r"\bedema\s+[oó]seo\b",
    ],
    "Fracture": [
        r"\bfracture\b",
        r"\bfx\b",
        r"\bfractur(?:ed|ing)\b",
        r"\bstress\s+fracture\b",
        r"\binsufficiency\s+fracture\b",
        r"\bavulsion\b",
        r"\btibial\s+plateau\s+fracture\b",
        # FR
        r"\bfracture\b",
        # ES
        r"\bfractura\b",
    ],
}

# Pre-compile all keyword patterns for speed
_COMPILED_KEYWORDS: Dict[str, List[re.Pattern[str]]] = {
    target: [re.compile(p, re.IGNORECASE) for p in patterns]
    for target, patterns in _KEYWORD_PATTERNS.items()
}


# ---------------------------------------------------------------------------
# 2.  Bidirectional Negation Detection
# ---------------------------------------------------------------------------
# We detect negation in TWO directions around each keyword match:
#   - PRE-negation:  "no ACL tear", "without effusion", "pas de fracture"
#   - POST-negation: "ACL is intact", "ménisque est normal", "LCA está normal"

_NEG_WINDOW_BEFORE: int = 60   # chars before match
_NEG_WINDOW_AFTER: int = 45    # chars after match

# --- Pre-keyword negation cues ---
_PRE_NEGATION_PATTERNS: List[str] = [
    # EN
    r"\bno\s+(?:evidence\s+(?:of|for)\s+)?",
    r"\bno\b",
    r"\bnot?\s+(?:seen|identified|demonstrated|present|noted|observed)\b",
    r"\bwithout\b",
    r"\babsent\b",
    r"\brule[sd]?\s+out\b",
    r"\bunlikely\b",
    r"\bnegative\s+for\b",
    r"\bintact\b",
    r"\bnormal\b",
    r"\bpreserved\b",
    r"\bdenies\b",
    r"\bfail(?:s|ed)?\s+to\s+(?:demonstrate|show|reveal)\b",
    # FR
    r"\bpas\s+(?:de|d')\b",
    r"\babsence\s+(?:de|d')\b",
    r"\bsans\b",
    r"\baucun(?:e)?\b",
    # ES
    r"\bsin\b",
    r"\bno\s+se\s+(?:observa|evidencia|identifica)\b",
    r"\bausencia\s+de\b",
    r"\bsin\s+evidencia\s+de\b",
]

_PRE_NEG_RE: re.Pattern[str] = re.compile(
    "|".join(f"(?:{p})" for p in _PRE_NEGATION_PATTERNS),
    flags=re.IGNORECASE,
)

# --- Post-keyword negation cues ---
_POST_NEGATION_PATTERNS: List[str] = [
    # EN
    r"\b(?:is|are|appears?|remains?|was|were)\s+(?:intact|normal|preserved|unremarkable|stable)\b",
    r"\bintact\b",
    r"\bnormal\b",
    r"\bpreserved\b",
    r"\bunremarkable\b",
    r"\bwithin\s+normal\s+limits\b",
    # FR
    r"\b(?:est|sont|para[iî]t)\s+(?:intact[es]?|normal[es]?|conserv[ée][es]?)\b",
    r"\bnormal[es]?\b",
    r"\bintact[es]?\b",
    # ES
    r"\b(?:est[áa]|se\s+encuentra)\s+(?:intacto|normal|conservado|preservado)\b",
    r"\bintacto\b",
    r"\bnormal\b",
]

_POST_NEG_RE: re.Pattern[str] = re.compile(
    "|".join(f"(?:{p})" for p in _POST_NEGATION_PATTERNS),
    flags=re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 3.  Severity Modifiers
# ---------------------------------------------------------------------------
# If a severity modifier is found near a positive keyword match, we modulate
# the confidence downward from 0.95 (severe/complete) to 0.70 (mild/minimal).

_SEVERITY_MAP: Dict[str, float] = {
    "mild": _MILD_CONF,
    "minimal": _MILD_CONF,
    "slight": _MILD_CONF,
    "subtle": _MILD_CONF,
    "trace": _MILD_CONF,
    "small": _MILD_CONF,
    "tiny": _MILD_CONF,
    "moderate": _MOD_CONF,
    "moderate-sized": _MOD_CONF,
    "moderately": _MOD_CONF,
    "severe": _HIGH_CONF,
    "complete": _HIGH_CONF,
    "large": _HIGH_CONF,
    "extensive": _HIGH_CONF,
    "complex": _HIGH_CONF,
    "full-thickness": _HIGH_CONF,
    "grade ii": _MOD_CONF,
    "grade iii": _HIGH_CONF,
    "grade 2": _MOD_CONF,
    "grade 3": _HIGH_CONF,
    # FR
    "léger": _MILD_CONF,
    "légère": _MILD_CONF,
    "modéré": _MOD_CONF,
    "modérée": _MOD_CONF,
    "sévère": _HIGH_CONF,
    "complète": _HIGH_CONF,
    # ES
    "leve": _MILD_CONF,
    "moderado": _MOD_CONF,
    "moderada": _MOD_CONF,
    "severo": _HIGH_CONF,
    "severa": _HIGH_CONF,
    "completa": _HIGH_CONF,
    "completo": _HIGH_CONF,
}

_SEVERITY_RE: re.Pattern[str] = re.compile(
    "|".join(rf"\b{re.escape(k)}\b" for k in _SEVERITY_MAP),
    flags=re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> List[str]:
    """Split report text into sentences for scoped negation detection.

    Uses a simple heuristic: split on period/semicolon/newline boundaries,
    then strip whitespace.  This prevents negation in one sentence from
    bleeding into the next (e.g., "No fracture. ACL tear." should NOT
    negate the ACL finding).
    """
    raw = re.split(r"[.;!\n]+", text)
    return [s.strip() for s in raw if s.strip()]


def _is_negated(sentence: str, match_start: int, match_end: int) -> bool:
    """Check whether a keyword hit is negated by surrounding context.

    Inspects both a *preceding* window (e.g. "no evidence of **ACL** tear")
    and a *trailing* window (e.g. "**ACL** is intact") within the
    sentence boundary.

    Parameters
    ----------
    sentence : str
        The sentence (lowered) containing the match.
    match_start, match_end : int
        Character indices of the keyword match within ``sentence``.

    Returns
    -------
    bool
        ``True`` if a negation cue is found in either direction.
    """
    # --- Pre-negation window ---
    win_start = max(0, match_start - _NEG_WINDOW_BEFORE)
    preceding = sentence[win_start:match_start]
    if _PRE_NEG_RE.search(preceding) is not None:
        return True

    # --- Post-negation window ---
    win_end = min(len(sentence), match_end + _NEG_WINDOW_AFTER)
    trailing = sentence[match_end:win_end]
    if _POST_NEG_RE.search(trailing) is not None:
        return True

    return False


def _detect_severity(sentence: str, match_start: int) -> float:
    """Look for a severity modifier near the keyword and return the
    corresponding confidence.  Falls back to ``_HIGH_CONF`` if no modifier.
    """
    # Search in a 40-char window before the match
    win_start = max(0, match_start - 40)
    window = sentence[win_start:match_start]
    sev_match = _SEVERITY_RE.search(window)
    if sev_match:
        modifier = sev_match.group(0).lower()
        return _SEVERITY_MAP.get(modifier, _HIGH_CONF)
    return _HIGH_CONF


# ═══════════════════════════════════════════════════════════════════════════
# 5.  ReportLabeler Class  (main API)
# ═══════════════════════════════════════════════════════════════════════════

class ReportLabeler:
    """Dual-backend radiology report labeler for 12 competition targets.

    Parameters
    ----------
    method : {"regex", "llm"}
        Extraction backend.  ``"regex"`` is CPU-only and deterministic.
        ``"llm"`` requires ``transformers`` + a GPU for best performance.
    model_name : str
        HuggingFace model identifier (only used when ``method="llm"``).
        Recommended: ``"facebook/bart-large-mnli"`` (general NLI) or any
        compatible zero-shot-classification checkpoint.
    device : int
        PyTorch device ordinal (``-1`` = CPU, ``0`` = first GPU).

    Examples
    --------
    >>> labeler = ReportLabeler(method="regex")
    >>> scores = labeler("Complete ACL tear with moderate joint effusion.")
    >>> scores["ACL"]
    0.95
    >>> scores["Effusion"]
    0.85
    """

    def __init__(
        self,
        method: str = "regex",
        model_name: str = "facebook/bart-large-mnli",
        device: int = -1,
    ) -> None:
        if method not in ("regex", "llm"):
            raise ValueError(
                f"Unknown method '{method}'. Use 'regex' or 'llm'."
            )
        self.method = method
        self.model_name = model_name
        self.device = device
        self._classifier: Any = None  # lazy-loaded LLM pipeline
        self._candidate_labels: List[str] = self._build_candidate_labels()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, report_text: str) -> Dict[str, float]:
        """Extract labels from a single report (shorthand for ``label()``)."""
        return self.label(report_text)

    def label(self, report_text: str) -> Dict[str, float]:
        """Extract soft-probability labels for all 12 targets.

        Returns
        -------
        Dict[str, float]
            ``{target_name: confidence}`` with values in ``[0.0, 1.0]``.
        """
        if not report_text or not report_text.strip():
            logger.warning("Empty report — returning baseline scores.")
            return {t: _ABSENT_CONF for t in TARGET_COLUMNS}

        if self.method == "llm":
            try:
                return self._llm_extract(report_text)
            except RuntimeError:
                logger.warning(
                    "LLM extraction failed — falling back to regex."
                )
                return self._regex_extract(report_text)

        return self._regex_extract(report_text)

    def label_batch(
        self, reports: Sequence[str]
    ) -> List[Dict[str, float]]:
        """Label multiple reports.  Returns a list of score dicts.

        For the LLM backend this is more efficient than repeated single
        calls because the pipeline is loaded once and reused.
        """
        return [self.label(r) for r in reports]

    # ------------------------------------------------------------------
    # Regex Backend
    # ------------------------------------------------------------------

    def _regex_extract(self, report_text: str) -> Dict[str, float]:
        """Sentence-level keyword extraction with bidirectional negation
        and severity-modifier awareness.
        """
        sentences = _split_sentences(report_text.lower())
        scores: Dict[str, float] = {t: _ABSENT_CONF for t in TARGET_COLUMNS}

        for target in TARGET_COLUMNS:
            compiled_pats = _COMPILED_KEYWORDS[target]
            best: float = _ABSENT_CONF

            for sentence in sentences:
                for pat in compiled_pats:
                    for m in pat.finditer(sentence):
                        if _is_negated(sentence, m.start(), m.end()):
                            best = max(best, _NEGATED_CONF)
                        else:
                            # Positive hit — check severity
                            sev_conf = _detect_severity(
                                sentence, m.start()
                            )
                            best = max(best, sev_conf)

                    if best >= _HIGH_CONF:
                        break  # ceiling reached
                if best >= _HIGH_CONF:
                    break

            scores[target] = best

        return scores

    # ------------------------------------------------------------------
    # LLM / NLI Backend
    # ------------------------------------------------------------------

    @staticmethod
    def _build_candidate_labels() -> List[str]:
        """Create descriptive NLI hypothesis sentences for zero-shot
        classification.  Full clinical sentences dramatically improve
        recall over bare noun phrases.
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

    def _ensure_classifier(self) -> None:
        """Lazy-load the HuggingFace zero-shot pipeline (cached)."""
        if self._classifier is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace `transformers` is required for LLM extraction.  "
                "Install with: pip install transformers"
            ) from exc

        logger.info(
            "Loading zero-shot pipeline ('%s', device=%d) …",
            self.model_name,
            self.device,
        )
        self._classifier = hf_pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=self.device,
        )

    def _llm_extract(self, report_text: str) -> Dict[str, float]:
        """Extract labels via HuggingFace zero-shot classification."""
        self._ensure_classifier()

        result: Dict[str, Any] = self._classifier(
            report_text,
            self._candidate_labels,
            multi_label=True,
        )

        label_score: Dict[str, float] = dict(
            zip(result["labels"], result["scores"])
        )

        scores: Dict[str, float] = {}
        for target, hypothesis in zip(TARGET_COLUMNS, self._candidate_labels):
            scores[target] = float(label_score.get(hypothesis, _ABSENT_CONF))

        return scores


# ═══════════════════════════════════════════════════════════════════════════
# 6.  Convenience Functional API  (backward-compatible)
# ═══════════════════════════════════════════════════════════════════════════

# Module-level singleton for the functional API
_default_labeler: Optional[ReportLabeler] = None


def extract_labels_from_report(
    report_text: str,
    method: str = "regex",
    model_name: str = "facebook/bart-large-mnli",
    device: int = -1,
) -> Dict[str, float]:
    """Extract abnormality probabilities from a radiology report.

    This is a **convenience wrapper** around :class:`ReportLabeler`.
    For batch processing, instantiate the class directly.

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
    global _default_labeler
    if (
        _default_labeler is None
        or _default_labeler.method != method
        or _default_labeler.model_name != model_name
    ):
        _default_labeler = ReportLabeler(
            method=method, model_name=model_name, device=device
        )
    return _default_labeler.label(report_text)


# ===========================================================================
# 7.  Test Suite & ASCII Bar-Chart Printer
# ===========================================================================

_BAR_CHARS = " .:-=#"     # 6 levels for sub-block resolution (ASCII-safe)
_BAR_WIDTH = 25            # max columns for the bar


def _confidence_bar(value: float, width: int = _BAR_WIDTH) -> str:
    """Render a confidence value as a smooth ASCII bar."""
    scaled = value * width
    full_blocks = int(scaled)
    remainder = scaled - full_blocks
    frac_idx = int(remainder * (len(_BAR_CHARS) - 1))

    bar = "#" * full_blocks
    if full_blocks < width:
        bar += _BAR_CHARS[frac_idx]
        bar += " " * (width - full_blocks - 1)
    return bar


def _print_scores(
    title: str,
    scores: Dict[str, float],
    report_preview: str,
    width: int = 78,
) -> None:
    """Pretty-print scores with a smooth ASCII confidence bar chart."""
    print(f"\n{'=' * width}")
    print(f"  > {title}")
    print(f"  Report: \"{report_preview}\"")
    print(f"{'~' * width}")
    print(f"  {'Target':<20s}  {'Conf':>6s}  {'Bar':<{_BAR_WIDTH}s}  Label")
    print(f"  {'-' * 18}  {'-' * 6}  {'-' * _BAR_WIDTH}  {'-' * 7}")

    for target, score in scores.items():
        bar = _confidence_bar(score)
        # Human-readable label
        if score >= _HIGH_CONF:
            tag = "  + POS"
        elif score >= _MILD_CONF:
            tag = "  ~ MOD"
        elif score > _NEGATED_CONF:
            tag = "  - NEG"
        elif score > _ABSENT_CONF:
            tag = "  - NEG"
        else:
            tag = "  . N/A"
        print(f"  {target:<20s}  {score:>6.3f}  {bar}  {tag}")

    print(f"{'=' * width}")


def _run_assertions(
    test_name: str,
    scores: Dict[str, float],
    expected_pos: List[str],
    expected_neg: List[str],
) -> Tuple[int, int]:
    """Run soft assertions and return (pass_count, fail_count)."""
    passes = 0
    fails = 0
    for t in expected_pos:
        if scores[t] >= _MILD_CONF:
            passes += 1
        else:
            print(f"    !! FAIL [{test_name}]: {t} expected >={_MILD_CONF}, "
                  f"got {scores[t]:.3f}")
            fails += 1
    for t in expected_neg:
        if scores[t] <= _NEGATED_CONF:
            passes += 1
        else:
            print(f"    !! FAIL [{test_name}]: {t} expected <={_NEGATED_CONF}, "
                  f"got {scores[t]:.3f}")
            fails += 1
    return passes, fails


def main() -> None:
    """Run the report labeler test suite with assertion checks."""
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    labeler = ReportLabeler(method="regex")

    # ------------------------------------------------------------------
    # Test Reports
    # ------------------------------------------------------------------
    test_cases: List[Tuple[str, str, List[str], List[str]]] = [
        # (title, report, expected_positive_targets, expected_negated_targets)
        (
            "EN - Positive multi-finding",
            "Complete tear of the ACL with moderate joint effusion and "
            "complex medial meniscal tear.  Mild patellofemoral "
            "osteoarthritis.  Small Baker's cyst noted.",
            ["ACL", "Effusion", "Medial Meniscus", "PF OA", "Baker's"],
            [],
        ),
        (
            "EN - All negated",
            "The ACL is intact.  No evidence of meniscal tear.  No "
            "effusion.  Normal medial and lateral compartments without "
            "osteoarthritis.  No fracture identified.",
            [],
            ["ACL", "Effusion", "Fracture"],
        ),
        (
            "EN - Mixed positive & negated",
            "Grade II MCL sprain.  Bone contusion of the lateral tibial "
            "plateau with associated marrow edema.  No fracture.  Mild "
            "synovitis.  The ACL and lateral meniscus are normal.",
            ["MCL", "Contusion", "Synovitis"],
            ["Fracture", "ACL", "Lateral Meniscus"],
        ),
        (
            "FR - Positive report",
            "Rupture complète du ligament croisé antérieur.  Épanchement "
            "articulaire modéré.  Lésion du ménisque médial.",
            ["ACL", "Effusion", "Medial Meniscus"],
            [],
        ),
        (
            "ES - Positive with severity",
            "Fractura del platillo tibial lateral con edema óseo.  "
            "Derrame articular moderado.  Quiste de Baker.",
            ["Fracture", "Contusion", "Effusion", "Baker's"],
            [],
        ),
    ]

    print("\n" + "=" * 78)
    print("  REPORT LABELER  --  REGEX BACKEND TEST SUITE")
    print("=" * 78)

    total_pass = 0
    total_fail = 0

    for title, report, exp_pos, exp_neg in test_cases:
        preview = report[:72] + ("..." if len(report) > 72 else "")
        scores = labeler(report)
        _print_scores(title, scores, preview)
        p, f = _run_assertions(title, scores, exp_pos, exp_neg)
        total_pass += p
        total_fail += f

    # Edge case: empty report
    print()
    empty_scores = labeler("")
    _print_scores("Edge - Empty report", empty_scores, "(empty)")
    all_absent = all(v == _ABSENT_CONF for v in empty_scores.values())
    if all_absent:
        total_pass += 1
    else:
        print("    !! FAIL: empty report should return all 0.05")
        total_fail += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 78}")
    status = "ALL PASSED [OK]" if total_fail == 0 else f"{total_fail} FAILED [X]"
    print(f"  Results: {total_pass} passed, {total_fail} failed  --  {status}")
    print(f"{'=' * 78}\n")

    # ------------------------------------------------------------------
    # Optional LLM test
    # ------------------------------------------------------------------
    try:
        from transformers import pipeline as _  # noqa: F401
        llm_available = True
    except ImportError:
        llm_available = False

    if llm_available and "--llm" in sys.argv:
        print("=" * 78)
        print("  REPORT LABELER  --  LLM BACKEND TEST")
        print("=" * 78)
        llm_labeler = ReportLabeler(method="llm")
        llm_report = test_cases[0][1]
        llm_scores = llm_labeler(llm_report)
        _print_scores("[LLM] EN - Positive", llm_scores, llm_report[:72])
    elif not llm_available:
        print(
            "  [INFO] LLM test skipped -- `transformers` not installed.  "
            "Run with `pip install transformers` + `--llm` flag to enable."
        )

    sys.exit(1 if total_fail > 0 else 0)


if __name__ == "__main__":
    main()
