"""Stage 9 - narrative generation with POSITIVE, STRUCTURAL validation.

Brief 7. The deterministic verdict is decided BEFORE any model is called. The
LLM's only job is to restate that verdict in prose. It never classifies.

A denylist over free-form output is NOT sufficient - an adversarial review
bypassed one 9 times out of 9 using negations ("not regular"), synonyms,
"irregularly irregular", hallucinated rates, and unicode homoglyphs.

Validation is therefore positive and structural, in this order:

  1. Every NUMBER in the prose must appear in the evidence JSON.
  2. The VERDICT WORD in the prose must match the verdict in the JSON.
  3. NFKC-normalise before checking, so homoglyphs collapse.
  4. Content denylist LAST, as a backstop only.

Any single failure discards the ENTIRE narrative - no patching, no partial
acceptance - and the deterministic template is used instead. The template is
always available, so the system never depends on the model being reachable or
well-behaved.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

from .verdict import REGULAR, IRREGULAR, UNABLE
from . import SCOPE_STATEMENT

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "medgemma"
NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")

# Spelled-out numerals evade digit-based grounding entirely ("zero point three
# five"). Found by expanded adversarial test.
NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
                "eight", "nine", "ten", "eleven", "twelve", "twenty", "thirty",
                "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
                "hundred", "thousand", "point")

# Hedging changes the clinical meaning while passing every word-presence check
# ("possibly regular", "borderline"). The verdict is deterministic; prose that
# softens it is misreporting it.
HEDGE_WORDS = ("possibly", "probably", "perhaps", "maybe", "might", "may be",
               "appears", "seems", "likely", "unlikely", "borderline",
               "suggestive", "suspicious", "roughly", "approximately", "about")

# First-person model voice implies the model formed a judgement. It did not.
MODEL_VOICE = ("i think", "i believe", "i would", "in my", "my assessment",
               "i suspect", "i conclude")

VERDICT_WORDS = {
    REGULAR: ("regular",),
    IRREGULAR: ("irregular",),
    UNABLE: ("unable to determine", "unable_to_determine", "cannot be determined"),
}

# Backstop only. The positive checks above do the real work.
DENYLIST = (
    "atrial fibrillation", "afib", "a-fib", "a fib", "flutter",
    "diagnos", "you have", "patient has", "treatment", "medication",
    "prescri", "benign", "normal sinus", "arrhythmia", "tachycardia",
    "bradycardia", "infarct", "ischemi", "ischaemi",
    # anatomical euphemisms: naming the mechanism without naming the condition
    "upper chamber", "lower chamber", "atri", "ventric", "quiver", "fibrillat",
    "healthy heart", "heart is healthy", "no cause for concern", "reassuring",
)


@dataclass(frozen=True)
class NarrativeResult:
    text: str
    source: str          # "medgemma" | "template"
    accepted: bool       # was a model narrative accepted?
    failures: tuple      # why a model narrative was rejected
    prompt: str | None = None


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()


def _evidence_values(evidence: dict) -> list[float]:
    out = []
    for v in evidence.values():
        if v is None or isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and v == v:
            out.append(float(v))
    return out


def _grounded(tok: str, values: list[float]) -> bool:
    """Is this prose token grounded in the evidence?

    The token is matched AT ITS OWN PRECISION: a token written to 4 decimals
    must equal an evidence value rounded to 4 decimals. Rounding both sides
    down to fewer digits would let "0.0350" match an evidence 0 (from, say,
    n_flagged=0), which is a false accept - found by adversarial test.
    """
    t = tok.replace(",", ".").lstrip("+")
    try:
        f = float(t)
    except ValueError:
        return False
    dp = len(t.split(".")[1]) if "." in t else 0
    return any(round(v, dp) == round(f, dp) for v in values)


def validate(text: str, verdict: str, evidence: dict) -> tuple[bool, tuple]:
    """Positive structural validation. Returns (ok, failures)."""
    failures = []
    norm = _nfkc(text)
    # The mandated scope statement (brief 8) itself contains "atrial
    # fibrillation" and "diagnosis". Checking it against the denylist rejects
    # the deterministic template - i.e. the fallback fails its own validator.
    # Found by adversarial test. It is fixed, mandated text, so it is removed
    # before content checks and validated by exact match instead.
    scope_norm = _nfkc(SCOPE_STATEMENT)
    body = norm.replace(scope_norm, " ")

    # 0. NFKC does NOT fold Cyrillic/Greek homoglyphs onto Latin, so a
    #    homoglyph "irregular" survives normalisation and evades the verdict
    #    check. Found by adversarial test. Require plain ASCII letters.
    for ch in norm:
        if ch.isalpha() and ord(ch) > 127:
            failures.append(f"non-ASCII letter {ch!r} (U+{ord(ch):04X}) - possible homoglyph")
            break

    # 1. every number in the prose must be grounded in the evidence
    values = _evidence_values(evidence)
    for tok in NUM_RE.findall(body):
        if not _grounded(tok, values):
            failures.append(f"ungrounded number {tok!r} not present in evidence JSON")

    # 2. the verdict word must be present, be the RIGHT one, and not be negated
    want = VERDICT_WORDS[verdict]
    hits = [m for w in want for m in re.finditer(
        rf"(?<![a-z]){re.escape(w)}(?![a-z])", body)]
    if not hits:
        failures.append(f"verdict word for {verdict} absent from narrative")
    for other, words in VERDICT_WORDS.items():
        if other == verdict:
            continue
        for w in words:
            if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", body):
                failures.append(f"narrative asserts {other} but verdict is {verdict}")

    # 2b. a negation before the verdict word inverts it while passing every
    #     word-presence check ("not regular"). Found by adversarial test.
    for m in hits:
        window = body[max(0, m.start() - 40):m.start()]
        if re.search(r"(?<![a-z])(not|no|never|without|n't|cannot|isn|aren|fails? to)"
                     r"(?![a-z])", window):
            failures.append(f"verdict word for {verdict} appears negated: "
                            f"...{norm[max(0,m.start()-40):m.end()]!r}")
            break

    # 3. every SENTENCE must be anchored to the evidence.
    #    Without this the validator is permissive by default for any sentence
    #    containing no digits and no denylisted word, which let through
    #    "consistent with a healthy heart" and "no quivering of the upper
    #    chambers" - an aetiology claim with the banned word removed.
    #    Found by expanded adversarial test. Anchoring is structural, so it
    #    does not depend on enumerating the things a model might say.
    for sent in re.split(r"(?<=[.!?])\s+", body):
        st = sent.strip()
        if not st or st in scope_norm or st in ("basis:",):
            continue
        has_num = any(_grounded(t, values) for t in NUM_RE.findall(st))
        has_verdict = any(re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", st)
                          for w in VERDICT_WORDS[verdict])
        if not (has_num or has_verdict):
            failures.append(f"unanchored sentence (no grounded number, no verdict "
                            f"word): {st[:60]!r}")

    # 3b. spelled-out numerals bypass digit grounding
    for w in NUMBER_WORDS:
        if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", body):
            failures.append(f"spelled-out numeral {w!r} - numbers must be digits "
                            f"so they can be grounded")
            break

    # 3c. hedging and model voice
    for w in HEDGE_WORDS:
        if re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", body):
            failures.append(f"hedging term {w!r} softens a deterministic verdict")
            break
    for w in MODEL_VOICE:
        if w in body:
            failures.append(f"model voice {w!r} - the model did not decide this")
            break

    # 4. denylist backstop
    for bad in DENYLIST:
        if bad in body:
            failures.append(f"denylisted term {bad!r}")

    return (not failures), tuple(failures)


def template(verdict: str, reason: str, evidence: dict) -> str:
    """Deterministic narrative. ALWAYS available; never fails."""
    cv = evidence.get("rr_cv")
    hr = evidence.get("hr_bpm")
    thr = evidence.get("threshold")
    parts = []
    if verdict == UNABLE:
        parts.append("This window is reported as UNABLE TO DETERMINE.")
    else:
        parts.append(f"This window is reported as {verdict}.")
    if cv is not None and cv == cv:
        parts.append(f"The RR coefficient of variation is {cv:.4f}, "
                     f"against a threshold of {thr:.4f}.")
    if hr is not None and hr == hr:
        parts.append(f"The mean heart rate over the window is {hr:.1f} beats per minute.")
    parts.append(f"Basis: {reason}")
    parts.append(SCOPE_STATEMENT)
    return " ".join(parts)


def build_prompt(verdict: str, reason: str, evidence: dict) -> str:
    ev = {k: v for k, v in evidence.items() if v is not None and v == v}
    return (
        "Restate the following ECG rhythm result in two or three plain sentences "
        "for a clinician.\n\n"
        "STRICT RULES:\n"
        f"- The verdict is {verdict}. State it exactly. Do not change, hedge or reinterpret it.\n"
        "- Use ONLY numbers that appear in the JSON below. Invent no other numbers.\n"
        "- Do not name any cardiac condition. Do not diagnose.\n"
        "- Do not mention atrial fibrillation or any other aetiology.\n\n"
        f"VERDICT: {verdict}\nBASIS: {reason}\n"
        f"EVIDENCE JSON:\n{json.dumps(ev, indent=2, default=str)}\n"
    )


def generate(verdict: str, reason: str, evidence: dict, *,
             use_model: bool = True, timeout_s: float = 30.0) -> NarrativeResult:
    """Try MedGemma; validate structurally; fall back to the template on ANY failure."""
    prompt = build_prompt(verdict, reason, evidence)
    if not use_model:
        return NarrativeResult(template(verdict, reason, evidence), "template",
                               False, ("model not requested",), prompt)
    if verdict == UNABLE:
        # Measured empirically (2026-09, live device data): every MedGemma
        # narrative attempted on an UNABLE_TO_DETERMINE window was rejected by
        # validate() below -- the model tends to assert more than the evidence
        # supports when the deterministic engine itself couldn't decide. That
        # made every one of these windows pay the full ~9.8s Ollama call for a
        # result that was always going to be discarded. Not a hard guarantee
        # (a compliant "cannot be determined" reply is possible in principle),
        # but skipping the call here trades a small, unproven chance of an
        # accepted model narrative for a real, consistent latency win on the
        # verdict this happens to fire most often.
        return NarrativeResult(template(verdict, reason, evidence), "template",
                               False, ("verdict is UNABLE_TO_DETERMINE -- model call skipped",), prompt)
    try:
        import urllib.request
        req = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps({"model": OLLAMA_MODEL, "prompt": prompt,
                             "stream": False,
                             "options": {"temperature": 0.1, "num_predict": 200}}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            text = json.loads(r.read()).get("response", "").strip()
    except Exception as exc:
        return NarrativeResult(template(verdict, reason, evidence), "template",
                               False, (f"model unreachable: {type(exc).__name__}",), prompt)

    ok, failures = validate(text, verdict, evidence)
    if not ok:
        # entire narrative discarded - no patching, no partial acceptance
        return NarrativeResult(template(verdict, reason, evidence), "template",
                               False, failures, prompt)
    return NarrativeResult(text, "medgemma", True, (), prompt)
