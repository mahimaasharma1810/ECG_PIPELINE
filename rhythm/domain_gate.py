"""Device-domain gate: what the system is allowed to emit, and when.

Clinical severity results (and the MedGemma report they would trigger) must
NOT be produced until R-peak timing has been validated on the device domain
against a controlled reference recording.

Why this is a hard gate and not a warning: device detection is currently known
to manufacture irregularity (~4.4% spurious peaks -> RR CV ~0.15 from a steady
rhythm). A severity label computed on that signal would be wrong, and a
MedGemma report restating it would give the wrong answer a clinical voice.

Three states:

  UNVALIDATED  no reference recording has been processed. Rhythm verdicts may
               be computed and shown as ENGINEERING output with evidence, but
               NO clinical severity and NO MedGemma report.
  FAILED       a reference recording was processed and detection did not meet
               criteria. Same restrictions, plus the failure reasons.
  VALIDATED    criteria met. Clinical severity and report generation permitted,
               scoped to the conditions the validation actually covered.

The gate reads a validation artifact written by the reference harness, so the
permission is auditable and cannot be granted by editing a flag.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

UNVALIDATED = "UNVALIDATED"
FAILED = "FAILED"
VALIDATED = "VALIDATED"

ARTIFACT = Path("reports_rhythm/device_domain_validation.json")


@dataclass(frozen=True)
class GateDecision:
    state: str
    may_emit_rhythm_verdict: bool
    may_emit_clinical_severity: bool
    may_call_medgemma: bool
    reason: str
    evidence: dict


def read_gate(artifact: Path = ARTIFACT) -> GateDecision:
    if not artifact.exists():
        return GateDecision(
            UNVALIDATED, True, False, False,
            "No device-domain validation artifact exists. Device R-peak timing "
            "has never been checked against a beat-level reference, so no "
            "clinical severity may be emitted and MedGemma must not be called. "
            "Rhythm verdicts remain available as engineering output only.",
            {})
    d = json.loads(artifact.read_text())
    v = d.get("validation", {})
    if not v.get("passed"):
        return GateDecision(
            FAILED, True, False, False,
            "Device-domain validation FAILED: "
            + "; ".join(v.get("failures", ["unspecified"])),
            d)
    return GateDecision(
        VALIDATED, True, True, True,
        f"Device-domain validation passed on {d.get('recording', 'unknown recording')}.",
        d)


def require_clinical(gate: GateDecision) -> None:
    """Raise unless clinical output is permitted. Fail closed."""
    if not gate.may_emit_clinical_severity:
        raise PermissionError(
            f"[{gate.state}] clinical severity is not permitted. {gate.reason}")


def write_validation(recording: str, validation: dict,
                     artifact: Path = ARTIFACT, **meta) -> Path:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(
        dict(recording=recording, validation=validation, **meta),
        indent=2, default=str))
    return artifact
