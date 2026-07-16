"""Stage 9 — MedGemma clinical report.

Recommendation #3 ("Teach the small AI to reason step-by-step, not just
answer" — High impact): the prompt requires MedGemma to emit its reasoning
chain before the final JSON verdict, not just a bare label, so its
decisions are explainable to clinicians and so any later student-model
distillation (see PROJECT_OVERVIEW.md's distillation plan) can be trained
on the reasoning trace, not just the answer.

Recommendation #10 ("Separate 'what the AI decided' from 'what the safety
rules changed'" — Medium impact, Low effort): `MergedDecision` keeps
`deterministic_decision`, `llm_decision`, and `final_decision` as three
distinct fields rather than collapsing them, so it's always possible to
see how much the LLM actually changed vs. how much came from fixed rules.

Safety (unchanged from baseline): CRITICAL alerts bypass the LLM entirely.
MedGemma may only RAISE an alert level, never lower a CRITICAL produced by
deterministic rules. If MedGemma disagrees with deterministic scoring by
more than one severity level, its output is rejected and the system falls
back to rule-based-only mode. Every decision — including any rejection —
is written to the SHA-256 hash-chained audit log.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import requests

from .audit import AuditLog
from .config import RISK_LEVELS
from .risk import RiskReport

OLLAMA_URL = "http://localhost:11434/api/generate"
MEDGEMMA_MODEL = "medgemma"
CRITICAL_LATENCY_TARGET_MS = 500


PROMPT_TEMPLATE = """You are assisting clinical staff monitoring a post-operative patient's ECG.

## Deterministic ECG risk summary
{ecg_summary}

## Rolling risk trend (last {trend_window} minutes)
{trend_summary}

## Similar past cases (nearest neighbours by waveform embedding)
{similar_cases_summary}

## EHR context
{ehr_context}

## Instructions
Think through this step by step BEFORE giving your final answer:
1. Summarize what the deterministic ECG metrics indicate.
2. Note any disagreement between the metrics and the similar past cases.
3. State your reasoning for a risk level.
4. Only then give the final answer.

You may only RAISE the deterministic risk level below, never lower it:
  deterministic_alert_level = {deterministic_level}

Respond with your step-by-step reasoning, then a final JSON object of the form:
{{"risk_level": "LOW|MEDIUM|HIGH|CRITICAL", "narrative": "...", "disclaimer": "..."}}
"""


@dataclass
class MergedDecision:
    deterministic_decision: dict
    llm_decision: dict | None
    llm_rejected_reason: str | None
    final_decision: dict
    bypassed_llm: bool


def build_prompt(risk_report: RiskReport, trend: dict, similar_cases_summary: str,
                  ehr_context: str = "Not available", trend_window_minutes: float = 15.0) -> str:
    ecg_summary = (
        f"PVC burden: {risk_report.pvc_burden_pct:.1f}% | PAC burden: {risk_report.pac_burden_pct:.1f}% | "
        f"VT runs: {risk_report.vt_run_count} | AFib burden: {risk_report.afib_burden_pct:.1f}% | "
        f"HRV suppressed: {risk_report.hrv_suppressed} | QRS width trend: {risk_report.qrs_width_trend:.2f} | "
        f"NEWS2: {risk_report.news2_score} | qSOFA: {risk_report.qsofa_score}"
    )
    trend_summary = (
        f"Slope: {trend.get('slope_per_minute', 0.0):.3f} risk-levels/min | "
        f"Worsening: {trend.get('worsening', False)} | Snapshots: {trend.get('n_snapshots', 0)}"
    )
    return PROMPT_TEMPLATE.format(
        ecg_summary=ecg_summary, trend_window=trend_window_minutes, trend_summary=trend_summary,
        similar_cases_summary=similar_cases_summary, ehr_context=ehr_context,
        deterministic_level=risk_report.alert_level,
    )


def call_medgemma(prompt: str, model: str = MEDGEMMA_MODEL, timeout_s: float = 10.0) -> dict | None:
    """Calls a locally-deployed MedGemma via Ollama's HTTP API. Returns
    None (not raises) if Ollama isn't reachable, so the pipeline degrades
    to rule-based-only mode instead of crashing — this is itself logged
    to the audit trail by the caller."""
    try:
        resp = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False},
                              timeout=timeout_s)
        resp.raise_for_status()
        text = resp.json().get("response", "")
    except (requests.RequestException, ValueError):
        return None

    try:
        json_start = text.rindex("{")
        parsed = json.loads(text[json_start:])
        parsed["_raw_reasoning"] = text[:json_start].strip()
        return parsed
    except (ValueError, json.JSONDecodeError):
        return None


def merge_decision(risk_report: RiskReport, llm_output: dict | None, audit: AuditLog) -> MergedDecision:
    deterministic = {"risk_level": risk_report.alert_level, "reasons": risk_report.alert_reasons}

    if risk_report.alert_level == "CRITICAL":
        audit.append("MEDGEMMA_BYPASSED", {"reason": "CRITICAL alert bypasses LLM",
                                            "latency_target_ms": CRITICAL_LATENCY_TARGET_MS})
        return MergedDecision(deterministic, None, None, deterministic, bypassed_llm=True)

    if llm_output is None:
        audit.append("MEDGEMMA_UNAVAILABLE", {"fallback": "rule_based_only"})
        return MergedDecision(deterministic, None, "MedGemma unavailable", deterministic, bypassed_llm=False)

    llm_level = llm_output.get("risk_level")
    if llm_level not in RISK_LEVELS:
        audit.append("MEDGEMMA_REJECTED", {"reason": "invalid risk_level in response", "raw": llm_output})
        return MergedDecision(deterministic, llm_output, "invalid risk_level", deterministic, bypassed_llm=False)

    det_idx, llm_idx = RISK_LEVELS.index(risk_report.alert_level), RISK_LEVELS.index(llm_level)

    if abs(llm_idx - det_idx) > 1:
        audit.append("MEDGEMMA_REJECTED", {"reason": "disagreement > 1 severity level",
                                            "deterministic": risk_report.alert_level, "llm": llm_level})
        return MergedDecision(deterministic, llm_output, "disagreement > 1 severity level", deterministic,
                               bypassed_llm=False)

    final_level = RISK_LEVELS[max(det_idx, llm_idx)]  # LLM may only raise, never lower
    final = {"risk_level": final_level, "narrative": llm_output.get("narrative", ""),
              "disclaimer": llm_output.get("disclaimer", "This is not a substitute for clinical judgement."),
              "reasoning": llm_output.get("_raw_reasoning", "")}
    audit.append("MEDGEMMA_ACCEPTED", {"deterministic": risk_report.alert_level, "llm": llm_level,
                                        "final": final_level})
    return MergedDecision(deterministic, llm_output, None, final, bypassed_llm=False)


def generate_report(risk_report: RiskReport, trend: dict, similar_cases_summary: str,
                     audit: AuditLog, ehr_context: str = "Not available") -> MergedDecision:
    if risk_report.alert_level == "CRITICAL":
        return merge_decision(risk_report, None, audit)

    prompt = build_prompt(risk_report, trend, similar_cases_summary, ehr_context)
    audit.append("MEDGEMMA_PROMPT_BUILT", {"prompt_chars": len(prompt)})
    llm_output = call_medgemma(prompt)
    return merge_decision(risk_report, llm_output, audit)
