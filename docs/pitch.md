# RiskSentinel — Hackathon Pitch Draft

## One-liner
RiskSentinel is an agentic systemic-risk simulator that turns "what-if" shock questions into deterministic contagion analytics, GPT-backed strategy, and judge-ready explainability KPIs.

## Problem
Portfolio teams can ask strategic risk questions quickly, but answers are often:
- not grounded in deterministic evidence,
- hard to validate,
- weak in explainability/auditability.

## Solution
RiskSentinel combines network science and multi-agent orchestration:
- deterministic shock engine (NetworkX + contagion models),
- control-plane workflow (`Planner -> Architect+Quant -> Advisor -> Critic`),
- one-click full demo flow for judges (Build + Commander + Autonomous + Co-Pilot),
- evidence-first outputs with citations and critic gate,
- judge dashboard metrics for reliability and latency.

## Why It Is Novel
- Hard separation of control plane and LLM reasoning.
- Immutable evidence ledger (`E1..`) + critic hard gate (max 1 revision).
- Evidence-RAG from historical crises + prior runs.
- Built-in benchmark and scenario-pack evaluation for demo reliability.

## Microsoft Stack
- Microsoft Agent Framework (`agent-framework`)
- Azure OpenAI (GPT-4o / GPT-4o-mini)
- Azure-ready configuration for deployment and judge access gating

## Live Demo Flow (2 min)
1. Ask: "What if JPM crashes 40% on 2025-12-01?"
2. Show deterministic cascade (nodes affected, waves, stress).
3. Switch to complex compare query (JPM vs GS) and show GPT control-plane output.
4. Open Explainability panel: evidence injected, critic status, route trace.
5. Show Planner/Executor/Critic badges + Judge Dashboard KPIs.
6. Export submission bundle (`report + trace + KPI + scenario eval`).

## Impact
- Faster risk triage and clearer mitigation actions.
- Stronger trust via explicit evidence and validation gates.
- Demo-safe reliability with graceful local fallback.

## Current Status
- Core app working on Streamlit.
- Chainlit chat app available (`apps/chainlit/app.py`).
- Tests passing: 61.

## Next Deliverables
- Final video recording.
- Final README polish and screenshots.
- Submission form package.
