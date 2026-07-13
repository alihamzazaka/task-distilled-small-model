# 01 — Overview (v2.0)

> Why v1.0 needs a Phase 2, what the four v2.0 features are, the honest state we are anchoring on, the goals and non-goals, and the single headline success criterion the phase is judged on. Everything here is a **plan**; the only measured numbers are the v1.0 baseline.

## Where we are — the v1.0 baseline (this is built)

v1.0 is a *complete, tested pipeline* that already produced a working distilled model. The measured facts, from [`reports/`](../../reports/) and [`configs/default.yaml`](../../configs/default.yaml):

- **Task (locked, Phase 0):** invoice/receipt text → canonical `Invoice` JSON (`vendor`, `date`, `currency`, `line_items[]`, `subtotal`, `tax`, `grand_total`, `payment_terms`); metric **M = field-F1** on the gold set; bar = **≥ 95%** of teacher. See [`src/distil_task/schema.py`](../../src/distil_task/schema.py).
- **Teacher (as run):** `qwen3:14b`, **local via Ollama**, `think:false`, temp 0.0. Free — so it bills **$0/1k**.
- **Student (as run):** `Qwen/Qwen2.5-0.5B-Instruct`, **full fine-tune**, 568 train / 78 dev, 3 epochs, ~160 s, final train loss 0.551.
- **Data:** 800 seeds labeled; 11 dropped on the consistency check (0 schema fails, 0 near-dupes); splits train 568 / dev 78 / test 53 / gold 60 / agreement-pool 30; **0 cross-split leaks**.
- **Gold set:** 60 items, **SILVER grade** — verified by *cross-model agreement* (`qwen3:14b` teacher vs an independent `qwen2.5-coder:32b`, field-F1 ≥ 0.95). **37** items passed and were scored; **0** are human-verified.
- **Measured quality (silver, n = 37):** field-F1 **0.9647** (**96.5%** of teacher), micro-F1 0.9676, precision 0.9619, recall 0.9685, **exact-match 0.5405 (54.1%)**, **schema-valid 100%**, teacher-agreement **93.5%**.
- **Latency / throughput (0.5B on RTX 5080):** p50 **4.04 s**, p95 **5.92 s**, **0.226 req/s**.
- **Cost:** teacher **$0/1k** (free local), student **$0.1178/1k** amortized; `cost_multiple = 0`, break-even **N/A**.
- **Tests:** **118** pure-Python tests green (schema / metrics / filtering / cost-model).

**The one-line honest verdict:** the *pipeline* works and the *quality* transfer is real, but the *thesis* — cheaper than a frontier API — is **unproven**, because both teacher and student ran free on the same box. See the note at the bottom of [`reports/money_table.md`](../../reports/money_table.md), which says so explicitly.

## Problem → solution (what Phase 2 fixes)

| | |
|---|---|
| **Problem** | v1.0 proved *quality parity* but not the *cost win*. The teacher was free/local (no dollar arbitrage), the gold set is silver (cross-model, not human — a residual circularity the SPEC forbids for the headline number), the student is a 0.5B fallback (below the SPEC's 1–3B band, exact-match only 54%), and there is no valid GGUF, so the "one-command local run" deliverable is not actually runnable. |
| **Solution** | Four targeted, dependency-ordered features that convert each unproven claim into a measured one **without touching the locked task, schema, or metric**: (F1) distill/price from a paid frontier teacher; (F2) human-verify + expand the gold set; (F3) scale the student into 1–3B with an ablation; (F4) ship a valid GGUF + real card + demo. |
| **Why now** | The plumbing for the hardest part — real cost accounting — already exists and is tested. The remaining work is *runs and verification*, not new infrastructure. This is the cheapest possible path from "proof of pipeline" to "proof of thesis." |

## The v2.0 vision

> **The same 0.5B-to-3B student that already reaches 96.5% of a 14B teacher, re-grounded so the whole story is defensible: distilled from a *paid* frontier teacher and shown to be `N×` cheaper per 1k requests with a real break-even curve; parity measured against a *human-verified* gold set; a *1–3B* student with a documented capacity ablation that quantifies the last few points of the gap; and a *valid GGUF* that `ollama run` loads to extract a real invoice on a laptop.**

This is deliberately *not* a bigger or fancier system. It is the v1.0 system with its four soft spots hardened into evidence.

## Motivation, feature by feature

### F1 — Paid frontier teacher (the thesis)
The project's headline is a **cost** claim. v1.0's money table has `cost_multiple = 0` because a local teacher costs $0/1k, so there is nothing to be cheaper *than*. The fix is to distill from (or at minimum price against) a real paid API teacher — the config already carries a `price_table` with `claude-sonnet-4-5` ($3 / $15 per Mtok), `claude-opus-4-6` ($15 / $75), and `claude-haiku-4-5` ($1 / $5), and `AnthropicTeacher` is already implemented. Then student **$0.1178/1k** vs the API's list price becomes the headline `N×`, and `cost_model.break_even_requests_per_day` yields the break-even volume the SPEC promises.

### F2 — Human-verified gold (the honesty)
v1.0's parity number is measured on a **silver** set: labels the teacher produced, kept when an independent model agreed. That is stronger than dev-grade but still a *model-vs-model* number — the exact cross-model circularity the SPEC says the headline must not have. Hand-verifying (and expanding) the gold set, with an adjudication protocol and a reported inter-annotator agreement, makes "96.5% of teacher" a claim against **human ground truth**.

### F3 — Scale the student (the spec band)
v1.0 shipped a **0.5B** student because "the 3B download was unreliable on this link" (config comment) — a real, mundane blocker, not a design choice. It is *below* the SPEC's stated 1–3B band, and its **exact-match is only 54%**. F3 brings an offline 1–3B base copy onto the GPU box and runs a controlled **capacity × data** ablation to answer: *how much of the residual field-F1 gap and the low exact-match is a capacity problem the SPEC band would fix, versus a data problem?*

### F4 — One-command local run (the deliverable)
The Phase 1 deliverables include `ollama run your-model` + a demo video + a blog. Today `export_ollama.py` *prints* the GGUF commands and writes a Modelfile, but a **valid GGUF has not been produced** — the note in the script records that Ollama's built-in converter yields a GGUF that crashes llama.cpp's sampler (`Assertion 'found' ... llama-sampling.cpp:660`) for this fine-tuned Qwen2.5 base, so the current path is a `FROM .` safetensors import. F4 produces and verifies a real GGUF via `convert_hf_to_gguf.py`, replaces the **auto-generated TRL stub** model card (which still shows a "time-machine" example, not invoices), and ships the demo + blog.

## Goals

- **G1** Fill the money table with a **real dollar cost win** and a published break-even volume against a paid frontier teacher.
- **G2** Re-measure the ≥95% parity bar on a **human-verified** gold set with a reported inter-annotator agreement.
- **G3** Train a student in the **1–3B** band and publish a **capacity-vs-data ablation** quantifying the gap that size closes.
- **G4** Produce a **verified GGUF** so `ollama run distil-invoice` extracts a real invoice, plus a real model card, demo video, and blog.
- **G5** Keep the honesty contract: every v2.0 number is measured, labeled, and its residual gap analyzed — no overclaiming.

## Non-goals

- **Changing the task, schema, or metric.** invoice→JSON, the `Invoice` contract, and field-F1 are locked from Phase 0 and stay locked.
- **A new distillation algorithm.** Still synthetic-data / sequence-level KD (API teachers have no logits). See [../02-architecture.md](../02-architecture.md).
- **Beating the teacher.** The goal remains *parity within tolerance at lower cost*, not superiority.
- **Multi-GPU / datacenter training or a production MLOps platform.** Everything still targets the single RTX 5080 (16 GB).
- **Retiring the local-teacher path.** The local teacher stays supported (privacy / offline story); F1 adds the paid path, it does not remove the free one.
- **13B+ students as the headline.** The SPEC band is 1–3B; 13B remains a tight QLoRA edge case only.

## Headline success criteria

v2.0 is "done and successful" when **all** hold (each is a *target* until measured):

1. **Dollar cost win quantified (F1).** The money table shows teacher API $/1k vs student **$0.1178/1k**, a headline `N×` multiple, and a break-even requests/day, computed with the honest amortized student model. *(Illustrative: with `claude-sonnet-4-5` at ~900 in / 350 out tokens/request ≈ $7.95/1k, the multiple is on the order of **60–70×**; the real number is whatever we measure.)*
2. **Parity on human gold (F2).** Student field-F1 ≥ 95% of teacher on the **human-verified** gold set; inter-annotator agreement reported; the silver→gold circularity removed.
3. **Scaled student + ablation (F3).** A 1–3B student is trained on the 5080, and a capacity×data table quantifies its field-F1 and exact-match gain over the 0.5B baseline.
4. **Runnable local model (F4).** A valid GGUF loads in Ollama; `ollama run distil-invoice "<invoice text>"` returns schema-valid JSON; the model card documents it; a demo video and blog are published.
5. **Failure analysis refreshed (F5, cross-cutting).** The residual gap on the human gold set is re-categorized, stating which cases still need the teacher — especially the exact-match misses.

> Illustrative figures (`N×`, 60–70×, token profiles) are **placeholders** carried over from the SPEC's convention. The contract, exactly as in Phase 1, is: **commit the bar, then measure honestly against it.**

## Related docs

- Architectural additions that deliver these goals: [02-architecture.md](02-architecture.md)
- The requirements each feature must satisfy: [03-requirements.md](03-requirements.md)
- How each goal is measured (and the money table): [05-evaluation-metrics.md](05-evaluation-metrics.md)
- The ordered build plan: [07-build-roadmap.md](07-build-roadmap.md)
- The v1.0 success criteria this extends: [../01-overview.md](../01-overview.md)
