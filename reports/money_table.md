# Money table — invoice extraction (local open teacher → distilled student)

_Teacher `qwen3:14b` and student `Qwen/Qwen2.5-0.5B-Instruct` both run locally on the RTX 5080; gold set, field_f1._

> **Silver-grade numbers:** scored on the 37 gold items where an INDEPENDENT second model (qwen2.5-coder:32b) agreed with the teacher's label (cross-model agreement removes the teacher-vs-itself circularity). This is stronger than dev-grade but not yet human-verified.


| Axis | Teacher (qwen3:14b) | Student (distilled) | Win |
|---|---|---|---|
| Model footprint | 14B-class | Qwen2.5-0.5B-Instruct (~0.5B) | **~28× smaller** |
| Quality (field_f1) | 1.0000 (ref) | 0.9647 (96.5% of teacher) | meets 95% bar |
| Schema-valid rate | — | 100.0% | robustness |
| Exact match | — | 54.1% | strictest view |
| p95 latency | full 14B forward | 5,921 ms | smaller footprint |
| Data egress | stays local | stays local | on-prem / private |
| $/1k (GPU amortized) | $0.0000 | $0.1178 | both ~free locally |

**Honest note on cost:** with a *free local* teacher, the "1/40th the cost of a frontier API" pitch does **not** apply — both models run on your own GPU, so the student is not cheaper than the teacher in dollars. The real value here is **footprint** (a ~0.5B student packs alongside other workloads and serves at 0.23 req/s) and **privacy** (nothing leaves the box). To make the dollar-cost case, distill from a **paid** frontier teacher: then student $0.1178/1k vs the API list price is the headline (set `teacher.provider` to a paid client in configs/default.yaml).

*Assumptions: GPU $1,100 amortized over 3 years, 360 W at $0.15/kWh, measured throughput 0.23 req/s.*
