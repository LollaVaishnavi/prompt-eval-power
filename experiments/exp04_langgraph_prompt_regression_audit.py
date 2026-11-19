"""Experiment 04: End-to-End Offline RAG / LangGraph Prompt Regression Audit.

Answers: On a 300-query stratified RAG corpus, how does a realistic concise
prompt rewrite (v2_concise_rewrite) behave under a small n=30 suite vs the full
n=300 suite, and which query buckets absorb the discordant pass-to-fail flips?
"""

from __future__ import annotations

import json
from pathlib import Path

from prompt_eval_power.config import CorpusConfig, GatePolicy
from prompt_eval_power.corpus import generate_rag_corpus
from prompt_eval_power.pipeline import RagStateGraph
from prompt_eval_power.power import evaluate_gate

OUT_PATH = Path("docs/experiments/exp04_langgraph_prompt_regression_audit.json")


def run() -> dict:
    cfg = CorpusConfig()
    policy = GatePolicy()
    corpus = generate_rag_corpus(cfg)
    graph = RagStateGraph(corpus)

    base_300 = graph.evaluate_prompt("v1_baseline_grounded", limit=300)
    v2_300 = graph.evaluate_prompt("v2_concise_rewrite", limit=300)
    v3_300 = graph.evaluate_prompt("v3_strict_citation", limit=300)

    dec_v2_300 = evaluate_gate(
        [o.passed for o in base_300],
        [o.passed for o in v2_300],
        buckets=[o.bucket for o in base_300],
        policy=policy,
    )
    dec_v2_30 = evaluate_gate(
        [o.passed for o in base_300[:30]],
        [o.passed for o in v2_300[:30]],
        buckets=[o.bucket for o in base_300[:30]],
        policy=policy,
    )
    dec_v3_300 = evaluate_gate(
        [o.passed for o in base_300],
        [o.passed for o in v3_300],
        buckets=[o.bucket for o in base_300],
        policy=policy,
    )

    payload = {
        "experiment": "exp04_langgraph_prompt_regression_audit",
        "corpus_queries": cfg.query_count,
        "v2_verdict": dec_v2_300.verdict,
        "v2_exit_code": dec_v2_300.exit_code,
        "v2_n30_verdict": dec_v2_30.verdict,
        "v2_n30_exit_code": dec_v2_30.exit_code,
        "v2_n30_deficit": dec_v2_30.sample_size_deficit,
        "v2_baseline_pass_pct": round(100.0 * dec_v2_300.overall.baseline_pass_rate, 1),
        "v2_candidate_pass_pct": round(100.0 * dec_v2_300.overall.candidate_pass_rate, 1),
        "v2_net_delta_pp": round(100.0 * dec_v2_300.overall.net_delta, 1),
        "v2_pass_to_fail_flips": dec_v2_300.overall.n_pass_to_fail,
        "v2_fail_to_pass_flips": dec_v2_300.overall.n_fail_to_pass,
        "v2_p_value": round(dec_v2_300.overall.p_value_regression, 6),
        "v3_verdict": dec_v3_300.verdict,
        "v3_fail_to_pass_flips": dec_v3_300.overall.n_fail_to_pass,
        "v2_buckets": [
            {
                "bucket": b.bucket,
                "n_queries": b.n_queries,
                "baseline_pass_pct": round(100.0 * b.baseline_pass_rate, 1),
                "candidate_pass_pct": round(100.0 * b.candidate_pass_rate, 1),
                "net_delta_pp": round(100.0 * b.net_delta, 1),
                "pass_to_fail": b.pass_to_fail,
                "fail_to_pass": b.fail_to_pass,
                "p_value": round(b.p_value_regression, 5),
                "breached": b.breached,
            }
            for b in dec_v2_300.bucket_results
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    data = run()
    print("=== Experiment 04: Offline RAG / LangGraph Prompt Audit (v1 vs v2_concise_rewrite) ===")
    print(
        f"{'Bucket':<18} {'Base %':>7} {'Cand %':>7} {'Delta':>8} "
        f"{'P->F':>6} {'F->P':>6} {'p-val':>9} {'Breach':>7}"
    )
    for b in data["v2_buckets"]:
        print(
            f"{b['bucket']:<18} {b['baseline_pass_pct']:>6.1f}% {b['candidate_pass_pct']:>6.1f}% "
            f"{b['net_delta_pp']:>+7.1f} {b['pass_to_fail']:>6} {b['fail_to_pass']:>6} "
            f"{b['p_value']:>9.5f} {str(b['breached']):>7}"
        )
    print(
        f"n=300: {data['v2_pass_to_fail_flips']} pass-to-fail flips -> {data['v2_verdict']} "
        f"(exit {data['v2_exit_code']}); at n=30 -> {data['v2_n30_verdict']} "
        f"(exit {data['v2_n30_exit_code']}, deficit={data['v2_n30_deficit']} queries)."
    )
