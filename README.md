# Proof-Carrying Recursive Task Runtime

A research artifact: a lightweight recursive LLM runtime that produces an auditable **proof tree** rather than a bare answer. Tasks are decomposed into typed contracts with verifiers, slot dataflow, and a repair loop; each accepted result carries the evidence that justified it.

The runtime is domain-neutral. Two plugins are included: a `code_tests` plugin (pytest verifier) and a `knowledge_qa` plugin (Wikipedia-backed multi-hop QA).

The full design rationale, experimental method, characterization results, and lessons are in **[Task_Runtime_Whitepaper.md](./Task_Runtime_Whitepaper.md)**.

## Status

Research artifact, not a tool. On the included mock-trial multi-hop benchmark, v1.2 reaches 5/10 cached-source A0 success — below the 7/10 threshold for a credible tool prototype. The artifact's central contribution is methodological: it makes LLM-agent failures **inspectable**, **localizable**, and **experimentally reducible**. See whitepaper §6 and §9.

## Requirements

- Python 3.10+
- `pytest` for the test suite
- `openai` for live LLM calls. Copy `.env.example` to `.env` and fill in your key (the example demos auto-load it).

Wikipedia access is provided by the bundled `task_runtime/wiki_tool.py` (stdlib-only) backed by the bundled `task_runtime/wiki_cache/` directory. No external setup is required for the cached-source reproductions below. To bypass the cache and hit live Wikipedia, set the `WIKI_CACHE_DIR` environment variable to an empty directory.

## Run the tests

```bash
# from the repo root
python -m pytest -q
```

Expected: 92 passed. (Test paths are scoped via `pytest.ini`.)

## Reproduce the frozen v1.2 results

```bash
# A0 baseline x 10 (cached sources)
python -m task_runtime.examples.mock_trial_multihop_demo \
    --runs 10 --modes oracle_chain_executor

# Founder-slot diagnostic (F0, F0_inverse, F1, F2)
python -m task_runtime.examples.founder_slot_diagnostic \
    --modes F0,F0_inverse,F1,F2

# Ablation matrix
python -m task_runtime.examples.mock_trial_ablation_harness \
    --runs 3 --ablations A1,A2,A4,A5,A6,A8
```

Frozen outputs and reproduction notes are in `task_runtime/examples/sample_outputs/mock_trial_v1.2_research_artifact/`.

## Layout

```
.env.example                template for the .env that the demos auto-load
.gitignore
pytest.ini                  scopes pytest to task_runtime/tests
Task_Runtime_Whitepaper.md  full design + experiments + lessons
README.md                   this file

task_runtime/               the package
  __init__.py
  core.py                   TaskContract, Verdict, ProofNode, plugin protocol
  runtime.py                run_task, spawn_subtasks, slot threading, repair loop
  wiki_tool.py              bundled cached Wikipedia client (stdlib only)
  wiki_cache/               pre-populated cache for the mock-trial sources

  plugins/
    code_tests.py             pytest verifier (universality demonstration)
    knowledge_qa.py           graph, mention resolution, judges, executors, schemas

  examples/
    slugify_demo.py             code-tests universality demo
    repair_loop_demo.py         verifier-driven repair demo
    atomic_qa_demo.py           single-fact QA
    mock_trial_multihop_demo.py     full chain benchmark
    founder_slot_diagnostic.py      F0/F0_inverse/F1/F2 modes
    mock_trial_ablation_harness.py  A1..A8 ablations
    sample_outputs/
      mock_trial_v1.2_research_artifact/  frozen final-state artifact

  tests/                    92 tests
```
