# KGIRS: Tool-using agent on MuSiQue — single tool vs routed

An LLM agent answers 100 multi-hop questions from MuSiQue (answerable dev split) using three tools:
**search**, the **Wikidata** knowledge graph, and a **calculator**. We compare an agent restricted to one tool
with agents that route between all tools.

## Conditions

| Condition | Tools | How the tool is chosen |
|---|---|---|
| `closed_book` | none | baseline, LLM memory only |
| `search_only` | search | single tool |
| `wikidata_only` | Wikidata | single tool |
| `calculator_only` | calculator | single tool (sanity check: MuSiQue is rarely arithmetic) |
| `routed_react` | all three | implicit routing: the ReAct agent picks a tool every step |
| `routed_router` | all three | explicit routing: a router LLM call picks the tool for the next sub-question, an executor call (seeing only that tool) writes the tool input |

All conditions share the same LLM, temperature 0, max 8 steps, and the same 100 questions.

## Tools

- **search** (`tools/search.py`): by default BM25 over the 20 Wikipedia paragraphs that ship with each MuSiQue
  question (the standard distractor setting; reproducible). `--search-backend wikipedia` uses live Wikipedia search instead.
- **wikidata** (`tools/wikidata.py`): `wikidata[entity | property]` resolves the entity with `wbsearchentities`,
  fetches all its statements via SPARQL, and filters by property label (with synonyms such as wife→spouse, born→date of birth).
  Returns alternative matches with QIDs so the agent can disambiguate.
- **calculator** (`tools/calculator.py`): AST-based safe arithmetic (no `eval`).

All tool and LLM calls are cached in `cache/` (SQLite), so runs are resumable and re-running is free.

## Data

`prepare_data.py` draws 100 questions from `musique_ans_v1.0_dev.jsonl` with seed 42, stratified by hop count
(52 two-hop, 31 three-hop, 17 four-hop) → `data/musique_100.jsonl`.

## Metrics

Official MuSiQue answer metrics: Exact Match and token F1 (SQuAD normalization, max over gold answer + aliases).
Also: finish rate, steps, tool calls per tool, share of questions where >1 tool was used, LLM calls, tokens, latency,
per-hop breakdown, and **complementarity**: the oracle upper bound if a perfect router chose the best single tool per question,
and how often each routed agent beats or loses to the single-tool agents.

## Running

```powershell
pip install -r requirements.txt
copy .env.example .env          # then put your API key in .env
python prepare_data.py          # already done; re-run only to change the sample
python smoke_test_tools.py      # check the tools (no LLM needed)
python -m tests.test_pipeline   # offline agent test with a scripted LLM

python run_experiments.py --limit 3 --conditions routed_react   # quick trial
python run_experiments.py                                       # full experiment (6 conditions x 100)
python analyze.py                                               # tables, charts, report.md
```

Outputs land in `results/<provider>_<model>/`: one `<condition>.jsonl` per condition (prediction, scores and the full
trajectory per question), `summary.csv`, `by_hops.csv`, `complementarity.csv`, `report.md`, and charts.

## Project layout

```
config.py              provider presets and settings (.env)
prepare_data.py        100-question stratified sample
tools/                 search, wikidata, calculator
agent/llm.py           OpenAI-compatible client with cache + retries
agent/prompts.py       ReAct / router / executor prompts
agent/react_agent.py   closed-book, ReAct (single or all tools), explicit router
evaluation/metrics.py  EM / F1
run_experiments.py     runs the conditions
analyze.py             comparison report
```
