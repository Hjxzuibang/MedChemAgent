# MedChemAgent

**Metric alignment in multi-objective hit-to-lead optimization: a closed-loop
large language model agent.**

MedChemAgent is a tool-grounded large language model (LLM) agent that runs a
closed-loop hit-to-lead optimization campaign. Given a protein target and a
starting hit, it retrieves literature (RAG), proposes analogs through an LLM
with function calling, scores them with RDKit drug-likeness / ADMET filters and
smina docking, and iterates — keeping the best candidates and a Pareto front.
The central finding is a *metric-mismatch*: docking-only evaluation
systematically undervalues LLM agents in hit-to-lead optimization, because the
agent's real gain is in drug-likeness while docking-supervised baselines chase
affinity alone.

## Repository layout

```
medchem_agent/            Core library (agent, scoring, Pareto front, tools)
scripts/                  Benchmark / ablation / validation entry points
data/targets/             Six targets: target.json + prepared receptor.pdb
data/pdb/                 Deposited co-crystal structures (redocking validation)
config.yaml               Runtime configuration (LLM endpoint, docking, weights)
environment.yml           Conda environment
CITATION.cff              Citation metadata
data_availability.md      Target / PDB accessions and reproducibility notes
```

## Installation

```bash
# 1. Create the environment (installs RDKit + smina)
conda env create -f environment.yml
conda activate medchem-agent

# 2. Install the local package
pip install -e .
```

The docking engine is [smina](https://sourceforge.net/projects/smina/) (an
AutoDock Vina fork), expected on `PATH` as `smina` — override with the
`SMINA_BIN` environment variable if it lives elsewhere.

## Configuration

Set the LLM endpoint and API key before running. The API key is read from the
`OPENAI_API_KEY` environment variable (or, alternatively, from a JSON file such
as `{"OPENAI_API_KEY": "..."}` referenced by `llm.auth_file`).

```yaml
# config.yaml (excerpt)
llm:
  base_url: "https://api.openai.com/v1"   # any OpenAI-compatible endpoint
  model: "gpt-4o"
```

All paths are repository-root-relative; run the scripts from the repository
root.

## Quick start

Run the agent on one target and save the scored candidates:

```bash
python scripts/run_agent.py --target EGFR --out results/EGFR_agent.json
```

The script scores the starting hit, then runs the closed loop (default 8 rounds
× 5 proposals) and dumps the Pareto front and the top candidates.

## Reproducing the paper's experiments

```bash
# Head-to-head benchmark: Pareto agent vs scalar agent vs GA vs baselines
python scripts/run_benchmark.py --seeds 0 1 2 3 4 5 \
    --targets EGFR CDK2 ABL1 BRAF VEGFR2 SRC

# Statistical comparison of the agent against each baseline (per target)
python scripts/run_significance.py

# Ablation: full agent vs no_rag / no_tools / no_feedback
python scripts/run_ablation.py --repeats 6
python scripts/run_ablation_significance.py

# Orthogonal validation of the smina docking protocol
python scripts/orthogonal_validation.py reproducibility   # score stability
python scripts/orthogonal_validation.py redock            # co-crystal RMSD
```

## Method

The agent and every baseline share one `Scorer` (`medchem_agent/evaluate.py`),
so any difference is attributable to the method, not to a larger budget. Four
objectives — docking affinity, QED, synthesizability (SA), and ADMET filter
pass-rate — are normalized to `[0, 1]` and combined into a single aggregate; the
Pareto front over raw docking vs. drug-likeness is tracked separately
(`medchem_agent/pareto.py`).

## Reproducibility note

The LLM loop is stochastic (temperature 0.7), so exact candidate lists vary
across runs and models. The runs reported in the paper used a closed-source LLM via an
OpenAI-compatible endpoint (the exact model is stated in the paper); plug in
your own endpoint via `config.yaml`. Docking and all baseline/ablation logic are deterministic given
the same `smina` seed.

## Data availability

See [data_availability.md](data_availability.md) for the six PDB structures,
the starting-hit SMILES, and the docking-box definitions.

## Citation

If you use this software, please cite it as described in
[CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE)
