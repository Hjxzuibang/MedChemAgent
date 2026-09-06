"""MedChemAgent: a tool-grounded LLM agent for closed-loop hit-to-lead optimization.

The agent acts as a medicinal-chemist "digital twin":
  - retrieves literature (RAG) about the target,
  - proposes analogs via an LLM (persona + function-calling tools),
  - scores candidates with RDKit / ADMET filters / docking,
  - iterates, keeping the best candidates and a Pareto front.
"""
import json
import re

from . import config, evaluate, llm
from .tools import rag

SYSTEM_PROMPT = """You are a senior medicinal chemist running a hit-to-lead optimization campaign.

Given a protein target and a starting hit compound, propose chemically valid analogs
that improve the hit across four objectives simultaneously:
  1. Binding affinity  (lower docking score in kcal/mol = better)
  2. Drug-likeness     (higher QED = better)
  3. Synthesizability  (lower SA score = better)
  4. ADMET safety      (pass more Lipinski/Veber/PAINS/Brenk filters)

Guidelines:
- Make small, drug-like modifications: bioisosteric replacements, adding/removing polar
  groups, rigidification, scaffold hopping, tuning solubility without destroying potency.
- Prefer synthetically accessible molecules (no exotic functional groups, no charged
  species, no metals).
- You may call `score_molecule` to check fast properties, and `search_literature` for
  known SAR of this target before finalizing proposals.
- Do NOT simply repeat the starting molecule; make meaningful modifications.
- Return your FINAL answer as a JSON object:
  {"proposals": [{"smiles": "...", "rationale": "..."}]}
  with exactly the requested number of proposals, all SMILES chemically valid.
"""

SCORE_TOOL = {
    "type": "function",
    "function": {
        "name": "score_molecule",
        "description": "Compute fast drug-likeness (QED, SA, MW, logP, TPSA, HBD/HBA, "
                       "Lipinski) and ADMET filter pass-rate for a SMILES. No docking.",
        "parameters": {
            "type": "object",
            "properties": {"smiles": {"type": "string", "description": "SMILES string"}},
            "required": ["smiles"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_literature",
        "description": "Search PubMed for medicinal-chemistry / SAR information relevant "
                       "to the given query (e.g. '<target> inhibitor SAR').",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}

TOOLS = [SCORE_TOOL, SEARCH_TOOL]


def _extract_json(text: str):
    """Best-effort extraction of a JSON object from an LLM reply."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


class MedChemAgent:
    def __init__(self, target_meta: dict, use_rag: bool = True, use_tools: bool = True,
                 feedback: bool = True, verbose: bool = True):
        self.target = target_meta
        self.verbose = verbose
        self.use_rag = use_rag
        self.use_tools = use_tools
        self.feedback = feedback
        self._rag = rag.PubMedRAG(
            str(config.resolve_path(config.get("paths", "pubmed_cache", default="data/pubmed_cache"))))
        self._literature = self._retrieve_literature() if use_rag else ""
        self.scorer = evaluate.Scorer(target_meta)
        # all scored candidates seen so far
        self.candidates = []
        self.round_history = []

    # ------------------------------------------------------------------ #
    # RAG
    # ------------------------------------------------------------------ #
    def _retrieve_literature(self, top_k: int = 8) -> str:
        query = f"{self.target['name']} inhibitor structure-activity relationship"
        try:
            docs = self._rag.retrieve(query, top_k=top_k)
        except Exception as e:  # noqa: BLE001 - literature is best-effort
            if self.verbose:
                print(f"  [rag] retrieval failed: {e}")
            return ""
        lines = []
        for d in docs:
            ab = d["abstract"][:500]
            lines.append(f"- {d['title']} (PMID {d['pmid']}): {ab}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _fast_score(self, smi: str):
        """RDKit + ADMET (no docking). Returns a compact dict or None."""
        return evaluate.fast_score(smi)

    # ------------------------------------------------------------------ #
    # LLM proposal (with function calling)
    # ------------------------------------------------------------------ #
    def _run_tool(self, name: str, args: dict) -> str:
        if name == "score_molecule":
            p = self._fast_score(args.get("smiles", ""))
            return json.dumps(p if p else {"error": "invalid SMILES"})
        if name == "search_literature":
            try:
                docs = self._rag.retrieve(args.get("query", ""), top_k=5)
            except Exception as e:  # noqa: BLE001
                return json.dumps({"error": str(e)})
            return json.dumps([{"title": d["title"], "abstract": d["abstract"][:400]} for d in docs])
        return json.dumps({"error": f"unknown tool {name}"})

    def propose(self, n: int = 5, max_tool_steps: int = 6):
        """Ask the LLM for n proposals, allowing tool calls (score/search)."""
        user = self._build_user_prompt(n)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + ("\n\n## Literature context\n" + self._literature if self._literature else "")},
            {"role": "user", "content": user},
        ]
        if not self.use_tools:
            # Ablation: no function-calling -- a single plain generation.
            resp = llm.chat(messages)
            text = resp.choices[0].message.content or ""
            data = _extract_json(text)
            if data and "proposals" in data:
                return [p.get("smiles", "") for p in data["proposals"]]
            return re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#/.\\]{3,}", text)[:n]
        for _ in range(max_tool_steps):
            resp = llm.chat(messages, tools=TOOLS, tool_choice="auto")
            msg = resp.choices[0].message
            if msg.tool_calls:
                messages.append({"role": "assistant", "content": msg.content or "",
                                 "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments or "{}")
                    result = self._run_tool(tc.function.name, args)
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
            else:
                data = _extract_json(msg.content or "")
                if data and "proposals" in data:
                    return [p.get("smiles", "") for p in data["proposals"]]
                # fallback: try to find SMILES in the raw text
                return re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#/.\\]{3,}", msg.content or "")[:n]
        return []

    def _build_user_prompt(self, n: int) -> str:
        lines = [
            f"Target: {self.target['name']} (co-crystal ligand: {self.target['drug']}).",
            f"Starting hit SMILES: {self.target['hit_smiles']}",
        ]
        if self.feedback and self.candidates:
            druglike = [
                c for c in self.candidates
                if c.get("affinity") is not None
                and (c.get("qed") or 0.0) >= 0.5
                and (c.get("sa") or 99.0) <= 4.0
            ]
            if druglike:
                lines.append("\nCurrent drug-like candidates (QED >= 0.5, SA <= 4; "
                             "ranked by docking affinity, lower is better):")
            else:
                lines.append("\nNo scored candidate yet passes the drug-like gate "
                             "(QED >= 0.5, SA <= 4).")
            for c in sorted(druglike, key=lambda x: x["affinity"])[:5]:
                lines.append(
                    f"  {c['smiles']}  | dock={c.get('affinity')} qed={c.get('qed', 0):.2f} "
                    f"sa={c.get('sa', 0):.2f} admet={c.get('admet_pass')}/{c.get('admet_total')}"
                )
            lines.append("\nPropose analogs that improve docking affinity (more negative) "
                         "while remaining inside the drug-like gate.")
        else:
            lines.append("\n(No candidates scored yet.)")
        lines.append(f"\nPropose {n} new analogs and return them as JSON with rationale.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, start_smiles: str, max_rounds: int = None, proposals_per_round: int = None):
        max_rounds = max_rounds or config.get("agent", "max_rounds", default=8)
        proposals_per_round = proposals_per_round or config.get("agent", "proposals_per_round", default=5)

        # score the starting hit first
        start = self.scorer.score_one(start_smiles)
        if start is not None:
            start["round"] = 0
            start["origin"] = "start"
            self.candidates.append(start)

        for r in range(1, max_rounds + 1):
            smis = self.propose(n=proposals_per_round)
            seen = {c["smiles"] for c in self.candidates}
            novel = [s for s in smis if s and s not in seen]
            if self.verbose:
                print(f"round {r}: {len(novel)} novel proposals -> scoring", flush=True)

            for p in self.scorer.score_many(novel):
                p["round"] = r
                p["origin"] = "agent"
                self.candidates.append(p)
            self.round_history.append({"round": r, "n_proposals": len(novel)})

        self.scorer.close()

        # sort + pareto
        self.candidates.sort(key=lambda x: -(x.get("aggregate") or 0))
        return self.candidates
