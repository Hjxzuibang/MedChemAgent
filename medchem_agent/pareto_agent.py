"""Pareto-front-guided, novelty-regularized closed-loop agent.

Extends :class:`MedChemAgent` to replace the "keep best aggregate" scalar loop
with a Pareto front over RAW (docking, drug-likeness) and diversity/novelty-
regularized anchor selection. The only changes vs. the base agent are
(a) which candidates are shown back to the LLM each round (front anchors, not
top-5 aggregate) and (b) how the front is maintained -- so ablations isolate
exactly this mechanism:

  full        ParetoMedChemAgent(diverse=True,  novelty_weight>0)
  -diversity  ParetoMedChemAgent(diverse=False)   # anchors = top-k by docking
  -Pareto     MedChemAgent                        # base scalar agent
"""
from . import config, pareto
from .agent import MedChemAgent


class ParetoMedChemAgent(MedChemAgent):
    def __init__(self, target_meta: dict, use_rag: bool = True, use_tools: bool = True,
                 feedback: bool = True, verbose: bool = True,
                 n_anchors: int = 3, novelty_weight: float = 0.2,
                 diverse: bool = True):
        super().__init__(target_meta, use_rag=use_rag, use_tools=use_tools,
                         feedback=feedback, verbose=verbose)
        self.n_anchors = n_anchors
        self.novelty_weight = novelty_weight
        self.diverse = diverse
        self.front = []
        self._anchors = []

    # ------------------------------------------------------------------ #
    # Anchor selection
    # ------------------------------------------------------------------ #
    def _select_anchors(self, front):
        if not front:
            return []
        if not self.diverse:
            # ablation: no diversity -- anchors are the top-k by docking
            top = sorted(front, key=lambda c: c.get("affinity") or 0.0)
            return top[:self.n_anchors]
        explored = [c.get("smiles") for c in self.candidates if c.get("smiles")]
        anchors = [max(front, key=lambda c: -(c.get("affinity") or 0.0))]
        picked = {anchors[0].get("smiles")}
        while len(anchors) < self.n_anchors and len(anchors) < len(front):
            best_c, best_score = None, -1.0
            for c in front:
                if c.get("smiles") in picked:
                    continue
                min_sim = min(pareto.tanimoto(c.get("smiles"), s) for s in picked)
                div = 1.0 - min_sim
                nov = pareto.novelty(c.get("smiles"), explored)
                score = div + self.novelty_weight * nov
                if score > best_score:
                    best_c, best_score = c, score
            if best_c is None:
                break
            anchors.append(best_c)
            picked.add(best_c.get("smiles"))
        return anchors

    # ------------------------------------------------------------------ #
    # Feedback prompt (front anchors instead of top-5 aggregate)
    # ------------------------------------------------------------------ #
    def _build_user_prompt(self, n):
        lines = [
            f"Target: {self.target['name']} (co-crystal ligand: {self.target['drug']}).",
            f"Starting hit SMILES: {self.target['hit_smiles']}",
        ]
        if self.feedback and self._anchors:
            lines.append("\nCurrent Pareto-front anchors (diverse trade-off representatives; "
                         "docking lower=better, drug-likeness higher=better):")
            for c in self._anchors:
                lines.append(
                    f"  {c.get('smiles')}  | dock={c.get('affinity')} "
                    f"dl={pareto.druglikeness(c):.2f} qed={c.get('qed', 0):.2f} "
                    f"sa={c.get('sa', 0):.2f}"
                )
            lines.append("\nPropose analogs that push the frontier -- stronger binding (lower "
                         "docking) OR higher drug-likeness -- while diversifying across the "
                         "anchors rather than collapsing onto one region.")
        elif self.feedback and self.candidates:
            lines.append("\nCurrent best candidates (scored):")
            for c in sorted(self.candidates, key=lambda x: -(x.get("aggregate") or 0))[:5]:
                lines.append(f"  {c.get('smiles')} | agg={c.get('aggregate', 0):.2f} "
                             f"dock={c.get('affinity')}")
        else:
            lines.append("\n(No candidates scored yet.)")
        lines.append(f"\nPropose {n} new analogs and return them as JSON with rationale.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, start_smiles, max_rounds=None, proposals_per_round=None):
        max_rounds = max_rounds or config.get("agent", "max_rounds", default=8)
        proposals_per_round = proposals_per_round or config.get("agent", "proposals_per_round", default=5)

        start = self.scorer.score_one(start_smiles)
        if start is not None:
            start["round"] = 0
            start["origin"] = "start"
            self.candidates.append(start)

        for r in range(1, max_rounds + 1):
            self.front = pareto.pareto_front_raw(self.candidates)
            self._anchors = self._select_anchors(self.front)
            smis = self.propose(n=proposals_per_round)
            seen = {c["smiles"] for c in self.candidates}
            novel = [s for s in smis if s and s not in seen]
            if self.verbose:
                print(f"round {r}: front={len(self.front)} anchors={len(self._anchors)} "
                      f"{len(novel)} novel -> scoring", flush=True)
            for p in self.scorer.score_many(novel):
                p["round"] = r
                p["origin"] = "pareto_agent"
                self.candidates.append(p)
            self.round_history.append({
                "round": r, "n_proposals": len(novel),
                "n_front": len(self.front), "n_anchors": len(self._anchors),
            })

        self.scorer.close()
        self.front = pareto.pareto_front_raw(self.candidates)
        self.candidates.sort(key=lambda x: -(x.get("aggregate") or 0))
        return self.candidates
