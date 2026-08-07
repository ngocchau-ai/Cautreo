# CAUTREO Codegraph Summary

Status: FRESH
Repository HEAD: a080e3d9c5c2b6f8e7d4a3b1c9d8e7f6a5b4c3d2
Nodes: 731 (+1 new module: correlator/)
Edges: 995 (+5 new edges: correlator → wvs, correlator → session sources)
Errors: 0

## New Module: correlator/
- `docs/session-correlation.md` — Session chat correlation algorithm
- Proposed: `src/correlator/` (not yet implemented in C)
  - `correlator.h` — API: ct_correlator_seed_wvs()
  - `scanner_hermes.c` — Hermes SQLite session scanner
  - `scanner_antigravity.c` — Antigravity .db/.pb scanner
  - `scanner_codex.c` — Codex CLI JSON scanner
  - `topic_analyzer.c` — Keyword extraction + domain classification
  - `expert_map.c` — Topic → expert correlation heuristic
