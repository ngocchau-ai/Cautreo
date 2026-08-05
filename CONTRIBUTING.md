# Contributing to CAUTREO

Thank you for your interest in contributing! CAUTREO is an open project and welcomes
contributions that improve correctness, performance, documentation, or test coverage.

---

## Ground Rules

1. **C11, no external runtime dependencies.**
   All code must compile cleanly with `-std=c11 -Wall -Wextra` under clang or gcc.
   Link only to the C standard library (`libc`, `libm`) and system libraries
   (`ws2_32` on Windows for the server).

2. **Each module has one role.**
   Do not add reasoning logic to the inference engine, and do not add generation
   logic to the reasoning core. Respect the decoupled architecture.

3. **Tests first.**
   Every new function should have a corresponding test. New modules require a full
   test suite in `tests/unit/`. Integration changes require a test in `tests/integration/`.

4. **No AI slop.**
   Code must be precise and minimal. Unused parameters, dead branches, and
   boilerplate-padded functions will be rejected.

5. **Document public APIs.**
   All public functions in `.h` files must have a clear comment explaining
   parameters, return values, and ownership semantics (who frees what).

---

## Development Workflow

```bash
# 1. Fork and clone
git clone https://github.com/<you>/CAUTREO.git
cd CAUTREO

# 2. Build
make

# 3. Run all tests
scripts\run_tests.bat         # Windows
# or
make test                     # Unix / MSYS2

# 4. Run integration tests
scripts\run_integration.bat   # Windows
# or
make integration              # Unix / MSYS2

# 5. Make your changes
# ... edit src/ tests/ docs/

# 6. Verify
make clean && make
scripts\run_tests.bat
scripts\run_integration.bat

# 7. Submit a PR
```

---

## Areas Open for Contribution

| Area | What's needed |
|---|---|
| **BPE tokenizer** | Proper SentencePiece/BPE tokenizer to replace the byte-fallback in `backend.c` |
| **Metal backend** | Apple Metal acceleration for the transformer forward pass |
| **CUDA backend** | NVIDIA CUDA attention + FFN kernels |
| **ROCm backend** | AMD ROCm equivalent |
| **Safetensors loader** | `src/gguf/` analog for Safetensors format |
| **Thread pool server** | Multi-connection support for `src/server/` |
| **Benchmark CI** | Automated regression benchmark in GitHub Actions |
| **More causal tests** | Expand the 8 intervention types in `src/core/causal/` |
| **Memory retrieval** | Improve `memory_retrieve_verified()` — top-K candidates, cosine reranking |
| **Documentation** | Tutorials, integration guides, model setup walkthroughs |

---

## Commit Style

```
module: concise description (imperative mood)

Examples:
  engine: add streaming generate API with per-token callback
  memory: fix memory_count signature to take layer parameter
  server: handle keep-alive connections in HTTP parser
  docs: add English architecture documentation
```

---

## Code Style

- `snake_case` for all identifiers.
- `ct_` prefix for public engine/server/agent API.
- `waste_` / `engine_` prefix for WASTE reasoning core API.
- 4-space indentation, no tabs.
- Opening braces on the same line as the statement.
- Maximum line length: 100 characters (soft limit).
- `const`-correct parameters wherever possible.

---

## Reporting Bugs

Open a GitHub Issue with:
1. Operating system and compiler version (`gcc --version` or `clang --version`)
2. The exact command that triggered the bug
3. Full output / error message
4. Minimal reproducer if possible

---

## Questions

Open a GitHub Discussion or Issue labeled `question`.
