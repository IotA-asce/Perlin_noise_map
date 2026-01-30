# Agent Notes (Perlin Noise Map)

This repository is a learning-first Perlin noise lab: implement noise correctly, make the intermediate steps inspectable, and keep outputs deterministic.

## Repo Layout

- `streamlit_app.py`: Streamlit UI (2D map, 3D surface, step-by-step inspector).
- `perlin/`: core noise implementation (pure, testable, minimal dependencies).
- `tests/`: pytest unit tests.
- `requirements.txt`: runtime + test deps.

There is currently no Cursor ruleset (`.cursor/rules/`, `.cursorrules`) and no Copilot instructions (`.github/copilot-instructions.md`). If any appear later, follow them.

## Setup
Use `python3` (macOS/Linux). This machine does not provide `python` by default.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

## Run
```bash
streamlit run streamlit_app.py
```

## Test Commands

Common patterns (prefer `python3 -m pytest ...` if PATH issues):

```bash
pytest
pytest -q
pytest tests/test_perlin2d.py
pytest tests/test_perlin2d.py::test_perlin2d_deterministic_for_seed
pytest -k deterministic
pytest -x
```

## "Build" / Sanity Checks
There is no build step (pure Python). Use these quick checks before committing:

```bash
python3 -m compileall -q .
python3 -m pip check
```

## Lint / Format
Ruff is configured via `pyproject.toml`.
If you add/need formatting locally, prefer `black` (optional):

```bash
ruff check .
python3 -m pip install black
black .
```

If you introduce these tools into the repo, also add configuration (ideally `pyproject.toml`) and update this document.

## Git Policy (Important)
Agents must create a commit for every logical change, and only after tests pass.

Rules:

- If you changed any tracked code/docs, run `pytest` and ensure it passes.
- Do not commit generated files (`__pycache__/`, `.ruff_cache/`, etc.). Keep `.gitignore` updated.
- Make small, focused commits with clear messages.

Suggested flow:

```bash
pytest
git status
git add -A
git commit -m "<concise message>"
```

If tests are failing, fix them before committing.

## Code Style Guidelines
### Imports

- Order imports: standard library, third-party, local (`perlin/...`).
- Use one import per line when it improves clarity.
- Avoid wildcard imports.
- Prefer `from __future__ import annotations` at the top of modules (already used).

### Formatting

- Keep functions small and single-purpose.
- Prefer early returns for simple guard cases.
- Keep docstrings short and factual; add them when behavior is non-obvious (e.g., fade curve definition).
- Aim for an 88 character line length (Black-compatible), but do not add a formatter unless the repo adopts it.

### Types

- Type annotate public functions and methods.
- Use `np.ndarray` for array inputs/outputs; use `float`/`int` for scalar parameters.
- Convert user-facing numeric inputs to concrete types at boundaries (`int(seed)`, `float(scale)`).
- In numeric code, normalize dtypes explicitly (`np.asarray(..., dtype=np.float64)`) to avoid accidental integer math.

### Naming

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private helpers: prefix `_` (e.g., `_noise_map`)

### Error Handling

- Validate inputs at API boundaries and raise `ValueError` with a specific message.
- Keep math kernels (noise generation) pure and deterministic; avoid hidden global state.
- For UI inputs (Streamlit), clamp/guard rather than crashing (e.g., avoid division by zero; handle constant maps).

### Numerical / Numpy Guidelines

- Prefer vectorized operations; avoid Python loops on per-pixel noise when possible.
- Keep hashing/indexing in integer dtypes (`np.int32`) and keep coordinate math in `np.float64`.
- When normalizing, handle degenerate ranges safely (min == max).
- Avoid unnecessary copies; use `np.asarray` and `np.zeros_like` appropriately.

### Streamlit Guidelines

- Heavy computations should be cached with `st.cache_data` (pure functions only).
- Cached functions must be deterministic for the same inputs.
- Keep UI code in `streamlit_app.py`; keep the `perlin/` package UI-agnostic.

### Tests

- Prefer deterministic tests (seeded outputs, stable shapes).
- Test behavior, not implementation details.
- Add tests with new features: correctness, reproducibility, and edge cases.

## When You Change Things

- Update `README.md` and `GOALS.md` when behavior or roadmap changes.
- Update `requirements.txt` if you add runtime/test dependencies.
- Add new tests in `tests/` for every new feature.
