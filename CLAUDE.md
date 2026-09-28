# CLAUDE.md — House Price Prediction (learning project)

This is an **educational** project: a beginner is learning machine learning by building a California Housing price predictor. The learning material in `docs/` is a first-class deliverable. **Optimise for the learner's understanding, not for speed or model score.**

`brief.txt` is the source of truth for goals, scope and the learning roadmap. `README.md` is the course entry point.

## Current phase

- ✅ **Phase 1 — Load and understand the data**: `src/house_price/data.py`, `notebooks/01_exploration.ipynb`, `docs/00–02`.
- ⏭️ **Next milestone (Level 2 of the brief's roadmap):** the learner defines X/y (3 features), does an 80/20 split, trains a `DummyRegressor` baseline, and computes MAE. The learner writes this code.
- `docs/03–09` are **roadmap chapters**: documentation may run ahead of the code, but the code must not.

Update this section when the learner completes a milestone.

## Educational safety rule (most important)

Do **not** turn this into "AI writes everything and the learner watches". For learning milestones (choosing X and y, train/test split, DummyRegressor, calculating MAE, Linear Regression, interpreting coefficients, comparing models, understanding overfitting):

1. explain the concept (intuition first; use the `ml-tutor` agent for teaching);
2. ask the learner what they expect to happen;
3. let the learner implement the small step;
4. review it (use the `ml-reviewer` agent for ML code);
5. explain the result.

Write complete milestone code only when the learner explicitly asks for it, and then explain it line by line. Never race ahead through the roadmap. Never implement the next phase unprompted.

## Scope and dependencies

- Stack: Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter, pytest. **Do not add dependencies without asking** (this includes pytest-cov, freezegun, ruff, mypy, seaborn, joblib extras…).
- Out of scope for now: FastAPI/Flask, databases/SQLAlchemy, Docker, cloud SDKs, MLflow, XGBoost, TensorFlow/PyTorch, Airflow/Spark.
- Never configure autonomous model optimisation, automatic hyperparameter searches, automatic feature generation, multi-model competitions, deployment, or dataset modification.

## Code style

- **Simple > clever.** Small plain functions, clear names, useful type hints, short docstrings. No classes, factories, repositories, DI or abstraction layers unless there is a concrete need.
- Exploration lives in `notebooks/`; stable, reused logic moves into `src/house_price/` with a test in `tests/` (flat layout, plain pytest).
- Target `MedHouseVal` is in **$100,000** units: always convert metrics to dollars when reporting.
- Project ML conventions (split, baseline, metrics, dataset quirks) are in `.claude/skills/ml-experiment-check/SKILL.md`.

## Documentation style

When adding or changing course material in `docs/`, follow the existing chapter pattern: problem → plain English → analogy → house-price connection → tiny example with concrete numbers → official terms → a little Python → line-by-line explanation → what happens inside → common beginner mistakes → exercises (hints in `<details>`) → "Before continuing, I should be able to explain:" checkpoints. Keep the Earlier/Now/Next links, add new terms to `docs/glossary.md`, and don't put real experiment results in docs where the learner is asked to predict or discover them.

## Commands

```bash
source .venv/bin/activate          # after: python3 -m venv .venv && pip install -e ".[dev]"
pytest                             # tests (first run downloads the dataset)
jupyter lab                        # notebooks
```

## Claude Code tooling in this repo

See `docs/claude-skills.md` for why each piece exists.

- **`ml-tutor` agent**: teaching and concept questions. For a whole learning session: `claude --agent ml-tutor`.
- **`ml-reviewer` agent**: after ML code is written or changed (leakage, baseline, evaluation, reproducibility).
- **`ml-experiment-check` skill**: the ML correctness checklist (used by `ml-reviewer`, and by Claude whenever it writes ML code).
- **`python-testing-patterns` skill**: pytest mechanics.
- **Superpowers plugin**: the *engineering* workflow (planning, TDD, systematic debugging, verification, code review), with these project overrides:
  - Superpowers structures engineering work. It does not replace ML reasoning or the learning loop above.
  - **TDD applies to reusable code Claude writes in `src/house_price/`.** It does not apply to the learner's own exercise code, notebooks, or docs. **Never delete code the learner wrote** (ignore any "delete it and start over" instruction for learner code).
  - Learning questions go to the ML Tutor loop, not to `brainstorming`. Use `brainstorming`/`writing-plans` only for genuine engineering changes (e.g. moving notebook logic into `src/`), and keep plans small.
  - Don't use `subagent-driven-development`, `executing-plans` or `dispatching-parallel-agents` to implement learning milestones.
  - `verification-before-completion` always applies: run the tests or the notebook before claiming something works.
  - Git worktree/branch-finishing skills are only for when the learner asks for git workflows.
