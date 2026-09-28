# House Price Prediction — Learn Machine Learning by Building

A beginner machine-learning project that predicts California housing values with Python and scikit-learn.

The goal is **not** the best possible score. The goal is to understand, step by step:

> What is happening when I give data to a machine-learning algorithm, train it, evaluate it, and ask it to predict something it has never seen before?

This repository is two things at once:

```text
             PROJECT
                │
       ┌────────┴────────┐
       │                 │
       ▼                 ▼
    CODEBASE         ML COURSE
  (src/, notebooks/,   (docs/)
   tests/)
       │                 │
       ▼                 ▼
experiment with      understand
real Python          what/why/how
       │                 │
       └────────┬────────┘
                ▼
          LEARN ML BY
            BUILDING
```

> **One thing to know up front:** each row of the dataset is a California **census block group** (a small neighbourhood), not one house. So strictly we predict *the median house value of a neighbourhood*. The target, `MedHouseVal`, is in units of **$100,000** (2.5 ≈ $250,000).

---

## Start Here

Follow this order:

1. Read [`docs/00-ml-big-picture.md`](docs/00-ml-big-picture.md): what ML is, with no code.
2. Work through [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb): look at the real data (setup below).
3. Read [`docs/01-understanding-the-data.md`](docs/01-understanding-the-data.md).
4. Read [`docs/02-features-and-target.md`](docs/02-features-and-target.md).
5. Complete the exercises and checkpoints at the end of each of those.
6. **Stop before model training until these concepts make sense.**

Keep the [glossary](docs/glossary.md) open while you read.

---

## Setup

You need Python 3.10 or newer. From the project folder:

```bash
# 1. Create and activate a virtual environment (an isolated place for this project's packages)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the project and its tools
pip install -e ".[dev]"

# 3. Check everything works (the first run downloads the ~400 KB dataset)
pytest

# 4. Open the notebook
jupyter lab notebooks/01_exploration.ipynb
```

What these do:

| Command | Purpose |
|---|---|
| `pip install -e ".[dev]"` | Installs pandas, NumPy, scikit-learn, matplotlib (the project) and Jupyter + pytest (the tools). `-e` means "editable": changes to `src/` take effect immediately. |
| `pytest` | Runs the tests in `tests/`, which check that the data loads and looks the way the course describes it. |
| `jupyter lab` | Opens Jupyter in your browser. (`jupyter notebook` works too.) |

The dataset is downloaded by scikit-learn into `~/scikit_learn_data` the first time it's used. Nothing needs to be committed to this repository.

---

## The learning path

The course covers the whole journey. The **code** only moves forward as you complete each milestone.

```text
Understanding data                          ← docs 01 + notebook   ┐
      ↓                                                            │ current phase
Features / target                           ← docs 02              ┘
      ↓                                                  ─ ─ ─ 🛑 stop point ─ ─ ─
Training / testing                          ← docs 03
      ↓
Baseline                                    ← docs 04
      ↓
Linear Regression                           ← docs 05
      ↓
Evaluation                                  ← docs 06
      ↓
Overfitting                                 ← docs 07
      ↓
Random Forest                               ← docs 08
      ↓
Complete ML workflow                        ← docs 09
```

| # | Chapter | Status |
|---|---|---|
| 00 | [The Big Picture](docs/00-ml-big-picture.md) | ✅ current phase |
| 01 | [Understanding the Data](docs/01-understanding-the-data.md) | ✅ current phase |
| 02 | [Features and Target](docs/02-features-and-target.md) | ✅ current phase |
| 03 | [Training and Testing](docs/03-training-and-testing.md) | 📍 roadmap |
| 04 | [Your First Baseline Model](docs/04-first-baseline-model.md) | 📍 roadmap |
| 05 | [Linear Regression](docs/05-linear-regression.md) | 📍 roadmap |
| 06 | [Evaluating a Model](docs/06-evaluating-a-model.md) | 📍 roadmap |
| 07 | [Overfitting and Generalization](docs/07-overfitting-and-generalization.md) | 📍 roadmap |
| 08 | [Random Forest](docs/08-random-forest.md) | 📍 roadmap |
| 09 | [The Complete ML Workflow](docs/09-ml-workflow.md) | 📍 roadmap |
| — | [Glossary](docs/glossary.md) | reference |

**📍 Roadmap** chapters can be read now for orientation. Their exercises are meant for when you reach that milestone in the project.

Every chapter follows the same pattern: the problem → the idea in plain English → an analogy → back to house prices → a tiny example → the official terms → a little Python, explained line by line → what happens inside → common mistakes → exercises → a checkpoint.

---

## Project structure

```text
.
├── notebooks/
│   └── 01_exploration.ipynb   # Phase 1: inspect the data (no training)
├── src/
│   └── house_price/
│       ├── __init__.py
│       └── data.py            # load_housing_data(), column names
├── tests/
│   └── test_data.py           # checks the data matches what the course describes
├── docs/                      # the ML course (00–09 + glossary)
├── pyproject.toml             # dependencies and project settings
├── brief.txt                  # project requirements and learning roadmap
└── README.md                  # you are here
```

The code grows as you progress. Model training, evaluation and prediction code will be added **by you** in later milestones, not in advance.

---

## Scope

**In scope:** Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter, pytest.

**Deliberately out of scope for now:** web APIs (FastAPI/Flask), databases, Docker, cloud services, MLflow, XGBoost, TensorFlow/PyTorch, Airflow/Spark. The first goal is to correctly train, evaluate, understand and improve a model, not to deploy one.

---

## Claude Code Setup

This repo includes a small [Claude Code](https://code.claude.com) setup designed to help you **learn**, not to write the project for you. Full reasoning: [`docs/claude-skills.md`](docs/claude-skills.md).

| Piece | What it's for |
|---|---|
| `ml-tutor` agent | Explains ML concepts intuition-first, asks you to predict outcomes, gives small exercises. Start a study session with `claude --agent ml-tutor`. |
| `ml-reviewer` agent | Checks your ML code for leakage, a missing baseline, unfair comparisons and misread metrics. Ask: *"use the ml-reviewer agent on notebooks/02_….ipynb"*. |
| `ml-experiment-check` skill | The ML correctness checklist both of the above rely on. |
| `python-testing-patterns` skill | How to write good pytest tests. |
| Superpowers plugin | Engineering workflow: systematic debugging, verification, TDD for `src/`, planning, code review. |
| `CLAUDE.md` | Project rules: current phase, scope, and "the learner writes the milestone code". |

**After cloning**, install the plugin once (the agents and skills are already in the repo):

```bash
claude plugin install superpowers@claude-plugins-official --scope project
```

Suggested rhythm for each milestone: **ml-tutor** (understand) → you write the code → `pytest` → **ml-reviewer** (fix conceptual problems) → verify → continue.

---

## About the dataset

The California Housing dataset comes from the 1990 US Census, was distributed through StatLib, and ships with scikit-learn via `fetch_california_housing()`. Here it's used as a standard educational ML dataset. No specific modern open-data license is claimed. If you ever need explicit redistribution or commercial-use guarantees, check the original source's terms separately.
