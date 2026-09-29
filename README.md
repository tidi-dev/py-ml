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

**New here? Read [`docs/how-to.md`](docs/how-to.md) first** (or its notebook version, [`course/how-to.ipynb`](course/how-to.ipynb)). It explains how the course works, how to do the exercises and checkpoints, and what to do when you're stuck.

Then follow this order:

1. Read [`docs/00-ml-big-picture.md`](docs/00-ml-big-picture.md): what ML is, with no code.
2. Work through [`notebooks/01_exploration.ipynb`](notebooks/01_exploration.ipynb): look at the real data (setup below).
3. Read [`docs/01-understanding-the-data.md`](docs/01-understanding-the-data.md).
4. Read [`docs/02-features-and-target.md`](docs/02-features-and-target.md).
5. Complete the exercises and checkpoints at the end of each of those.
6. **Stop before model training until these concepts make sense.**

Keep the [glossary](docs/glossary.md) open while you read.

> 📓 **Prefer notebooks?** Every chapter also exists as a Jupyter notebook in [`course/`](course/), with the same text split into sections and every Python example as a cell you can run. Open them in JupyterLab (see [Setup](#setup)) from the `course/` folder in the file browser. The Markdown files in `docs/` and the notebooks have the same content, so use whichever is easier to read.

---

## Setup

Everything runs in **Docker**. You don't install Python or any packages on your own computer. The only thing you need is Docker (e.g. [Docker Desktop](https://www.docker.com/products/docker-desktop/) or OrbStack), running.

From the project folder:

```bash
# 1. Build the environment and start Jupyter (the first build takes a few minutes)
docker compose up --build

# 2. Open the link it prints, which looks like:
#    http://127.0.0.1:8888/lab?token=...
#    Then open notebooks/01_exploration.ipynb from the file browser on the left.

# 3. When you're done: press Ctrl+C in that terminal (or run: docker compose down)
```

Other commands (run them in a second terminal, from the project folder):

```bash
docker compose run --rm lab pytest                                   # run the tests
docker compose run --rm lab python scripts/build_course_notebooks.py # rebuild course/ after editing docs/
```

What's going on:

| Piece | Purpose |
|---|---|
| `Dockerfile` | Describes the environment: Python 3.13 plus pandas, NumPy, scikit-learn, matplotlib, Jupyter and pytest, installed **inside the image**, not on your machine. |
| `compose.yaml` | Starts that environment as a service called `lab`, with Jupyter on `127.0.0.1:8888` (reachable only from your computer). |
| Project folder → `/app` | Your project folder is shared with the container, so notebooks you edit and code you write are saved **on your machine** as normal files. |
| `sklearn-data` volume | The dataset (~400 KB) is downloaded the first time it's used and kept in a Docker volume, so it isn't downloaded again. |
| `docker compose run --rm lab pytest` | Runs the tests in `tests/` inside the container. They check that the data loads and looks the way the course describes it. |

You only need to rebuild the image (`docker compose up --build` or `docker compose build`) if the dependencies in `pyproject.toml` change. Changes to `src/`, notebooks and docs take effect immediately.

If the link doesn't work, run `docker compose logs lab` and look for the line starting with `http://127.0.0.1:8888/lab?token=`. If port 8888 is already in use, stop the other Jupyter, or change the first `8888` in `compose.yaml` to another number (e.g. `"127.0.0.1:8890:8888"`) and open that port instead.

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

| # | Chapter (Markdown) | Notebook | Status |
|---|---|---|---|
| — | [How to Use This Course](docs/how-to.md) | [📓 open](course/how-to.ipynb) | 👉 start here |
| 00 | [The Big Picture](docs/00-ml-big-picture.md) | [📓 open](course/00-ml-big-picture.ipynb) | ✅ current phase |
| 01 | [Understanding the Data](docs/01-understanding-the-data.md) | [📓 open](course/01-understanding-the-data.ipynb) | ✅ current phase |
| 02 | [Features and Target](docs/02-features-and-target.md) | [📓 open](course/02-features-and-target.ipynb) | ✅ current phase |
| 03 | [Training and Testing](docs/03-training-and-testing.md) | [📓 open](course/03-training-and-testing.ipynb) | 📍 roadmap |
| 04 | [Your First Baseline Model](docs/04-first-baseline-model.md) | [📓 open](course/04-first-baseline-model.ipynb) | 📍 roadmap |
| 05 | [Linear Regression](docs/05-linear-regression.md) | [📓 open](course/05-linear-regression.ipynb) | 📍 roadmap |
| 06 | [Evaluating a Model](docs/06-evaluating-a-model.md) | [📓 open](course/06-evaluating-a-model.ipynb) | 📍 roadmap |
| 07 | [Overfitting and Generalization](docs/07-overfitting-and-generalization.md) | [📓 open](course/07-overfitting-and-generalization.ipynb) | 📍 roadmap |
| 08 | [Random Forest](docs/08-random-forest.md) | [📓 open](course/08-random-forest.ipynb) | 📍 roadmap |
| 09 | [The Complete ML Workflow](docs/09-ml-workflow.md) | [📓 open](course/09-ml-workflow.ipynb) | 📍 roadmap |
| — | [Glossary](docs/glossary.md) | [📓 open](course/glossary.ipynb) | reference |

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
├── docs/                      # the ML course (00–09 + glossary), Markdown
├── course/                    # the same course as runnable notebooks
├── scripts/
│   └── build_course_notebooks.py  # rebuilds course/ from docs/
├── Dockerfile                 # the development environment (Python + packages, inside Docker)
├── compose.yaml               # starts that environment: JupyterLab, tests, scripts
├── .dockerignore              # keeps the Docker build small
├── pyproject.toml             # dependencies and project settings
├── brief.txt                  # project requirements and learning roadmap
└── README.md                  # you are here
```

The code grows as you progress. Model training, evaluation and prediction code will be added **by you** in later milestones, not in advance.

---

## Scope

**In scope:** Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter, pytest, and Docker **only as the development environment** (so nothing is installed on your machine).

**Deliberately out of scope for now:** web APIs (FastAPI/Flask), databases, deploying models with Docker, cloud services, MLflow, XGBoost, TensorFlow/PyTorch, Airflow/Spark. The first goal is to correctly train, evaluate, understand and improve a model, not to deploy one.

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

Suggested rhythm for each milestone: **ml-tutor** (understand) → you write the code → `docker compose run --rm lab pytest` → **ml-reviewer** (fix conceptual problems) → verify → continue.

---

## About the dataset

The California Housing dataset comes from the 1990 US Census, was distributed through StatLib, and ships with scikit-learn via `fetch_california_housing()`. Here it's used as a standard educational ML dataset. No specific modern open-data license is claimed. If you ever need explicit redistribution or commercial-use guarantees, check the original source's terms separately.
