# How to Use This Course

This page explains **how to study with this repository**: where things are, which order to go in, how each chapter works, and what to do when you're stuck. Read it once before you start, and come back whenever you're unsure what to do next.

> **Short version:** read [chapter 00](00-ml-big-picture.md) → run the [exploration notebook](../notebooks/01_exploration.ipynb) → read chapters [01](01-understanding-the-data.md) and [02](02-features-and-target.md) → do their exercises and checkpoints → **stop** until the checkpoints feel easy.

---

## 1. What's in this repository

The repository is two things at once: a small **ML project** you build step by step, and a **course** that explains what you're doing and why.

| Folder / file | What it is | When you use it |
|---|---|---|
| `docs/*.md` | The course chapters as Markdown (easy to read on GitHub or in your editor) | Reading |
| `course/*.ipynb` | **The same chapters as Jupyter notebooks**, with every Python example as a runnable cell | Reading *and* trying the code |
| `notebooks/` | The project's working notebooks. `01_exploration.ipynb` is where you first look at the real data | Doing the project |
| `src/house_price/` | Reusable project code (for now, just the data loader) | Later milestones |
| `tests/` | Checks that the project code works (`pytest`) | After changing `src/` |
| `docs/glossary.md` / `course/glossary.ipynb` | Plain-English definitions of every ML term | Whenever a word feels fuzzy |
| `README.md` | The front page and the chapter list | Finding your way |
| `brief.txt` | The full project requirements and roadmap | For the big picture |

### Markdown or notebook? Pick one.

Every chapter exists twice, with **the same content**:

- **Markdown** (`docs/03-training-and-testing.md`) is best for calm reading, on GitHub or in VS Code's preview.
- **Notebook** (`course/03-training-and-testing.ipynb`) is best when you want to **run the examples** and experiment with them.

You don't need to read both. Many people read the Markdown first, then open the notebook to play with the code.

---

## 2. One-time setup

Everything runs inside **Docker**, so nothing is installed on your own computer. You only need Docker itself (Docker Desktop or OrbStack) installed and running.

The first time, in a terminal from the project folder:

```bash
docker compose up --build          # build the environment (a few minutes, once) and start Jupyter
```

It prints a link like `http://127.0.0.1:8888/lab?token=...`. Open it in your browser. That's JupyterLab, running inside the container.

Every time you come back to study:

```bash
docker compose up                  # start Jupyter again, then open the printed link
```

Press `Ctrl + C` in that terminal when you're done (or run `docker compose down`).

Your project folder is shared with the container. Everything you save in Jupyter (notebooks, code) is stored **on your computer** as normal files, and survives stopping the container. Use the file browser on the left to open `course/` or `notebooks/`.

Useful extra commands, run in a second terminal from the project folder:

| Command | What it does |
|---|---|
| `docker compose run --rm lab pytest` | Runs the project's tests |
| `docker compose logs lab` | Shows Jupyter's output, including the link with the token |
| `docker compose build` | Rebuilds the environment (only needed if `pyproject.toml` dependencies change) |

### Check that your setup works

Run this in a notebook cell. You should see `(20640, 9)`:

```python
from house_price.data import load_housing_data

df = load_housing_data()
print(df.shape)
```

If you get `ModuleNotFoundError: No module named 'house_price'`, you're probably using a Jupyter or Python installed on your own computer instead of the one in the container. Open the `127.0.0.1:8888` link printed by `docker compose up`, and make sure you started it from the project folder.

---

## 3. The order to study in

```text
 ┌───────────────────────── CURRENT PHASE ─────────────────────────┐
 │ 00 Big picture  →  exploration notebook  →  01 Data  →  02 X & y │
 └──────────────────────────────────────────────────────────────────┘
                                 │
                        🛑 STOP POINT: checkpoints of 00–02 feel easy
                                 │
 ┌───────────────────────── ROADMAP (later) ───────────────────────┐
 │ 03 Train/test → 04 Baseline → 05 Linear regression →              │
 │ 06 Evaluation → 07 Overfitting → 08 Random forest → 09 Workflow   │
 └──────────────────────────────────────────────────────────────────┘
```

| Step | Read / do | Goal |
|---|---|---|
| 1 | [00 — The Big Picture](00-ml-big-picture.md) | Understand what ML is, with no code |
| 2 | [`notebooks/01_exploration.ipynb`](../notebooks/01_exploration.ipynb) | Look at the real data yourself |
| 3 | [01 — Understanding the Data](01-understanding-the-data.md) | Understand what you just saw |
| 4 | [02 — Features and Target](02-features-and-target.md) | Understand `X` and `y` |
| 5 | Exercises + checkpoints of 00–02 | Prove to yourself you understand |
| 🛑 | **Stop** | Don't start model training until step 5 feels easy |

### What are "roadmap" chapters?

Chapters 03–09 have a **📍 Roadmap chapter** banner. They're already written so you can see where the course is going. You can skim them for orientation at any time. But **do their exercises only when you reach that milestone** in the project, because the project code deliberately doesn't do those steps yet. You'll write that code yourself.

---

## 4. How every chapter works

Each chapter follows the same pattern, so you always know where you are:

| Section | What it gives you |
|---|---|
| **Course map** + **Where we are** (Earlier / Now / Next) | How this chapter connects to the others |
| 1. What problem are we trying to solve? | *Why* this topic exists |
| 2. The idea in plain English | The concept with no jargon |
| 3. An everyday analogy | Something familiar to hang the idea on |
| 4. Back to house prices | The idea applied to our project |
| 5. A tiny example | Small, concrete numbers you can check by hand |
| 6. The official words | The technical terms, *after* you understand the idea |
| 7. A little Python | A short code example |
| 8. The Python, line by line | What every line does and why |
| 9. What happens inside | What the computer is actually doing |
| 10. Common beginner mistakes | Traps to avoid |
| 11. Exercises | Small tasks for you (with hidden hints) |
| 12. Before continuing, I should be able to explain | Your checkpoint |

If a section feels hard, re-read the **tiny example** (section 5). It's usually the clearest way in.

---

## 5. How to do the exercises

1. **Predict first.** Before running any code, write down what you expect: a number, a shape, a yes/no. Being wrong is where most of the learning happens.
2. **Try on your own.** Write the code in a notebook cell. It's fine if it's messy.
3. **Use the hint only when stuck.** Hints are hidden. Click **▶ Hint** to open one.

   <details>
   <summary>Hint</summary>

   Like this! Try for a few minutes before opening a hint.
   </details>

4. **Compare and explain.** When it works, explain the result in one or two sentences in your own words (a Markdown cell is a good place).

Where do I write exercise code?

- For chapters 00–02: in a new cell at the end of [`notebooks/01_exploration.ipynb`](../notebooks/01_exploration.ipynb), or in the course notebook itself.
- For later milestones: in a new project notebook (for example `notebooks/02_modeling.ipynb`) that you create when you reach chapter 03.

> The course notebooks in `course/` are generated from the Markdown files, and rebuilding them overwrites your changes. It's fine to experiment in them, but keep work you care about in `notebooks/`.

---

## 6. How to use the checkpoints

Each chapter ends with **"Before continuing, I should be able to explain:"** and 3–6 questions.

- Answer each one **out loud or in writing, without looking back**.
- If you can't answer one, re-read the section it came from.
- Only move to the next chapter when every box feels like a ✅.

A good test is to explain it to someone who has never programmed. If you can do that, you understand it.

---

## 7. Tips for working in Jupyter

| Action | How |
|---|---|
| Run a cell | `Shift + Enter` |
| Add a cell below | `B` (when the cell isn't being edited; press `Esc` first) |
| Turn a cell into text (Markdown) | `M` (after `Esc`) |
| Start fresh (clear all variables) | *Kernel → Restart Kernel and Clear Outputs of All Cells* |
| Run everything top to bottom | *Run → Run All Cells* |

Two habits that prevent confusion:

- Run cells **from top to bottom**. A cell may depend on variables created earlier.
- If something behaves strangely, **restart the kernel and run everything again**. Notebooks can keep old variables around.

---

## 8. When you're stuck

1. Look the word up in the [glossary](glossary.md).
2. Re-read the chapter's **tiny example** and **common beginner mistakes**.
3. Read the error message from the bottom up. The last line usually says what went wrong.
4. Ask Claude Code for help. This repository includes two helpers designed for learners:

| Helper | Use it for | How |
|---|---|---|
| **ML Tutor** | "What does this concept mean?", "Why does this step exist?", "Is my explanation right?" | Start a study session with `claude --agent ml-tutor` |
| **ML Reviewer** | Checking *your* ML code for mistakes like data leakage or a missing baseline | In Claude Code: *"use the ml-reviewer agent on notebooks/02_modeling.ipynb"* |

The tutor is designed to **teach, not to write the milestone code for you**. It will explain, ask you to predict, and let you implement. That's on purpose. Details are in [`claude-skills.md`](claude-skills.md).

---

## 9. Track your progress

A simple checklist you can copy into your own notes:

```text
[ ] Setup works (docker compose run --rm lab pytest passes, load_housing_data() prints (20640, 9))
[ ] 00 Big picture — read, checkpoint passed
[ ] Exploration notebook — run, exercises done
[ ] 01 Understanding the data — read, exercises done, checkpoint passed
[ ] 02 Features and target — read, exercises done, checkpoint passed
🛑 Stop point reached — ready for model training
[ ] 03 Training and testing
[ ] 04 Baseline (DummyRegressor + MAE)
[ ] 05 Linear regression
[ ] 06 Evaluation (MAE, RMSE, R²)
[ ] 07 Overfitting
[ ] 08 Random forest
[ ] 09 Complete workflow
```

From chapter 04 onwards, also keep an **experiment log**: one row per experiment, recording what changed, why, and the train/test error. [Chapter 09](09-ml-workflow.md) shows how.

---

## 10. Frequently asked questions

**Do I need to know statistics or maths?**
No. Every idea is explained in plain English first. Maths appears only after the intuition, and every symbol is explained.

**Can I skip ahead to Random Forest?**
You can *read* ahead. But the project only makes sense in order: a model score means nothing without a baseline (chapter 04) and honest testing (chapter 03).

**Why don't the chapters tell me the real results?**
So you can predict them first and then discover them yourself. That's how the ideas stick.

**I edited a course notebook and my changes disappeared.**
The notebooks in `course/` are rebuilt from `docs/*.md`. Keep your own work in `notebooks/`.

**Where do I start again after a break?**
Open this page's progress checklist, find your first unchecked box, and re-do the previous chapter's checkpoint as a warm-up.

---

**Start now:** [00 — The Big Picture →](00-ml-big-picture.md) · [README](../README.md) · [Glossary](glossary.md)
