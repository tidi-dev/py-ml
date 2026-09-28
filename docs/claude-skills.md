# Claude Code Skills, Plugins and Agents — Decision Record

This page explains **what Claude Code tooling this repository uses and why**, so nobody later wonders why something exists (or doesn't). Decisions were made on **2026-09-28** after reviewing the current sources.

The goal was **not** to install as much as possible. It was to build the **smallest setup that helps a beginner learn Python ML correctly**, without AI doing the learning for them.

```text
                LEARNING
                   │
          ┌────────┴────────┐
          ▼                 ▼
    Engineering         ML knowledge
      quality               │
          │                 │
     Superpowers        ml-tutor (agent)
     python-testing-    ml-reviewer (agent)
       patterns         ml-experiment-check (skill)
          │                 │
          └────────┬────────┘
                   ▼
              MY OWN CODE
                   ▼
             UNDERSTANDING
```

## How the pieces are split

| Kind | What it's for | In this repo |
|---|---|---|
| **Project instructions** (`CLAUDE.md`) | Short rules that should *always* apply | Scope, current phase, educational safety rule, code style, Superpowers overrides |
| **Skill** (`.claude/skills/<name>/SKILL.md`) | Reusable know-how loaded *only when relevant* | `python-testing-patterns`, `ml-experiment-check` |
| **Plugin** | A packaged bundle of skills/hooks | Superpowers |
| **Agent** (`.claude/agents/<name>.md`) | A specialised worker with its own instructions and restricted tools | `ml-tutor`, `ml-reviewer` |

Rule of thumb used: no agent where a skill is enough, and no skill for something that fits in `CLAUDE.md`.

---

## Installed

### Superpowers (plugin)

- **Source:** https://github.com/obra/superpowers, via Anthropic's official plugin marketplace (`claude-plugins-official`)
- **Version:** 6.4.1 (the version pinned by the official marketplace at install time; upstream HEAD was 6.4.2)
- **Installation method:** `claude plugin install superpowers@claude-plugins-official --scope project`. This is the official marketplace install, at project scope. It wrote only `.claude/settings.json` (`enabledPlugins`). Plugin files live in the user's Claude Code plugin cache, not in this repo.
- **Purpose:** the *engineering* workflow around the ML project.
- **Relevant workflows here:**
  - `systematic-debugging`: find the root cause before fixing (e.g. a failing test or a notebook error);
  - `verification-before-completion`: run tests/notebooks before claiming something works;
  - `test-driven-development`: for reusable code Claude writes in `src/house_price/`;
  - `brainstorming` → `writing-plans`: only for real engineering changes (e.g. moving notebook logic into `src/`);
  - `requesting-code-review` / `receiving-code-review`: general code review workflow.
- **Security review:**
  - It installs one `SessionStart` hook that reads a local skill file and injects it as context. There's no network access, no credentials and no writes.
  - Optional helper scripts (a local-only brainstorming web page, bash helpers for plan execution) run only when those workflows are used.
  - MIT licence.
- **Project overrides (in `CLAUDE.md`):**
  - TDD's "delete code written before the test" never applies to the learner's code, notebooks or exercises.
  - Learning questions go to the ML Tutor, not `brainstorming`.
  - Plan-execution/subagent workflows must not be used to implement learning milestones.
  - Superpowers itself states that `CLAUDE.md` instructions take precedence over its skills.

### python-testing-patterns (skill)

- **Source:** https://github.com/wshobson/agents, path `plugins/python-development/skills/python-testing-patterns`
- **Installed with:** `DISABLE_TELEMETRY=1 npx skills@1.7.0 add https://github.com/wshobson/agents --skill python-testing-patterns -a claude-code --copy -y`
  - `--copy` puts plain files in `.claude/skills/` (no symlinks, no `.agents/` folder).
  - `skills-lock.json` records the source and content hash.
  - The installed files were checked to be identical to the reviewed source.
- **Purpose:** pytest mechanics: test structure (arrange/act/assert), fixtures, `parametrize`, testing exceptions, one behaviour per test, descriptive test names.
- **Why selected:** pytest is part of the stack and the project has tests. This skill covers the *how* of pytest, which complements Superpowers' TDD *process* instead of duplicating it. It's focused, widely used, recently maintained, contains **no scripts**, and requests no tools or credentials.
- **Caveats:**
  - Some examples are backend-flavoured (API clients, databases, retries), and it suggests `pytest-cov`/`freezegun`. `CLAUDE.md` forbids new dependencies without asking.
  - It suggests a nested `tests/` layout, but this project keeps a flat `tests/` folder.
- **When it's used:** automatically, when writing or fixing tests in `tests/`.

### ml-experiment-check (skill, written for this project)

- **Source:** this repository, `.claude/skills/ml-experiment-check/SKILL.md`
- **Purpose:** a checklist for correct, honest ML experiments:
  - correct X/y construction and prediction-time availability of features (no leakage);
  - split before preprocessing, and a baseline first;
  - the same split for every comparison, and one change per experiment;
  - train vs test error, MAE in dollars, RMSE vs MAE, R² not treated as accuracy;
  - error inspection, reproducibility;
  - dataset quirks (the $500k cap, outliers).
- **Why a project skill instead of a third-party one:** no ML/data-science skill on skills.sh met the bar (see *Considered but not installed*). The best focused candidate was generic, partly inaccurate, and pushed "always use `Pipeline`" and grid-search tuning. That contradicts this project's baseline-first, one-concept-at-a-time approach.
- **When it's used:** automatically when Claude writes or reviews code that splits, trains, evaluates or compares models. It is also preloaded into the `ml-reviewer` agent, so the checklist lives in one place.

---

## Considered but not installed

Candidates were found by searching skills.sh for: python, python best practices, pytest/testing, machine learning, ML engineering, scikit-learn, pandas, numpy, data analysis, data science, jupyter, code review, documentation/teaching. The two finalists (`python-testing-patterns`, `scikit-learn-best-practices`) and Superpowers' hooks were read in full before deciding. The others were rejected after reading their SKILL.md during the research pass.

| Skill | Source | Reason |
|---|---|---|
| scikit-learn-best-practices | mindrally/skills | **Low quality / conflicts with the brief.** Generic list; "always use Pipeline", GridSearch + `n_jobs=-1`; lists `LabelEncoder` for features (a misuse); mostly classification; nothing on dummy baselines or error analysis. |
| scikit-learn | k-dense-ai/scientific-agent-skills | **Too broad + security concern.** Pre-approves Bash via `allowed-tools`, bundles scripts, and tells the agent to fetch and add citations; no baseline guidance. |
| mle-workflow | affaan-m/ecc | **Too advanced.** Good baseline-first/leakage ideas (borrowed for `ml-experiment-check`), but aimed at production serving, monitoring and rollbacks, and depends on ~20 other skills. |
| probabl-ai skills pack | probabl-ai/skills | **Unnecessary dependencies + against the learning goal.** Tied to the `skore`/`skrub` tools and designed for agents to do the implementation. |
| ml-pipeline | jeffallan/claude-skills | **Excluded technology** (MLflow, Kubeflow, Airflow…). |
| machine-learning | mindrally/skills | **Unrelated** (JAX/Flax deep learning). |
| exploratory-data-analysis | k-dense-ai | **Unrelated / heavy** (scientific file formats, 14 scripts). |
| python-code-style | wshobson/agents | **Dependency creep.** Asks to install ruff + mypy in strict mode. `CLAUDE.md` covers the simple style rules. |
| python-anti-patterns | wshobson/agents | **Unrelated.** Backend-oriented (retries, ORMs, async). |
| python-patterns | affaan-m/ecc | **Too broad.** 750 lines incl. async, FastAPI, a long tooling stack. |
| python-pro | jeffallan/claude-skills | **Too advanced.** Async-first, `mypy --strict`, 90% coverage, Poetry. |
| python-best-practices | nathan-gage/python-skills | **Too advanced.** High quality, but aimed at services and data modelling with Pydantic; 77 rule files. |
| python-best-practices | alleneubank/claude-code | **Low quality / too advanced.** Advanced typing patterns; contains an invalid example. |
| pandas-pro, numpy-best-practices, data-analysis-jupyter | various | **Unnecessary.** Nothing beyond Claude's default knowledge at this project's level. |
| requesting/receiving-code-review (standalone) | obra/superpowers | **Duplicate.** Already included in the Superpowers plugin. |
| learning-output-style (plugin) | anthropics/claude-plugins-official | **Duplicate.** An always-on "learner writes the code" mode. It overlaps with the ML Tutor and `CLAUDE.md`'s educational safety rule. |

---

## Project agents

| Agent | Purpose | When to use | Tools |
|---|---|---|---|
| **`ml-tutor`** | Teaches ML concepts with intuition first, California Housing examples, prediction questions, small exercises and checkpoints. Doesn't implement milestones for you. | Learning a new concept; "why does this step exist?"; checking your own explanation; deciding what to learn next. For a whole study session: `claude --agent ml-tutor`. | Read, Grep, Glob, Bash (no edit tools, on purpose) |
| **`ml-reviewer`** | Reviews ML code for leakage, missing baseline, unfair comparisons, metric misinterpretation, overfitting and reproducibility. Reports under *ML correctness / Experiment design / Evaluation / Reproducibility / Code quality*, with hints rather than rewrites. | After writing or changing code that builds X/y, splits, trains, evaluates or compares models. Not for every trivial change. | Read, Grep, Glob, Bash (no edit tools); preloads `ml-experiment-check` |

**No separate Python Reviewer.** Its job (readability, typing, pytest, simplicity) is already covered by `CLAUDE.md`'s code-style rules, `python-testing-patterns`, and Superpowers' code-review workflow. A third reviewer would overlap.

Responsibilities:

```text
ml-tutor              → teaches concepts
ml-reviewer           → checks ML correctness
Superpowers           → engineering workflow (debug, TDD for src/, verify, review)
python-testing-patterns → how to write the pytest tests
CLAUDE.md             → scope, phase, and the "learner writes the code" rule
```

---

## Files and version control

| Path | Commit? | Why |
|---|---|---|
| `CLAUDE.md` | yes | Project instructions |
| `.claude/settings.json` | yes | Enables Superpowers for everyone who opens the project |
| `.claude/agents/*.md` | yes | Project agents |
| `.claude/skills/*/` | yes | Skills are plain Markdown and small. Committing them makes the setup reproducible without re-running the installer |
| `skills-lock.json` | yes | Records where the third-party skill came from, plus its content hash |
| `.claude/settings.local.json` | **no** (git-ignored) | Personal, machine-specific settings |

## Setup after cloning

The skills and agents are already in the repository. Only the plugin needs a one-time install per machine:

```bash
claude plugin install superpowers@claude-plugins-official --scope project
```

To update the third-party skill later, re-run the install command from the *python-testing-patterns* section above (or see `npx skills --help`), then **review the diff** before committing.
