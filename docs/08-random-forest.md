# 08 — Random Forest: Many Simple Questions, Many Opinions

> **Course map:** [00 Big picture](00-ml-big-picture.md) → [01 Data](01-understanding-the-data.md) → [02 Features & target](02-features-and-target.md) → [03 Training & testing](03-training-and-testing.md) → [04 Baseline](04-first-baseline-model.md) → [05 Linear regression](05-linear-regression.md) → [06 Evaluation](06-evaluating-a-model.md) → [07 Overfitting](07-overfitting-and-generalization.md) → **08 Random forest** → [09 Workflow](09-ml-workflow.md) · [Glossary](glossary.md)

> 📍 **Roadmap chapter.** The project code doesn't do this yet. According to the project brief, Random Forest comes only **after** the baseline → Linear Regression → evaluation workflow makes sense to you.

| Where we are | |
|---|---|
| **Earlier** | Linear Regression fits straight-line relationships ([chapter 05](05-linear-regression.md)). We learned to compare training and test error to spot overfitting ([chapter 07](07-overfitting-and-generalization.md)). |
| **Now** | Understand a model that can learn **curved and combined** relationships: first a single decision tree, then a forest of them. |
| **Next** | Put every step together into a repeatable workflow ([chapter 09](09-ml-workflow.md)). |

---

## 1. What problem are we trying to solve?

Linear Regression makes a strong assumption: **each feature has a straight-line effect**, the same everywhere. Real housing markets don't always behave like that.

**Example 1 — Location.** In California, two regions are especially expensive: the San Francisco Bay Area (latitude ≈ 37.8) and the Los Angeles coast (latitude ≈ 34). The land between them is generally cheaper.

```text
Value
  ↑
  │     ●●                         ●●
  │    ●  ●                       ●  ●
  │   ●    ●                     ●    ●
  │  ●      ●●●●●●●●●●●●●●●●●●●●●      ●
  └──────────────────────────────────────→ Latitude
       LA (≈34)       cheaper       SF (≈37.8)
```

A straight line through latitude can only say "further north → more expensive" or "further north → cheaper". It **cannot** say "expensive here, cheap there, expensive again".

**Example 2 — Effects that level off.** Going from 2 to 4 rooms might matter a lot. Going from 9 to 11 rooms probably matters much less. A straight line adds the *same* amount for every extra room.

**Example 3 — Effects that depend on each other.** "Old houses" might be cheap in rural areas but very expensive in historic city centres. The effect of age *depends on* location. A plain linear model adds effects separately, so it can't express "it depends".

---

## 2. The idea in plain English

### A decision tree: a game of "20 questions"

Instead of a formula, a **decision tree** predicts by asking a series of yes/no questions about the features:

```text
Is MedInc > X?
        │
     yes/no
        │
        ▼
another question
        │
        ▼
      ...
        │
        ▼
final answer: the average value of the training areas that ended up here
```

Each question splits the areas into two groups. It keeps asking until it reaches a final group, called a **leaf**. The prediction is the **average value of the training areas in that leaf**.

### A forest: ask many trees and average

A single tree is quick to overfit ([chapter 07](07-overfitting-and-generalization.md)) and sensitive to small changes in the data. So:

> Instead of trusting one decision tree, build many slightly different trees and combine their predictions.

Each tree makes its own mistakes, and many of those mistakes cancel out when you average.

---

## 3. An everyday analogy

**Tree:** a doctor's flowchart. *"Fever? → yes → Cough? → no → Rash? → …"* Each answer leads to the next question, until you reach a conclusion.

**Forest:** asking **100 estate agents** for a valuation, where each agent has seen a *different random selection* of past sales. Any one agent might be biased by the particular houses they happened to see. The *average* of 100 opinions is usually more reliable than any single one. This is the "wisdom of the crowd".

---

## 4. Back to house prices

Here's a real (small) tree trained on our training set using only `MedInc`, with at most 2 questions in a row (thresholds rounded):

```text
                        Is MedInc ≤ 5.1?
                     ┌───────yes──┴──no────────┐
                     ▼                         ▼
             Is MedInc ≤ 3.1?          Is MedInc ≤ 6.9?
             ┌──yes──┴──no──┐          ┌──yes──┴──no──┐
             ▼              ▼          ▼              ▼
          predict        predict    predict        predict
           1.36           2.09       2.95           4.26
         (≈$136k)       (≈$209k)   (≈$295k)       (≈$426k)
```

An area with `MedInc = 4.0` goes: "≤ 5.1? yes" → "≤ 3.1? no" → **predict 2.09 (≈ $209,000)**. That number is simply the average `MedHouseVal` of all training areas with income between 3.1 and 5.1.

With all 8 features and more depth, the tree can ask questions like *"Is Latitude ≤ 38.0? … Is Longitude ≤ −122.0? … Is MedInc > 6?"*. Combined, those questions can carve out "rich neighbourhood on the Bay Area coast", something a straight line can't express.

---

## 5. A tiny example

Six made-up training areas:

```text
Area   MedInc   Coastal?   Value
A      2        no         $120k
B      3        no         $150k
C      2        yes        $260k
D      7        no         $320k
E      8        yes        $480k
F      3        yes        $280k
```

A tree might first ask **"Is it coastal?"**:

```text
Coastal? ── no ──►  A, B, D   → average ($120k + $150k + $320k) / 3 = $197k
         └─ yes ─►  C, E, F   → average ($260k + $480k + $280k) / 3 ≈ $340k
```

Then, inside each group, **"Is MedInc > 5?"**:

```text
not coastal, MedInc ≤ 5  → A, B  → $135k
not coastal, MedInc > 5  → D     → $320k
coastal,     MedInc ≤ 5  → C, F  → $270k
coastal,     MedInc > 5  → E     → $480k
```

A new coastal area with MedInc = 2.5 → **$270k**.

Now a forest: three trees, each trained on a slightly different random selection of rows, give $250k, $290k and $275k for that area. The forest predicts their average: **$271.7k**.

---

## 6. The official words

| Plain English | Official term |
|---|---|
| A relationship that isn't a straight line | **Nonlinear relationship** |
| When one feature's effect depends on another | **Interaction** |
| A model of nested yes/no questions | **Decision tree** |
| One yes/no question | **Split** (a feature plus a **threshold**) |
| A final group at the bottom of the tree | **Leaf** |
| How many questions in a row the tree may ask | **Depth** (`max_depth`) |
| Many trees whose predictions are averaged | **Random Forest** |
| Combining several models into one | **Ensemble** |
| Each tree trains on a random sample of rows, drawn with replacement | **Bootstrap sampling** ("bagging") |
| Settings *you* choose before training, like `max_depth` or number of trees | **Hyperparameters** |

---

## 7. A little Python

Compare all three models on the **same** split, with **both** training and test MAE:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Dummy": DummyRegressor(strategy="mean"),
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_mae = mean_absolute_error(y_train, model.predict(X_train))
    test_mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"{name:<18} train MAE={train_mae:.3f}   test MAE={test_mae:.3f}")
```

(Training the forest takes a few seconds: it builds 100 trees.)

Fill this in when you reach this milestone:

| Model | Train MAE | Test MAE | Test MAE in $ | Gap (test − train) |
|---|---:|---:|---:|---:|
| Dummy | | | | |
| Linear Regression | | | | |
| Random Forest | | | | |

---

## 8. The Python, line by line

| Code | What it does |
|---|---|
| `from sklearn.ensemble import RandomForestRegressor` | "Ensemble" is scikit-learn's module for models built from many models. |
| `RandomForestRegressor(random_state=42)` | 100 trees by default (`n_estimators=100`). The trees are random, so `random_state` makes the forest identical every run. |
| `models = {...}` | A dictionary of name → untrained model. Every model has the same `fit`/`predict` interface, so one loop can handle them all. |
| `model.fit(X_train, y_train)` | For the forest: build 100 trees, each on its own random sample of the training rows. |
| `model.predict(X_test)` | For the forest: send each area down all 100 trees and average the 100 answers. |
| `train_mae` and `test_mae` | Both, so we can compare **performance** *and* spot **overfitting** (chapter 07). |

---

## 9. What happens inside (high level)

### How a tree chooses its questions

At each step the tree tries **many possible questions**: every feature, and many thresholds for each. For each candidate question it asks: *"if I split here, how similar are the values inside each of the two groups?"* It picks the question that makes the groups **most similar inside** (the smallest squared error around each group's average). Then it repeats inside each group.

So the tree isn't told "ask about income first". It *discovers* that income is the most useful first question because that split reduces error the most.

### How a forest makes its trees different

If every tree saw the same data, they'd all be identical, and averaging them would be pointless. Random Forest makes them differ by training each tree on a **bootstrap sample**: a random draw of rows from the training set, with some rows picked several times and others left out. (It can also consider only a random subset of features at each question.) Different data → different trees → different mistakes → averaging cancels many of them.

### Linear Regression vs Random Forest

| | Linear Regression | Random Forest |
|---|---|---|
| Shape of relationships | Straight lines, added together | Step-like pieces that can approximate curves and interactions |
| What it stores | 1 intercept + 1 coefficient per feature (9 numbers for 8 features) | About 2 million questions and leaf values (100 deep trees) |
| Can you explain one prediction? | Yes, easily ([chapter 05](05-linear-regression.md)) | Much harder: 100 trees |
| Training error | Usually close to test error | Usually much lower than test error |
| Predicting outside the range seen in training | Can extend its lines, for better or worse | **Cannot**: every prediction is an average of training values, so it never predicts above the highest training value |
| Cares about units / scale of features? | Coefficients do | No: questions are just "above or below this threshold?" |
| Speed | Instant | Seconds (more with more trees) |

Neither is "better" in general. Different algorithms represent relationships differently, so **they must be compared empirically**, on the same test set, against the same baseline.

---

## 10. Common beginner mistakes

- **Starting with Random Forest.** Without a baseline and a simple model, you can't tell how much its complexity is actually buying you.
- **Being impressed by the training error.** A forest's training error is typically far below its test error. Judge it by the test error.
- **Assuming the more powerful model wins.** Measure. Then ask: *is the improvement big enough to justify a slower, harder-to-explain model?*
- **Comparing on different splits** (or different feature sets) and calling it a model comparison.
- **Tuning dozens of settings to squeeze out a slightly better score.** That's not the goal of this project. Understanding *why* one model beats another is.
- **Expecting a forest to predict values it never saw.** It can't predict $600k if the training data tops out at $500k.

---

## 11. Exercises

**Exercise 1 — Follow the tree.**
Using the real tree in section 4, what does it predict for areas with `MedInc` = 2.0, 5.0 and 9.0? Convert to dollars.

**Exercise 2 — Predict the table.**
Before running section 7's code, predict the *order* of the three models by test MAE, and which one will have the biggest train/test gap. Then run it.

**Exercise 3 — Is it worth it?**
How many dollars of test MAE does Random Forest save compared with Linear Regression? Given that it's slower and harder to explain, would you choose it for this project? Justify your answer.

**Exercise 4 — Look inside one tree.**
Train `DecisionTreeRegressor(max_depth=2)` on `X_train[["MedInc"]]` and print it with `sklearn.tree.export_text`. Do you get the same thresholds as section 4?

<details>
<summary>Hint</summary>

`from sklearn.tree import DecisionTreeRegressor, export_text`, then `print(export_text(tree, feature_names=["MedInc"]))`.
</details>

---

## 12. Before continuing, I should be able to explain:

- □ Give an example of a relationship in housing data that a straight line can't capture.
- □ How does a decision tree make a prediction for one area?
- □ Why does a forest use many trees instead of one, and how are the trees made different?
- □ Why is a Random Forest's training error usually much lower than its test error?
- □ When might Linear Regression still be the better choice?
- □ Why must models be compared on the same test set and against a baseline?

---

**Previous:** [← 07 — Overfitting and Generalization](07-overfitting-and-generalization.md) · **Next:** [09 — The Complete ML Workflow →](09-ml-workflow.md)
