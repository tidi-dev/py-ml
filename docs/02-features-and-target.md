# 02 — Features and Target: What Goes In, What Comes Out

> **Course map:** [00 Big picture](00-ml-big-picture.md) → [01 Data](01-understanding-the-data.md) → **02 Features & target** → [03 Training & testing](03-training-and-testing.md) → [04 Baseline](04-first-baseline-model.md) → [05 Linear regression](05-linear-regression.md) → [06 Evaluation](06-evaluating-a-model.md) → [07 Overfitting](07-overfitting-and-generalization.md) → [08 Random forest](08-random-forest.md) → [09 Workflow](09-ml-workflow.md) · [Glossary](glossary.md)

| Where we are | |
|---|---|
| **Earlier** | We inspected the dataset: 20,640 neighbourhoods, 9 numeric columns, one of which (`MedHouseVal`) is the value we care about ([chapter 01](01-understanding-the-data.md)). |
| **Now** | Split the columns into two roles: the **information** the model receives (`X`) and the **answer** it must learn to produce (`y`). |
| **Next** | Split the *rows* into a part for learning and a part for testing ([chapter 03](03-training-and-testing.md)). |

This chapter is short but **extremely important**. Getting `X` and `y` wrong is one of the most common ways ML projects go silently wrong.

---

## 1. What problem are we trying to solve?

Our table has 9 columns. They don't all play the same role:

- **8 columns** describe the neighbourhood. We know them in advance.
- **1 column**, `MedHouseVal`, is the thing we want to **guess**.

A training algorithm can't know which column is which unless we tell it. So we have to separate the table into "what you get to look at" and "what you're trying to predict".

---

## 2. The idea in plain English

Imagine someone gives you information about a house and asks you to guess its value.

- The **information you receive** (the area's income, the age of the houses, the number of rooms…) is **`X`**.
- The **correct answer you are trying to learn to produce** (the real median house value) is **`y`**.

```text
X = information provided TO the model
y = the answer the model is trying to learn
```

During training, the model sees **both** `X` and `y` for each example, so it can learn how they relate. Later, when predicting, it sees **only `X`** and must produce its own guess of `y`.

---

## 3. An everyday analogy

Think of a quiz card:

```text
┌─────────────────────────────────────────────┐
│ FRONT (the question)                        │
│   Income: $50,000   Age: 20 yrs   Rooms: 5  │
├─────────────────────────────────────────────┤
│ BACK (the answer)                           │
│   Value: $250,000                           │
└─────────────────────────────────────────────┘
```

- **While practising** (training), you read the front, guess, then flip the card to check the back. That's how you learn.
- **In the real world** (prediction), you only ever get the front. Nobody hands you the back. That's the whole point: you're asked because the answer isn't known yet.

The **front** is `X`. The **back** is `y`.

---

## 4. Back to house prices

For the California Housing data:

```text
X  (8 features — the front of the card)
├── MedInc
├── HouseAge
├── AveRooms
├── AveBedrms
├── Population
├── AveOccup
├── Latitude
└── Longitude

        ↓
      model
        ↓

y  (1 target — the back of the card)
└── MedHouseVal
```

We don't have to use every feature. In fact, our first model will start small, with three easy-to-understand features:

```text
MedInc
HouseAge
AveRooms
        ↓
        X

MedHouseVal
        ↓
        y
```

Starting small lets us ask a clear question later: *"Can income, house age and rooms explain some of the variation in house value?"* We can then compare against all 8 features and see whether the extra information actually helps.

---

## 5. A tiny example

Here are the same three made-up areas from chapter 00, split into `X` and `y`:

```text
             X (information)                     y (answer)
        ┌─────────┬──────────┬──────────┐      ┌─────────────┐
        │ MedInc  │ HouseAge │ AveRooms │      │ MedHouseVal │
Area A  │   5.0   │    20    │    5     │      │    2.5      │  ($250k)
Area B  │   8.0   │    10    │    7     │      │    4.5      │  ($450k)
Area C  │   6.5   │    15    │    6     │      │     ?       │  ← we want to predict this
        └─────────┴──────────┴──────────┘      └─────────────┘
```

Two things to notice:

1. **Each row of `X` has exactly one matching value in `y`.** Row A's answer is 2.5, row B's is 4.5. If we shuffled `y` without shuffling `X`, every answer would be attached to the wrong question.
2. **For Area C we have `X` but not `y`.** That's the situation the model is built for.

---

## 6. The official words

| Plain English | Official term |
|---|---|
| A piece of information given to the model | **Feature** (also "input", "predictor", "independent variable") |
| All the features for all rows together | **`X`**, the **feature matrix** |
| The value we want to predict | **Target** (also "label", "output", "dependent variable") |
| All the target values together | **`y`**, the **target vector** |
| Accidentally giving the model information it wouldn't really have when predicting | **Data leakage** |

Why a capital **X** and a lowercase **y**? It's a maths habit: capital letters for tables (many columns), lowercase for a single column. It's just a naming convention, but you'll see it in almost all ML code.

---

## 7. A little Python

scikit-learn already separates them for us:

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)

X = housing.data
y = housing.target

print(X.shape)
print(y.shape)
```

We can do the same thing ourselves from the full table:

```python
df = housing.frame

X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]
```

And to start with only three features:

```python
features = ["MedInc", "HouseAge", "AveRooms"]

X = df[features]
y = df["MedHouseVal"]
```

---

## 8. The Python, line by line

### `X = housing.data`

A DataFrame with the 8 feature columns. Shape: **20,640 rows × 8 columns**.

### `y = housing.target`

A pandas **Series** (a single column with an index) containing `MedHouseVal`. Shape: **20,640 values**.

### `print(X.shape)` → `(20640, 8)` and `print(y.shape)` → `(20640,)`

```text
X                                   y

20,640 rows                         20,640 answers
×
8 features
┌───┬───┬───┬───┬───┬───┬───┬───┐   ┌───┐
│   │   │   │   │   │   │   │   │   │   │   row 0  → answer for row 0
│   │   │   │   │   │   │   │   │   │   │   row 1  → answer for row 1
│ … │ … │ … │ … │ … │ … │ … │ … │   │ … │   …
│   │   │   │   │   │   │   │   │   │   │   row 20,639
└───┴───┴───┴───┴───┴───┴───┴───┘   └───┘
```

`(20640,)` with a trailing comma means "a one-dimensional thing with 20,640 values": one answer per row. **`X` and `y` must have the same number of rows**, because every example needs exactly one answer. If they don't match, scikit-learn will refuse to train.

### `df.drop(columns=["MedHouseVal"])`

Returns a **new** DataFrame with every column except `MedHouseVal`. It doesn't change `df` itself.

### `df["MedHouseVal"]` vs `df[["MedHouseVal"]]`

This trips up almost everyone:

| Code | Returns | Shape |
|---|---|---|
| `df["MedHouseVal"]` (one pair of brackets, a string) | a **Series**: one column | `(20640,)` |
| `df[["MedHouseVal"]]` (two pairs of brackets, a list) | a **DataFrame** with one column | `(20640, 1)` |
| `df[["MedInc", "HouseAge", "AveRooms"]]` | a DataFrame with 3 columns | `(20640, 3)` |

Rule of thumb: **`y` is a Series** (single brackets). **`X` is a DataFrame** (a list of column names, so double brackets), even when it has only one feature.

---

## 9. What happens inside (high level)

When we later call `model.fit(X, y)`, scikit-learn lines up row *i* of `X` with value *i* of `y`, for every *i*. The model then adjusts itself so that, for each row, the guess it would make from `X` gets as close as possible to `y`.

That's why the row alignment matters so much: the model learns "**this** information goes with **this** answer".

### Why the target can't be an input

What if we accidentally left `MedHouseVal` inside `X`?

```text
X = [MedInc, HouseAge, ..., MedHouseVal]      y = MedHouseVal
```

The model would discover the easiest possible rule: *"copy the MedHouseVal column"*. It would score perfectly in testing.

And it would be **completely useless**. In real life, when we ask "what's this area worth?", we don't *have* `MedHouseVal`. If we had it, we wouldn't need a model. It's like a quiz where the answer is printed on the front of the card: you get 100% and learn nothing.

### Data leakage: a first look

That example is the most extreme form of **data leakage**: the model gets information during training that it **would not have at prediction time**.

Leakage is usually sneakier than copying the target. Imagine a housing dataset with these columns:

| Hypothetical feature | Leaks the target? | Why |
|---|---|---|
| `final_sale_price` | **Yes, directly** | It *is* the answer. |
| `price_per_square_foot` (computed from the sale price) | **Yes, indirectly** | Multiply it by the size and you get the price back. |
| `tax_assessment_after_sale` | **Yes, indirectly** | It's based on the sale price and only exists *after* the sale. |
| `number_of_bedrooms` | No | Known before the sale. |

The golden rule:

> **A feature may only contain information that would actually be available at the moment we make the prediction.**

California Housing is a clean teaching dataset, so none of its 8 features obviously leak. But you should ask this question about **every** feature in **every** future project.

---

## 10. Common beginner mistakes

- **Confusing features and target.** `y` is the *one* thing you predict. Everything the model may look at goes in `X`.
- **Leaving the target inside `X`.** Always check: `"MedHouseVal" in X.columns` should be `False`.
- **Using single brackets for `X`.** `df["MedInc"]` is a Series, but models expect `X` to be 2-D (a table). Use `df[["MedInc"]]`.
- **Breaking the row alignment.** Sorting, filtering or shuffling `X` without doing exactly the same to `y` attaches answers to the wrong rows. Do row operations on `df` *before* splitting into `X` and `y`.
- **Assuming more features is always better.** Extra features can add noise, or leak. We'll *test* whether they help instead of assuming.
- **Using information from the future.** If a value only becomes known after the thing you're predicting, it doesn't belong in `X`.

---

## 11. Exercises

Use the notebook (a new cell at the end is fine).

**Exercise 1 — Predict the shapes.**
Before running anything, write down what you expect `X.shape` and `y.shape` to be for (a) all 8 features and (b) the 3-feature version. Then check.

**Exercise 2 — Build `X` and `y` yourself.**
Starting from `df`, write the code that creates `X` with only `MedInc`, `HouseAge` and `AveRooms`, and `y` with `MedHouseVal`. Print the first 3 rows of each.

<details>
<summary>Hint</summary>

A list of column names inside the brackets gives you a DataFrame. `.head(3)` shows the first three rows.
</details>

**Exercise 3 — Prove the alignment.**
Show that row 0 of your `X` and row 0 of your `y` belong to the same block group as row 0 of `df`.

<details>
<summary>Hint</summary>

`X.iloc[0]`, `y.iloc[0]` and `df.iloc[0]` each give the first row. Compare them.
</details>

**Exercise 4 — Spot the leak.**
You're predicting whether a patient will be readmitted to hospital within 30 days. Which of these features would be leakage? `age`, `number_of_previous_visits`, `discharge_date`, `readmission_date`, `diagnosis_at_admission`.

<details>
<summary>Hint</summary>

The prediction is made on the day the patient leaves hospital. For each feature, ask: *would I already know this value on that day?*
</details>

**Exercise 5 — Single vs double brackets.**
Run `type(df["MedInc"])` and `type(df[["MedInc"]])`. Explain in one sentence why they differ.

---

## 12. Before continuing, I should be able to explain:

- □ What is a feature? Give two examples from this dataset.
- □ What is the target in this project, and what are its units?
- □ Why can't `MedHouseVal` be included inside `X`?
- □ Why do `X` and `y` have the same number of rows?
- □ What is data leakage, in one sentence, and what question should I ask about every feature?
- □ Why is `X` usually a DataFrame and `y` a Series?

> **🛑 Stop point.** Chapters 00–02 plus the exploration notebook are the **current learning phase**. Before continuing to chapter 03 (the first chapter that leads to model training), make sure you can answer every checkpoint above *in your own words*. The project code intentionally stops here too.

---

**Previous:** [← 01 — Understanding the Data](01-understanding-the-data.md) · **Next:** [03 — Training and Testing →](03-training-and-testing.md)
