# 03 — Training and Testing: Practice Questions and the Exam

> **Course map:** [00 Big picture](00-ml-big-picture.md) → [01 Data](01-understanding-the-data.md) → [02 Features & target](02-features-and-target.md) → **03 Training & testing** → [04 Baseline](04-first-baseline-model.md) → [05 Linear regression](05-linear-regression.md) → [06 Evaluation](06-evaluating-a-model.md) → [07 Overfitting](07-overfitting-and-generalization.md) → [08 Random forest](08-random-forest.md) → [09 Workflow](09-ml-workflow.md) · [Glossary](glossary.md)

> 📍 **Roadmap chapter.** The project code doesn't do this yet. Read it now for orientation, or when you reach this milestone. Do the exercises when you get here in the project.

| Where we are | |
|---|---|
| **Earlier** | We separated the **columns** into `X` (information) and `y` (answer) ([chapter 02](02-features-and-target.md)). |
| **Now** | Split the **rows** into a part the model learns from and a part we keep hidden to test it. |
| **Next** | Use the training part to build our very first "model": a deliberately dumb one ([chapter 04](04-first-baseline-model.md)). |

---

## 1. What problem are we trying to solve?

We want a model that works on **neighbourhoods it has never seen**. That's the only situation where a prediction is useful. If we already knew the answer, we wouldn't need to predict it.

So how do we find out whether the model works on unseen data **before** using it for real? We need some data where we *know* the answers but the model *doesn't*.

---

## 2. The idea in plain English

Before learning starts, we take our 20,640 rows and put about a fifth of them aside, in a locked drawer.

- The model **learns** only from the remaining four fifths.
- When learning is finished, we open the drawer, ask the model to predict the house values for those hidden rows, and compare its guesses with the real answers.

The hidden rows act like **new neighbourhoods**. How well the model does on them is our best estimate of how well it will do in the real world.

### What does "training" actually mean?

People often say "we fit the model" as if that explains it. Here's what actually happens:

1. The model starts with some rule, e.g. "value = a + b × income", where `a` and `b` are **not yet known**.
2. It looks at the training examples: for each one it knows the features (`X`) *and* the correct answer (`y`).
3. It measures how far its current guesses are from the correct answers.
4. It adjusts `a` and `b` to make those errors smaller.
5. It stops when the errors are as small as it can make them.

The result is a rule with **specific numbers filled in**. That rule is the trained model. Training is **searching for the settings that make the guesses match the known answers as closely as possible**.

---

## 3. An everyday analogy

Think about preparing for an exam:

```text
Practice questions (with answer key)
      ↓
Training: you study, check your answers, adjust your understanding

Questions never seen before (answers hidden from you)
      ↓
Test: shows whether you actually learned the subject
```

Now imagine the teacher sets **the exact practice questions** as the exam. A student who memorised the answer key scores 100% without understanding anything. The score says nothing about whether they could solve a *new* question.

Evaluating a model on its own training data is exactly the same mistake.

---

## 4. Back to house prices

```text
20,640 neighbourhoods
        │
        ├───────────────────┐
        │                   │
       80%                 20%
   16,512 rows          4,128 rows
        │                   │
        ▼                   ▼
   TRAINING SET          TEST SET
   (X_train, y_train)    (X_test, y_test)
        │                   │
        ▼                   │
   model learns here        │   ← locked away during training
                            ▼
                     model predicts here
                            │
                            ▼
                  compare predictions with y_test
```

The model never sees `y_test` while learning. That's what makes the test honest.

---

## 5. A tiny example

Say we have 10 neighbourhoods (made-up values in $100k):

```text
Area:     A    B    C    D    E    F    G    H    I    J
Value:   1.2  2.5  3.1  0.9  4.0  1.8  2.2  3.5  1.5  2.9
```

An 80/20 split *randomly* picks 2 areas for the test set, for example **C** and **H**:

```text
Training set (8):  A  B  D  E  F  G  I  J   → the model learns from these
Test set     (2):  C  H                     → hidden, used only for evaluation
```

After training, the model predicts C and H. Suppose it predicts **2.8** for C (real: 3.1) and **3.9** for H (real: 3.5). The errors are 0.3 and 0.4, i.e. **$30,000 and $40,000**. That's our estimate of how wrong it will be on new areas.

Why random? If we took "the last 20%" and the data happened to be sorted (California Housing's first rows are all from the Bay Area!), the test set might contain only one region, and the result would be misleading.

---

## 6. The official words

| Plain English | Official term |
|---|---|
| Rows the model learns from | **Training set** (`X_train`, `y_train`) |
| Rows hidden from the model and used to check it | **Test set** (`X_test`, `y_test`), also "hold-out set" |
| Learning the model's settings from the training set | **Training** / **fitting** |
| Doing well on data the model never saw | **Generalization** |
| A third set, used to *choose between* models or settings before the final test | **Validation set** (more on this in [chapter 09](09-ml-workflow.md)) |

We want the model to work on neighbourhoods it did not see during training. This ability is called **generalization**, and the test set is how we *measure* it.

---

## 7. A little Python

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

df = fetch_california_housing(as_frame=True).frame

features = ["MedInc", "HouseAge", "AveRooms"]
X = df[features]
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
```

Output:

```text
(16512, 3) (16512,)
(4128, 3) (4128,)
```

---

## 8. The Python, line by line

### `from sklearn.model_selection import train_test_split`

Imports scikit-learn's helper for splitting data. "Model selection" is the part of scikit-learn about evaluating and comparing models.

### `X_train, X_test, y_train, y_test = train_test_split(X, y, ...)`

The function returns **four** things, always in this order:

| Variable | Contains | Shape here |
|---|---|---|
| `X_train` | features of the training rows | 16,512 × 3 |
| `X_test` | features of the test rows | 4,128 × 3 |
| `y_train` | answers for the training rows | 16,512 |
| `y_test` | answers for the test rows | 4,128 |

The order is **X, X, y, y**, not X, y, X, y. Mixing it up is a classic bug: Python won't complain, you'll just be training on the wrong things.

`train_test_split` keeps rows **aligned**: if row 17 goes to the test set, both its features (in `X_test`) and its answer (in `y_test`) go there together.

### `test_size=0.2`

Put 20% of rows into the test set, and the remaining 80% into the training set. 20% of 20,640 = **4,128** test rows. 80/20 is a common default: enough training data to learn from, and enough test data for a trustworthy score.

### `random_state=42`

Before splitting, the function **shuffles** the rows randomly. That's good (see section 5), but it means every run would give a *different* split, and different scores. That makes it impossible to compare experiments fairly.

`random_state` fixes the shuffle. It's like **shuffling a deck of cards in exactly the same way every time**: same number → same shuffle → same split. The number itself doesn't matter (42 is just a popular joke from *The Hitchhiker's Guide to the Galaxy*). What matters is using **the same value** whenever you want comparable results.

This property is called **reproducibility**: someone else running your code gets exactly your numbers.

---

## 9. What happens inside (high level)

`train_test_split`:

1. creates a list of row positions `[0, 1, 2, …, 20639]`;
2. shuffles that list using a random generator seeded with `random_state`;
3. takes the first 4,128 shuffled positions as the test set and the rest as the training set;
4. selects those rows from `X` **and** `y`.

No learning happens here. It's pure bookkeeping, and that's exactly why it's so important to do *first*, before anything learns from the data.

### Why testing on training data is misleading

A model's error on its training data tells you how well it **fits the examples it has seen**. A model can shrink that number simply by **memorising**. The error on the test set tells you how well it **generalises**. That's the number that predicts real-world performance. We'll see this dramatically in [chapter 07](07-overfitting-and-generalization.md).

---

## 10. Common beginner mistakes

- **Evaluating on the training data.** It produces flattering numbers that don't reflect real-world performance.
- **Getting the return order wrong.** It's `X_train, X_test, y_train, y_test`.
- **Splitting `X` and `y` separately** (two different calls). The rows would no longer match up. Always pass both to the same call.
- **Forgetting `random_state`,** then being confused when results change every run.
- **Comparing models on different splits.** If model A is tested on one test set and model B on another, the comparison isn't fair. Use the same `random_state` (the same test set) for every model you compare.
- **Peeking at the test set to make decisions.** If you keep adjusting your model until the *test* score looks good, the test set is no longer "unseen". You've trained on it indirectly. Save the test set for the final check (see *validation set*, [chapter 09](09-ml-workflow.md)).
- **Doing data preparation on all the data before splitting.** Anything that "learns" from data (e.g. computing an average to fill in missing values) must learn from the training set only. Otherwise information from the test set leaks in.

---

## 11. Exercises

**Exercise 1 — Predict the shapes.**
With `test_size=0.25`, what shapes will `X_train` and `X_test` have (using the 3 features)? Work it out on paper, then run the code.

<details>
<summary>Hint</summary>

25% of 20,640 = ? If the result isn't a whole number, scikit-learn rounds the test size **up**.
</details>

**Exercise 2 — See `random_state` in action.**
Run the split twice with `random_state=42` and print `y_test.head()` both times. Then use `random_state=0`. What changes and what stays the same?

**Exercise 3 — Different splits, different scores.**
(Do this after chapter 04.) Train the same model with `random_state=0`, `1` and `2`, and compare the test scores. What does this tell you about trusting a single split?

**Exercise 4 — Explain it.**
In two sentences, explain to a non-programmer why a model must be tested on data it didn't learn from.

---

## 12. Before continuing, I should be able to explain:

- □ What does "training" mean, beyond "calling `fit()`"?
- □ Why is evaluating a model on its training data misleading?
- □ What are `X_train`, `X_test`, `y_train` and `y_test`, and what are their shapes here?
- □ What does `test_size=0.2` do?
- □ What does `random_state` do, and why does it matter when comparing experiments?
- □ What is generalization, and how does the test set measure it?

---

**Previous:** [← 02 — Features and Target](02-features-and-target.md) · **Next:** [04 — Your First Baseline Model →](04-first-baseline-model.md)
