# 00 — The Big Picture: What Is Machine Learning?

> **Course map:** **00 Big picture** → [01 Data](01-understanding-the-data.md) → [02 Features & target](02-features-and-target.md) → [03 Training & testing](03-training-and-testing.md) → [04 Baseline](04-first-baseline-model.md) → [05 Linear regression](05-linear-regression.md) → [06 Evaluation](06-evaluating-a-model.md) → [07 Overfitting](07-overfitting-and-generalization.md) → [08 Random forest](08-random-forest.md) → [09 Workflow](09-ml-workflow.md) · [Glossary](glossary.md)

| Where we are | |
|---|---|
| **Earlier** | Nothing yet — this is the start. |
| **Now** | Understand what machine learning *is*, before touching any code. |
| **Next** | Look at the real California Housing data ([chapter 01](01-understanding-the-data.md)). |

By the end of this chapter you should be able to answer, in your own words:

> *What is happening when I give data to a machine-learning algorithm, train it, and ask it to predict something it has never seen before?*

You will not fully answer it yet. But you'll have a rough map of the whole course, and every later chapter will fill in one part of it.

---

## 1. What problem are we trying to solve?

Imagine a friend says:

> "There's a neighbourhood in California. People there earn fairly good incomes, the houses are about 20 years old and have about 5 rooms on average. What do you think a typical house there is worth?"

You'd probably make a guess. How? You don't have a formula in your head. You've **seen examples** before: richer areas tend to have more expensive houses, bigger houses tend to cost more, and so on. You use those patterns to make a reasonable guess about a place you've never seen.

Machine learning is about getting a computer to do the same thing:

1. look at many examples where we already know the answer;
2. discover useful patterns in them;
3. use those patterns to make a guess about a **new** case.

In this project the question is:

> Given information about a California neighbourhood, what is the typical (median) house value there?

---

## 2. The idea in plain English

### Normal programming

You already know normal programming. **You** write the rules:

```text
Rules written by programmer
        +
Input
        ↓
Program
        ↓
Answer
```

For example, a tax calculator: the programmer knows the tax rules and writes them down as code.

```python
def tax(income):
    if income < 10_000:
        return 0
    return (income - 10_000) * 0.2
```

This works because a human **knows the rules**.

### When nobody can write the rules

Now try writing the rules for "what is a house worth?".

```text
def house_value(income, house_age, rooms, location, ...):
    if income > 5 and rooms > 6 and ...:
        return ???      ← what number goes here? and for every other case?
```

You would get stuck quickly. The real rules are messy, they interact with each other, and nobody knows them exactly. But we *do* have something else: **thousands of examples where we know the answer.**

### Supervised machine learning

So we flip the process around. Instead of writing the rules, we give the computer examples **and** the correct answers, and let an algorithm work out the rules:

```text
Examples
+
Correct answers
        ↓
Training algorithm
        ↓
Model
```

Later, once the model exists, we use it on new cases:

```text
New example
     ↓
trained model
     ↓
prediction
```

That's the whole idea. Most of this course is about doing these two steps **carefully and honestly**.

---

## 3. An everyday analogy

Think of a new estate agent in their first year on the job.

- **Before:** they know nothing about local prices.
- **Learning:** they watch hundreds of houses being sold. For each one, they see the details (area, size, age) *and* the final price.
- **What they learn:** not a list of prices, but *rules of thumb*: "In this area a house costs roughly this much", "each extra bedroom adds about that much", "old houses near the coast still sell high".
- **Using it:** a new house comes on the market. The agent has never seen *this* house, but they can estimate its price from their rules of thumb.
- **Being checked:** later the house sells and we can see how close their estimate was.

The estate agent is doing machine learning in their head.

---

## 4. Back to house prices

In this project:

| Estate agent | Our project |
|---|---|
| Watching past sales | Looking at 20,640 rows of California data |
| Details of each house | Income, house age, rooms, location, … |
| Final sale price | `MedHouseVal`, the median house value |
| Rules of thumb in their head | The **model** |
| Estimating a new house | A **prediction** |
| Checking against the real sale price | **Evaluation** |

One important detail, which [chapter 01](01-understanding-the-data.md) explains: each row in our data describes a **small neighbourhood** (a census block group), not a single house. So strictly we predict *the typical house value in an area*. We'll still call it "house price prediction" for short.

---

## 5. A tiny example

Here are two made-up neighbourhoods:

```text
Area A:
income = 5        (median income, in $10,000s → $50,000)
age    = 20       (years)
rooms  = 5        (average rooms per home)
value  = $250k    ← the answer

Area B:
income = 8        (→ $80,000)
age    = 10
rooms  = 7
value  = $450k    ← the answer
```

Even from two examples, a pattern seems to appear: **higher income and more rooms go with higher value.**

Now a new area arrives:

```text
Area C:
income = 6.5
age    = 15
rooms  = 6
value  = ???
```

Most people would guess somewhere between $250k and $450k, maybe about $350k. You just did something very close to what a model does:

- you looked at examples with known answers;
- you noticed a pattern;
- you applied the pattern to a new case.

With two examples your guess is shaky. With 20,640 examples, a computer can find patterns much more reliably. It also finds them *consistently*, so we can measure exactly how good it is.

---

## 6. The official words

Now that you have the idea, here are the names for it. (All of them are also in the [glossary](glossary.md).)

| Plain English | Official term |
|---|---|
| Learning from examples that come with correct answers | **Supervised learning** |
| The answer we want to predict (house value) | **Target** (called `y`) |
| The information we use to predict it (income, age, …) | **Features** (called `X`) |
| One example (one neighbourhood) | **Sample** or **observation** |
| The process of finding patterns in the examples | **Training** |
| The recipe that finds the patterns (e.g. "fit a straight line") | **Algorithm** |
| The result of training, used to make guesses | **Model** |
| A guess the model makes | **Prediction** |
| Using a trained model on new data | **Inference** |
| Predicting a *number* (like a price) | **Regression** |
| Predicting a *category* (like "spam / not spam") | **Classification** |

Our project is **supervised regression**:

- *supervised*, because every training example comes with the correct answer;
- *regression*, because the answer is a number on a continuous scale ($123,400, $312,000, …), not a category.

---

## 7. A little Python

You won't write this code yet. Here's a preview of the pattern we'll use for most of the course, so you can see that it follows the same two steps as the diagrams above:

```python
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

housing = fetch_california_housing(as_frame=True)
X = housing.data      # the information about each neighbourhood
y = housing.target    # the correct answer for each neighbourhood

model = LinearRegression()   # choose an algorithm
model.fit(X, y)              # training: learn patterns from examples + answers

new_area = X.head(1)             # pretend this is a new neighbourhood
print(model.predict(new_area))   # inference: make a prediction
```

(This preview skips an important step. It trains and predicts on the *same* data, which is exactly the thing [chapter 03](03-training-and-testing.md) teaches you **not** to do. We'll fix it there.)

---

## 8. The Python, line by line

| Line | What it means |
|---|---|
| `fetch_california_housing(as_frame=True)` | Download (once) and load the dataset as pandas tables. |
| `X = housing.data` | The **features**: 8 columns of information about each neighbourhood. |
| `y = housing.target` | The **target**: one column with the correct answer for each neighbourhood. |
| `model = LinearRegression()` | Pick an algorithm. At this point the model has learned nothing. It's an empty recipe. |
| `model.fit(X, y)` | **Training.** Show the algorithm the examples and the answers, and let it adjust itself until its guesses on these examples are as close to the answers as it can get. |
| `model.predict(new_area)` | **Inference.** Apply what was learned to feature values and get back a guessed house value. |

`fit` means "learn from these examples". `predict` means "use what you learned". Almost every scikit-learn model follows this same two-step pattern.

---

## 9. What happens inside (high level)

So what *is* a model? This is one of the most important ideas in the course.

**A model is a small set of learned numbers plus a fixed way of combining them to turn inputs into a guess.**

For example, a trained model might boil down to something like this (made-up numbers):

```text
value ≈ 0.5 + 0.4 × income
```

- The **form** of the rule ("a starting number plus something times income") comes from the algorithm you chose.
- The **numbers** (0.5 and 0.4) are what training discovered from the data.

Training searches for the numbers that make the rule's guesses closest to the known answers. Once it has found them, the training examples are no longer needed to make predictions.

So a model is **not**:

- ❌ **a database of answers.** It doesn't look up "this row → $250k". It keeps a *compressed pattern* (a few numbers or a set of questions), not the examples themselves. That's why it can give an answer for an area it has never seen.
- ❌ **magic.** Everything it does is arithmetic you could, in principle, do by hand with a calculator.
- ❌ **"AI" in the science-fiction sense.** It doesn't understand what a house is. It found a relationship between some columns of numbers and another column of numbers. If the data is misleading, the model will be misleading too.

A model **is**:

- ✅ a **pattern learned from examples** that you can use on new cases;
- ✅ usually **approximately right**, never perfectly right;
- ✅ only as good as the data and the evaluation behind it.

---

## 10. Common beginner mistakes

- **"The model memorises the answers."** A good model learns *general patterns*. A model that memorises its examples is actually a *bad* model. We call this overfitting ([chapter 07](07-overfitting-and-generalization.md)).
- **"If the code runs, the model works."** `model.fit()` almost never crashes, even when the model is useless. Only honest evaluation on data the model hasn't seen tells you if it works ([chapters 03](03-training-and-testing.md) and [06](06-evaluating-a-model.md)).
- **"Machine learning finds the true cause of things."** It finds *patterns that help predict*. Income predicting house value doesn't mean we've proven income *causes* high value.
- **"More complicated algorithm = better."** Often a simple model plus a good understanding of the data beats a complicated model used blindly. That's why we start with the simplest possible "model" in [chapter 04](04-first-baseline-model.md).
- **"Accuracy should be 100%."** Real-world predictions always have errors. The question is *how big* the errors are and whether they're small enough to be useful.

---

## 11. Exercises

These are thinking exercises. There's no code yet.

**Exercise 1 — Rules or examples?**
For each problem, decide whether normal programming (writing rules) or machine learning (learning from examples) is the better fit, and why:

1. Converting Celsius to Fahrenheit.
2. Guessing whether an email is spam.
3. Sorting a list of names alphabetically.
4. Estimating how long a taxi ride will take.

<details>
<summary>Hint</summary>

Ask yourself: *can a human write down the exact rule?* If yes, just write the rule. If the rule is fuzzy but you have lots of past examples with known answers, ML may help.
</details>

**Exercise 2 — Regression or classification?**
Which of these predict a *number* and which predict a *category*?

1. Tomorrow's temperature.
2. Whether a customer will cancel their subscription (yes/no).
3. The price of a used car.
4. Which of 10 animals is in a photo.

**Exercise 3 — Explain it to a friend.**
Write 3–4 sentences, with no technical words, explaining what our project will do. Pretend you're explaining it to someone who has never programmed.

---

## 12. Before continuing, I should be able to explain:

- □ What is the difference between normal programming and supervised machine learning?
- □ What is a model, and why is it *not* a database of answers?
- □ What does training do, in one sentence?
- □ What does prediction (inference) do, in one sentence?
- □ Why is our project "regression" and not "classification"?
- □ Why does each row in our data describe a neighbourhood rather than a single house? (You'll confirm this in chapter 01.)

---

**Next:** [01 — Understanding the Data →](01-understanding-the-data.md)

Before reading chapter 01, open [`notebooks/01_exploration.ipynb`](../notebooks/01_exploration.ipynb) and run it cell by cell. Then read chapter 01 to understand what you saw.
