# 01 — Understanding the Data

> **Course map:** [00 Big picture](00-ml-big-picture.md) → **01 Data** → [02 Features & target](02-features-and-target.md) → [03 Training & testing](03-training-and-testing.md) → [04 Baseline](04-first-baseline-model.md) → [05 Linear regression](05-linear-regression.md) → [06 Evaluation](06-evaluating-a-model.md) → [07 Overfitting](07-overfitting-and-generalization.md) → [08 Random forest](08-random-forest.md) → [09 Workflow](09-ml-workflow.md) · [Glossary](glossary.md)

| Where we are | |
|---|---|
| **Earlier** | ML means learning patterns from examples that come with correct answers ([chapter 00](00-ml-big-picture.md)). |
| **Now** | Look carefully at the examples we'll learn from: the California Housing dataset. |
| **Next** | Decide which columns are the *information* (`X`) and which is the *answer* (`y`) ([chapter 02](02-features-and-target.md)). |

**Companion notebook:** [`notebooks/01_exploration.ipynb`](../notebooks/01_exploration.ipynb) runs every command in this chapter on the real data.

---

## 1. What problem are we trying to solve?

A model can only learn from the data we give it. If we don't understand that data, we can't:

- tell whether the model's predictions make sense;
- notice when the data contains something strange (typos, caps, impossible values);
- choose sensible inputs for the model;
- explain what the model's answers even *mean*.

Suppose we don't know that `MedHouseVal = 2.5` means **$250,000**. We might think the model is predicting "$2.50" and conclude that something is badly broken. Or we might report errors in the wrong units to someone who relies on them.

So before any training, the first milestone is:

> **I can explain every column, and what one row of the dataset represents.**

---

## 2. The idea in plain English

A **dataset** is a table of examples, like a spreadsheet.

- Each **row** is one example: one thing we observed.
- Each **column** is one kind of information we recorded about every example.
- Each **cell** is one value: *this* piece of information about *this* example.

In California Housing, every value is a **number**. Some numbers are measured on a smooth scale (income, latitude). Others are counts (population), but pandas treats them all simply as numbers.

Looking at a dataset means asking simple questions:

- How many examples are there?
- What columns exist, and what do they mean?
- What are typical values? What are the smallest and largest?
- Is anything missing?
- Does anything look *suspicious*?

---

## 3. An everyday analogy

Imagine you start a new job and inherit a huge spreadsheet from a colleague who has left. Before using it for anything important, you would:

1. scroll through the first few rows to get a feel for it;
2. check how big it is;
3. read the column headers and work out what they mean;
4. check whether any cells are empty;
5. look at the smallest and largest numbers in each column ("wait, someone is 250 years old?").

That's exactly what we'll do. The pandas commands in this chapter are just quick ways to do those five things.

---

## 4. Back to house prices

### What does one row represent?

This is the single most important fact about this dataset:

> **Each row is one California *census block group*, not one individual house.**

A **census block group** is a small geographic area that the US Census Bureau uses to publish statistics. It's roughly a neighbourhood, typically with a population of about 600 to 3,000 people. The data comes from the **1990 US Census**.

That's why many columns are *averages* or *medians*: they summarise all the households in that area.

```text
One row  =  one neighbourhood (block group)
         ≠  one house
```

So when the model predicts `MedHouseVal`, it predicts the **median house value in that neighbourhood**, not the price of a specific house.

### What does each column mean?

| Column | Meaning | Unit / scale | Example value (row 0) |
|---|---|---|---|
| `MedInc` | Median household income in the block group | tens of thousands of US dollars (3.5 ≈ $35,000) | 8.3252 (≈ $83,000) |
| `HouseAge` | Median age of the houses in the block group | years | 41 |
| `AveRooms` | Average number of rooms per household | rooms | 6.98 |
| `AveBedrms` | Average number of bedrooms per household | bedrooms | 1.02 |
| `Population` | Number of people living in the block group | people | 322 |
| `AveOccup` | Average number of people per household | people | 2.56 |
| `Latitude` | How far north the block group is | degrees (higher = further north) | 37.88 |
| `Longitude` | How far west the block group is | degrees (more negative = further west) | −122.23 |
| `MedHouseVal` | **Median house value** in the block group, **the target** | **hundreds of thousands of US dollars** | 4.526 (≈ $452,600) |

Two unit conversions to memorise:

```text
MedInc      = 3.5    →  3.5 × $10,000   = $35,000   median income
MedHouseVal = 2.5    →  2.5 × $100,000  = $250,000  median house value
```

(Row 0, at latitude 37.88 and longitude −122.23, is in the San Francisco Bay Area, near Berkeley.)

---

## 5. A tiny example

Here are the first three rows of the real dataset (numbers rounded):

```text
      MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
row 0   8.33        41      6.98       1.02         322      2.56     37.88    -122.23        4.526
row 1   8.30        21      6.24       0.97        2401      2.11     37.86    -122.22        3.585
row 2   7.26        52      8.29       1.07         496      2.80     37.85    -122.24        3.521
```

Read row 1 aloud as a sentence:

> "In this neighbourhood, 2,401 people live in households with a median income of about $83,000. The houses are about 21 years old on average (median) and have about 6.2 rooms and 1 bedroom per household, with about 2.1 people per household. It sits at latitude 37.86, longitude −122.22. The median house value there is about $358,500."

If you can turn any row into a sentence like that, you understand the data.

---

## 6. The official words

| Plain English | Official term |
|---|---|
| The whole table of examples | **Dataset** |
| One row, i.e. one thing we observed | **Observation** or **sample** (also "example", "record", "instance") |
| One column, i.e. one kind of information | **Column**, **variable**, or (when used as model input) **feature** |
| A value that's a number on a scale | **Numerical value** |
| How the values of one column are spread out: which values are common, which are rare | **Distribution** |

Some care with the word **sample**: in ML it usually means *one row*. In statistics it can also mean *a group of rows* taken from a bigger population. In this course, "sample" means one row.

---

## 7. A little Python

These are the commands you run in the notebook:

```python
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing(as_frame=True)
df = housing.frame

df.head()
df.shape
df.columns
df.info()
df.describe()
df.isna().sum()
```

---

## 8. The Python, line by line

### `housing = fetch_california_housing(as_frame=True)`

Downloads the dataset the first time (about 400 KB) and loads it. In this project's Docker setup the download is kept in the `sklearn-data` Docker volume, so it only happens once. `as_frame=True` asks for pandas tables instead of plain NumPy arrays, so we get column names.

`housing` is a *Bunch*: a scikit-learn container that works like a dictionary. It holds several things:

- `housing.frame`: the whole table (features + target), a pandas `DataFrame`;
- `housing.data`: only the 8 feature columns, a `DataFrame`;
- `housing.target`: only the target column, a pandas `Series`;
- `housing.DESCR`: a long text description of the dataset. Try `print(housing.DESCR)`.

In this project, `house_price.data.load_housing_data()` does this step for you and returns `housing.frame`.

### `df = housing.frame`

A **DataFrame** is pandas' table: rows, named columns, and a row label (the **index**, 0, 1, 2, …) on the left.

### `df.head()`

Shows the **first 5 rows**. It answers: *"What does the data look like?"* Use `df.head(10)` for 10 rows, or `df.sample(5)` for 5 random rows. Random rows are useful because the first rows are all from the same area (the Bay Area).

### `df.shape`

Returns `(20640, 9)`: **(number of rows, number of columns)**. It answers: *"How big is the data?"* Note that there are no parentheses: `shape` is an attribute, not a method.

```text
20,640 rows  ×  9 columns
(block groups)  (8 features + 1 target)
```

### `df.columns`

Lists the column names. It answers: *"What information do I have?"*

### `df.info()`

Prints one line per column: its name, how many values are **non-null** (not missing), and its **dtype** (data type). It answers: *"Is anything missing, and is every column the type I expect?"*

Here every column shows `20640 non-null` and `float64`:

- `20640 non-null` means no missing values;
- `float64` means a decimal number.

If a column you expect to be numeric showed `object`, that would usually mean some values are text. That's a sign of dirty data.

### `df.describe()`

Summarises each numeric column with 8 numbers. It answers: *"What values are typical, and what are the extremes?"*

| Row of `describe()` | Plain English |
|---|---|
| `count` | How many non-missing values |
| `mean` | The average |
| `std` | "Standard deviation": roughly, *how far a typical value is from the average*. Small = values are bunched together; large = values are spread out |
| `min` | Smallest value |
| `25%` | A quarter of the values are below this |
| `50%` | Half the values are below this (the **median**, the middle value) |
| `75%` | Three quarters of the values are below this |
| `max` | Largest value |

Here is the real output, rounded:

```text
            MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal
count     20640     20640     20640      20640      20640     20640     20640      20640        20640
mean       3.87     28.64      5.43       1.10     1425.48      3.07     35.63    -119.57         2.07
std        1.90     12.59      2.47       0.47     1132.46     10.39      2.14       2.00         1.15
min        0.50      1.00      0.85       0.33        3.00      0.69     32.54    -124.35         0.15
25%        2.56     18.00      4.44       1.01      787.00      2.43     33.93    -121.80         1.20
50%        3.53     29.00      5.23       1.05     1166.00      2.82     34.26    -118.49         1.80
75%        4.74     37.00      6.05       1.10     1725.00      3.28     37.71    -118.01         2.65
max       15.00     52.00    141.91      34.07    35682.00   1243.33     41.95    -114.31         5.00
```

How to read it, using `MedHouseVal`:

- the typical (median) block group has a median house value of about 1.80 → **$180,000**;
- the cheapest is 0.15 → **$15,000**, and the most expensive is 5.00 → **$500,000**;
- the mean (2.07) is higher than the median (1.80). That usually means a "tail" of high values pulls the average up.

### `df.isna().sum()`

`df.isna()` builds a table of `True`/`False` values: `True` wherever a value is missing. `.sum()` counts the `True` values in each column (Python treats `True` as 1). It answers: *"How many missing values does each column have?"* Here every count is `0`.

---

## 9. What happens inside (high level)

When you look at a dataset, you are building a **mental model of the data**. The machine-learning model will later build its own, purely numerical, picture of the same data. Your job now is to understand the data well enough to spot when the machine's picture is wrong.

### Distributions: the shape of a column

A **distribution** describes *which values are common and which are rare* in a column. The quickest way to see one is a **histogram**: split the range of values into buckets and count how many rows fall into each bucket.

Here is the distribution of `MedHouseVal` (real counts, bucket width 0.5 = $50,000):

```text
MedHouseVal     in dollars       rows (each █ ≈ 200 rows)
0.0 – 0.5       $0k – $50k       █                          199
0.5 – 1.0       $50k – $100k     █████████████████         3397
1.0 – 1.5       $100k – $150k    ████████████████████      3960
1.5 – 2.0       $150k – $200k    ██████████████████████    4329
2.0 – 2.5       $200k – $250k    ███████████████           2926
2.5 – 3.0       $250k – $300k    ██████████                1963
3.0 – 3.5       $300k – $350k    ██████                    1214
3.5 – 4.0       $350k – $400k    ████                       881
4.0 – 4.5       $400k – $450k    ██                         477
4.5 – 5.0       $450k – $500k    ██                         302
5.0 (the cap)   "$500k or more"  █████                      992   ← spike!
```

What this shows:

1. Most neighbourhoods have median values between about $100k and $250k.
2. A long tail stretches to the right: fewer and fewer expensive areas.
3. **There's a spike at the very top.** Values above $500,000 were **capped** at 5.00001 when the dataset was created. So "5.0" really means "$500,000 *or more*". A model can never learn that some areas are worth $800k, because the data never says so.

You'll draw real histograms with matplotlib in the next phase (exploratory data analysis). For now, the numbers from `describe()` are enough to spot most of this.

### Suspicious values

`describe()` also reveals values that deserve a second look:

| Column | Suspicious value | Why it's suspicious |
|---|---|---|
| `AveRooms` | max ≈ 141.9 | 142 rooms per household? Probably a block group with very few households but many rooms, e.g. holiday homes or a resort. |
| `AveBedrms` | max ≈ 34.1 | Similar story. |
| `AveOccup` | max ≈ 1,243 | 1,243 people per household? Likely a place like a dormitory, prison, or barracks, where many people but few "households" are counted. |
| `Population` | max = 35,682 | Much larger than a typical block group. |
| `HouseAge` | max = 52, with many rows at exactly 52 | Probably also capped: "52" means "52 years *or older*". |
| `MedHouseVal` | max = 5.00001, with many rows at the cap | Capped at $500,000 (see above). |

These values aren't necessarily *errors*, but they're **unusual**. We call such values **outliers**. We won't change them now. We just note them, because they may later explain some of the model's biggest mistakes.

### Why understand data *before* training?

- **Garbage in, garbage out.** A model will happily learn from broken data and give confident, wrong answers.
- **Units matter.** An error of "0.4" means nothing until you know it's 0.4 × $100,000 = $40,000.
- **Caps and quirks limit what's possible.** No model trained on this data can predict $700,000 correctly, because the data tops out at $500,000.
- **Choosing inputs requires meaning.** You can't reason about which features should help until you know what they are.

---

## 10. Common beginner mistakes

- **Thinking each row is one house.** It's one *neighbourhood*. `AveRooms = 6` means "households here have 6 rooms *on average*".
- **Forgetting the units.** `MedHouseVal = 2` is $200,000, not $2. `MedInc = 5` is $50,000, not $5.
- **Only looking at `head()`.** The first rows are all from the same part of California. Use `df.sample(5)` and `describe()` to see the whole picture.
- **Trusting `mean` alone.** The mean is pulled by extreme values (look at `AveOccup`: mean 3.07 but max 1,243). The median (`50%`) is often a better "typical" value.
- **"No missing values" means "clean data".** There are no empty cells here, but there are still capped values and strange outliers.
- **Deleting outliers automatically.** Don't remove data just because it looks odd. First understand *why* it's odd.

---

## 11. Exercises

Use the notebook for these.

**Exercise 1 — Predict first, then check.**
Before running `df.shape`, write down how many rows and columns you expect. Then run it. Were you right?

**Exercise 2 — Read a random row.**
Run `df.sample(1, random_state=0)`. Turn the row into a plain-English sentence like the one in section 5, converting income and house value to dollars.

<details>
<summary>Hint</summary>

Multiply `MedInc` by $10,000 and `MedHouseVal` by $100,000.
</details>

**Exercise 3 — Find the extremes.**
Which row has the highest `AveOccup`? Look at its other values. What kind of place could it be?

<details>
<summary>Hint</summary>

`df["AveOccup"].idxmax()` gives the index (row label) of the largest value, and `df.loc[that_index]` shows that row.
</details>

**Exercise 4 — Count the cap.**
How many rows have `MedHouseVal` at its maximum value? What percentage of the dataset is that?

<details>
<summary>Hint</summary>

`df["MedHouseVal"] == df["MedHouseVal"].max()` gives `True`/`False` for each row, and `.sum()` counts the `True` values. Divide by `len(df)` for a fraction.
</details>

**Exercise 5 — Hypotheses.**
Without running any code, which three columns do you think will be most useful for predicting `MedHouseVal`? Write down your reasons. You'll test these hypotheses in later chapters.

---

## 12. Before continuing, I should be able to explain:

- □ What does one row of this dataset represent? (Hint: not a house.)
- □ What does each of the 9 columns mean, and what are the units of `MedInc` and `MedHouseVal`?
- □ What do `df.shape`, `df.info()`, `df.describe()`, and `df.isna().sum()` each tell me?
- □ Are there missing values? Are there suspicious values? Give two examples.
- □ What does "distribution" mean, and what's unusual about the distribution of `MedHouseVal`?
- □ Why must I understand the data before training a model?

---

**Previous:** [← 00 — The Big Picture](00-ml-big-picture.md) · **Next:** [02 — Features and Target →](02-features-and-target.md)
