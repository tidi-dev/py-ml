# Glossary

Beginner-friendly definitions of every term used in this course. Each entry has a **plain-English** meaning and a **project example** from California Housing. The chapter link points to where the idea is explained properly.

> **Tip:** don't try to memorise this page. Come back to it whenever a word feels fuzzy.

**Jump to:** [The big ideas](#the-big-ideas) · [Data](#data) · [Models and training](#models-and-training) · [Splitting data](#splitting-data) · [Evaluation](#evaluation) · [Generalization](#generalization) · [Good practice](#good-practice)

---

## The big ideas

### Machine learning
- **Plain English:** Getting a computer to find patterns in examples, instead of a programmer writing the rules by hand.
- **Project example:** Instead of writing rules for house prices, we show the computer 16,000+ neighbourhoods with known values and let it find the relationship itself.
- **Learn more:** [00](00-ml-big-picture.md)

### Supervised learning
- **Plain English:** Machine learning where every training example comes with the correct answer, so the computer can learn by comparing its guesses with the answers.
- **Project example:** Every row has its real `MedHouseVal`, so the model can check how wrong each guess was while learning.
- **Learn more:** [00](00-ml-big-picture.md)

### Regression
- **Plain English:** Predicting a **number** on a continuous scale.
- **Project example:** Predicting `MedHouseVal = 2.37` (≈ $237,000). It could be any value, not one of a fixed set of options.
- **Learn more:** [00](00-ml-big-picture.md), [06](06-evaluating-a-model.md)

### Classification
- **Plain English:** Predicting a **category** from a fixed set of options.
- **Project example:** Not what we do here. If we instead predicted "cheap / medium / expensive area", that would be classification.
- **Learn more:** [00](00-ml-big-picture.md)

---

## Data

### Dataset
- **Plain English:** The whole table of examples we learn from.
- **Project example:** California Housing: 20,640 rows × 9 columns.
- **Learn more:** [01](01-understanding-the-data.md)

### Sample
- **Plain English:** One example: one row of the dataset. (In statistics it can also mean a *group* of rows. In this course, it means one row.)
- **Project example:** One census block group, e.g. row 0 with `MedInc = 8.33` and `MedHouseVal = 4.526`.
- **Learn more:** [01](01-understanding-the-data.md)

### Observation
- **Plain English:** Another word for one row: one thing we observed and recorded.
- **Project example:** Each of the 20,640 block groups is an observation.
- **Learn more:** [01](01-understanding-the-data.md)

### Feature
- **Plain English:** A piece of information we give the model to help it make a prediction.
- **Project example:** `MedInc` is a feature because median income is information the model can use to predict `MedHouseVal`.
- **Learn more:** [02](02-features-and-target.md)

### Target
- **Plain English:** The value we want the model to predict: the "answer".
- **Project example:** `MedHouseVal`, the median house value, in units of $100,000.
- **Learn more:** [02](02-features-and-target.md)

### X
- **Plain English:** The table of all feature values: the information the model gets. By convention written with a capital letter because it's a table.
- **Project example:** `X = df[["MedInc", "HouseAge", "AveRooms"]]`, which is 20,640 rows × 3 columns.
- **Learn more:** [02](02-features-and-target.md)

### y
- **Plain English:** The column of correct answers, one per row of `X`. Written lowercase because it's a single column.
- **Project example:** `y = df["MedHouseVal"]`, which is 20,640 values.
- **Learn more:** [02](02-features-and-target.md)

### Distribution
- **Plain English:** How the values of one column are spread out: which values are common, which are rare, and what the extremes are.
- **Project example:** Most `MedHouseVal` values are between 1 and 2.5, with a long tail of expensive areas and a spike at the 5.0 cap.
- **Learn more:** [01](01-understanding-the-data.md)

### Outlier
- **Plain English:** A value that is very different from the rest. It might be an error, or just unusual.
- **Project example:** `AveOccup = 1,243` people per household, probably a dormitory or similar.
- **Learn more:** [01](01-understanding-the-data.md)

### Preprocessing
- **Plain English:** Preparing or transforming data before a model uses it, e.g. filling in missing values, rescaling numbers, or turning text categories into numbers.
- **Project example:** California Housing needs very little: it's all numeric with no missing values. Richer datasets need much more.
- **Learn more:** [09](09-ml-workflow.md)

---

## Models and training

### Algorithm
- **Plain English:** The *recipe* for learning: the method used to find patterns in the data.
- **Project example:** "Find the best straight line" (Linear Regression) and "build many question trees and average them" (Random Forest) are two different algorithms.
- **Learn more:** [00](00-ml-big-picture.md)

### Model
- **Plain English:** What you get **after** running the algorithm on data: a small set of learned numbers (or questions) that turns feature values into a prediction. It's not a database of answers.
- **Project example:** A trained `LinearRegression` on 3 features is just 4 numbers: an intercept and 3 coefficients.
- **Learn more:** [00](00-ml-big-picture.md), [05](05-linear-regression.md)

### Training
- **Plain English:** Showing the algorithm examples *with* their answers and letting it adjust its internal numbers until its guesses are as close to the answers as it can make them. In scikit-learn: `model.fit(X_train, y_train)`.
- **Project example:** Linear Regression finds the intercept and coefficients that make its predictions on the 16,512 training areas as close as possible to their real values.
- **Learn more:** [03](03-training-and-testing.md)

### Parameter
- **Plain English:** A number the model **learns** from the data during training.
- **Project example:** The coefficients (`model.coef_`) and intercept (`model.intercept_`) of a Linear Regression.
- **Learn more:** [05](05-linear-regression.md)

### Hyperparameter
- **Plain English:** A setting **you** choose *before* training. The model doesn't learn it.
- **Project example:** `max_depth` of a decision tree, or `n_estimators` (number of trees) of a Random Forest. Compare with *parameter*.
- **Learn more:** [07](07-overfitting-and-generalization.md), [08](08-random-forest.md)

### Prediction
- **Plain English:** The model's guess of the target for one example.
- **Project example:** `model.predict(X_test)` returns guesses like 2.31 (≈ $231,000) for each test area.
- **Learn more:** [05](05-linear-regression.md)

### Inference
- **Plain English:** Using an already-trained model to make predictions. No learning happens during inference.
- **Project example:** Calling `model.predict(new_areas)` for neighbourhoods the model has never seen.
- **Learn more:** [00](00-ml-big-picture.md)

### Baseline
- **Plain English:** A deliberately simple strategy that any real model must beat. Without it, a score has nothing to be compared with.
- **Project example:** `DummyRegressor(strategy="mean")` always predicts the average training value (≈ $207k).
- **Learn more:** [04](04-first-baseline-model.md)

---

## Splitting data

### Training set
- **Plain English:** The part of the data the model is allowed to learn from.
- **Project example:** `X_train`, `y_train`: 80% of rows (16,512 block groups).
- **Learn more:** [03](03-training-and-testing.md)

### Test set
- **Plain English:** The part of the data hidden during training, used only to check how well the model works on unseen examples.
- **Project example:** `X_test`, `y_test`: 20% of rows (4,128 block groups).
- **Learn more:** [03](03-training-and-testing.md)

### Validation set
- **Plain English:** A third portion of data (carved out of the training data) used to **make choices**, such as which model or settings to use, so the test set stays untouched until the very end.
- **Project example:** Later in the project, choosing `max_depth` using a validation set instead of peeking at the test set.
- **Learn more:** [07](07-overfitting-and-generalization.md), [09](09-ml-workflow.md)

### Cross-validation
- **Plain English:** Evaluating a model several times on different splits of the training data and averaging the scores, so the result doesn't depend on one lucky or unlucky split.
- **Project example:** 5-fold cross-validation: split the training data into 5 parts, train on 4 and evaluate on the 5th, rotate 5 times, and average.
- **Learn more:** [09](09-ml-workflow.md)

---

## Evaluation

### Error
- **Plain English:** How far one prediction is from the truth: actual − predicted. Also called a *residual*.
- **Project example:** Actual $300k, predicted $270k → error = $30k.
- **Learn more:** [06](06-evaluating-a-model.md)

### MAE
- **Plain English:** *Mean Absolute Error*: the average size of the mistakes, ignoring whether they're too high or too low. In the same units as the target.
- **Project example:** MAE = 0.40 means predictions are off by about 0.40 × $100,000 = **$40,000** on average.
- **Learn more:** [06](06-evaluating-a-model.md)

### RMSE
- **Plain English:** *Root Mean Squared Error*: like MAE, but big mistakes count much more, because errors are squared before averaging. Always ≥ MAE.
- **Project example:** Errors of $0k, $0k, $90k give MAE = $30k but RMSE ≈ $52k. The gap reveals one big mistake.
- **Learn more:** [06](06-evaluating-a-model.md)

### R²
- **Plain English:** "How much better than always guessing the average?" 1 = perfect, 0 = no better than the average, below 0 = worse. It is **not** an accuracy percentage.
- **Project example:** R² = 0.85 means the model removed 85% of the squared error that "always guess the average" would make.
- **Learn more:** [06](06-evaluating-a-model.md)

---

## Generalization

### Generalization
- **Plain English:** A model's ability to work well on new data it didn't see during training. This is the real goal of ML.
- **Project example:** Good predictions for the 4,128 test areas, not just the 16,512 training areas.
- **Learn more:** [03](03-training-and-testing.md), [07](07-overfitting-and-generalization.md)

### Overfitting
- **Plain English:** The model learned the training examples *too* specifically, including their noise and quirks, so it does much better on training data than on new data. Like a student who memorised the practice answers.
- **Project example:** An unlimited-depth decision tree gets a training MAE of 0 but a noticeably higher test MAE.
- **Learn more:** [07](07-overfitting-and-generalization.md)

### Underfitting
- **Plain English:** The model is too simple to capture the real patterns, so it does poorly on training *and* test data.
- **Project example:** `DummyRegressor` ignores all features, so its error is high everywhere.
- **Learn more:** [07](07-overfitting-and-generalization.md)

---

## Good practice

### Data leakage
- **Plain English:** Accidentally giving the model information during training that it wouldn't really have when making a real prediction. It makes results look better than they will be in real use.
- **Project example:** Putting `MedHouseVal` (or something computed from it) inside `X`. Another example is computing statistics on all the data before splitting.
- **Learn more:** [02](02-features-and-target.md), [03](03-training-and-testing.md)

### Pipeline
- **Plain English:** A chain of steps ("prepare the data, then apply the model") bundled into one object with one `fit()` and one `predict()`. It keeps preparation consistent and prevents leakage.
- **Project example:** Later: `Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])`.
- **Learn more:** [09](09-ml-workflow.md)

### Reproducibility
- **Plain English:** Getting exactly the same results when the code is run again, by you or anyone else.
- **Project example:** Using `random_state=42` so the train/test split (and Random Forest) come out the same every time.
- **Learn more:** [03](03-training-and-testing.md)

---

**Back to:** [README](../README.md) · [00 — The Big Picture](00-ml-big-picture.md)
