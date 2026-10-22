# Machine Learning Algorithms from Scratch (NumPy + pandas only)

This repository contains pure Python implementations of foundational machine learning algorithms, built entirely from scratch.

Rather than relying on high-level, optimized libraries like `scikit-learn` or `TensorFlow`, every algorithm here is constructed using only the standard `math` library, `numpy` for matrix operations, and `pandas` for data manipulation. The goal is to demystify the "black box" of machine learning by translating calculus, linear algebra, and probability theory into working code. Every project is a small, readable script that you can open, run, and trace line by line.

> **Status:** actively growing. New projects are added regularly, so check back or star the repo.

## Core philosophy

Using `.fit()` and `.predict()` is easy. Understanding the calculus behind a gradient update, the logic of a recursive greedy split, or the array manipulation behind a proper evaluation pipeline is hard.

This repository is a transparent exploration of how these algorithms actually learn. If you have ever called `model.fit()` and wondered what happens inside, each project here is my attempt to answer that question, one algorithm at a time.

## Technology stack

- **Python 3.9+**
- **NumPy:** vectorized matrix and dot-product operations and array manipulation, replacing slow native Python `for` loops wherever possible.
- **pandas:** loading datasets, isolating features, structuring data, and preparing time-series features (returns, rolling windows) in the stock project.
- **`math` (standard library):** raw logarithms, exponentials, and other math primitives where needed.
- **Optional extras:** `yfinance` to download prices for the SPY project, and `scikit-learn` for the comparison scripts only. The from-scratch code never depends on it.

---

## Who this is for

- **Learners** who want to see how an algorithm works under the hood, not just how to call it.
- **Anyone** studying for ML interviews or courses who wants working reference implementations in plain Python.
- **Me**, as a record of what I have built and learned.

## What you will find

- Plain Python scripts, one per algorithm or experiment, with small datasets stored next to the code.
- Comments that explain the *why* behind each step.
- Several projects have their own README with a longer write-up: how to run it, how to tune it, and what I learned.
- Some projects include a second script that compares the from-scratch result with scikit-learn, used purely as a sanity check. The from-scratch version is always the main implementation.

---

## Repository map

```
.
├── Decision_trees/
│   ├── CART_Regression_tree/
│   ├── ID3_classification_tree/
│   ├── Predict_SPY_closing_price_using_random_forest/
│   └── RandomForestRegressor_predicting_bike_rentals_each_day/
├── K-Fold cross Validation/
├── Lasso Regression on Housing Data/
├── Linear regression/
│   ├── Encoding Non_num columns/
│   ├── linear regression single variables/
│   ├── Multivarient Linear Regression/
│   └── Ploynomial Regression/
│       ├── single variable polynomial regression/
│       └── Multivarient polynomial regression/
├── Logistic regression/
│   ├── Logistic regression 1/
│   ├── Multinomial Logistic Regression/
│   ├── Ordinal_logistics_regression/
│   └── logisctic regression spam detection/
└── spam-detection/
```

---

## Project index

### Linear models for regression

| Project | What it covers | Data |
|---|---|---|
| [Single-variable linear regression](Linear%20regression/linear%20regression%20single%20variables) | The simplest model: fit a line to one feature | `Salary_dataset.csv` |
| [Multivariate linear regression](Linear%20regression/Multivarient%20Linear%20Regression) | Several features at once; two versions (`LR2.py`, `LR2.1.py`) | `insurance.csv` |
| [Encoding non-numeric columns](Linear%20regression/Encoding%20Non_num%20columns) | One-hot encoding and ordinal encoding, to turn categories into numbers | n/a |
| [Polynomial regression (single variable)](Linear%20regression/Ploynomial%20Regression/single%20variable%20polynomial%20regression) | Fitting curves by adding polynomial terms | `Ice_cream selling data.csv` |
| [Polynomial regression (multivariable, with ridge)](Linear%20regression/Ploynomial%20Regression/Multivarient%20polynomial%20regression) | Multiple features with polynomial terms, plus a ridge (L2) regularized version | `algeas.csv` |
| [Lasso regression on housing data](Lasso%20Regression%20on%20Housing%20Data) | L1 regularization for shrinking and selecting features. Has its own README | `train.csv`, `test.csv` |

### Classification with logistic regression

| Project | What it covers | Data |
|---|---|---|
| [Logistic regression](Logistic%20regression/Logistic%20regression%201) | Binary classification basics | n/a |
| [Multinomial logistic regression](Logistic%20regression/Multinomial%20Logistic%20Regression) | Softmax for more than two classes, applied to the Iris data, with a scikit-learn comparison | Iris |
| [Ordinal logistic regression](Logistic%20regression/Ordinal_logistics_regression) | Classification where classes have a natural order (wine quality). Has its own README | `WineQT.csv` |
| [Spam detection (logistic regression)](Logistic%20regression/logisctic%20regression%20spam%20detection) | Email text classification, with sample emails to try the model on | `emails.csv` |
| [Spam detection](spam-detection) | A standalone spam classifier script | `Emails.csv` |

### Decision trees and random forests

| Project | What it covers | Data |
|---|---|---|
| [ID3 classification tree](Decision_trees/ID3_classification_tree) | A classification tree built with information gain, on the classic Play Tennis problem | `PlayTennis.csv` |
| [CART regression tree](Decision_trees/CART_Regression_tree) | A regression tree that picks splits by minimizing variance. Has its own README | `day.csv` |
| [Random forest: bike rentals](Decision_trees/RandomForestRegressor_predicting_bike_rentals_each_day) | Bootstrap sampling plus random feature subsets, averaging many trees. Has its own README | bike-sharing daily data |
| [Random forest: SPY stock returns](Decision_trees/Predict_SPY_closing_price_using_random_forest) | A time-aware random forest for next-day SPY returns, with leak-free splits and honest baselines. Has its own README | downloaded with `yfinance` |

### Model evaluation

| Project | What it covers |
|---|---|
| [K-fold cross validation](K-Fold%20cross%20Validation) | Splitting data into folds to estimate how well a model generalizes |

---

## Suggested reading order

If you are new to this, the projects build on each other. A natural path:

1. **Linear regression (single variable)**, the foundation: loss, gradients, fitting.
2. **Multivariate linear regression**, the same idea with many features.
3. **Encoding non-numeric columns**, getting real-world data into a model.
4. **Polynomial regression**, then **ridge** and **Lasso**, which cover non-linearity and regularization.
5. **K-fold cross validation**, which shows how to evaluate a model properly.
6. **Logistic regression**, then **multinomial** and **ordinal** variants, which move from numbers to classes.
7. **Spam detection**, an end-to-end text classification example.
8. **ID3**, then **CART**, which cover decision trees for classification and regression.
9. **Random forest (bike rentals)**, then **Random forest (SPY)**, which add ensembles and the harder question of evaluating on time-ordered data.

---

## How to run a project

**Requirements:** Python 3.9+ and:

```bash
pip install numpy pandas
```

A few projects need extra packages:
- The SPY project also needs `yfinance` to download prices.
- Scripts named `*sklearn*` need `scikit-learn`. They are optional comparisons.

**Run a script:**

```bash
git clone https://github.com/<your-username>/ml-from-scratch.git
cd ml-from-scratch
pip install numpy pandas

cd Decision_trees/CART_Regression_tree
python CART_regression_decsion_tree.py
```

Datasets sit in the same folder as the script that uses them. Many scripts load their CSV with a relative path, so **run the script from inside its own folder**. If you get a "file not found" error, that is almost always the reason.

---

## How each project folder is organized

- **The `.py` file(s):** the core algorithm: the math, the training loop, and the prediction logic. Start here. Most scripts also include a runnable demo (usually under `if __name__ == "__main__":`) that loads a dataset, builds the model, trains it, and evaluates the results, using cross-validation where it applies (see the K-fold project).
- **The `.csv` file(s):** the raw dataset used to train and test the model (for example Wine Quality or Play Tennis).
- **`README.md` (when present):** a deeper write-up covering the approach, how to tune it, the results, and what I learned. The folders with one right now are CART regression tree, both random forest projects, Lasso regression, and ordinal logistic regression.

---

## Design principles

- **From scratch:** the core algorithm is always written with NumPy and pandas only.
- **Vectorized, but readable:** NumPy array operations replace slow Python loops wherever that does not hurt clarity (for example, scoring every possible tree split with cumulative sums). Clarity comes first, and any performance trick is explained where it is used.
- **Honest evaluation:** results are compared against simple baselines, and train/test splits are done carefully (for example, chronologically for time series) to avoid leakage.
- **Learn in public:** mistakes, dead ends, and lessons are part of the write-ups.

---

## Roadmap

New projects are added over time. Ideas I'm considering:

- Gradient-boosted trees
- k-nearest neighbors and k-means clustering
- Naive Bayes
- Neural network built from scratch with backpropagation
- PCA

*(Edit this list to match your real plans.)*

---

## Contributing and feedback

This is a personal learning project, but suggestions and corrections are welcome. If you spot a bug, a confusing explanation, or a better way to do something, please open an issue or a pull request.

## Disclaimer

Everything here is educational. The stock-related project is a study of methodology, not financial advice or a trading strategy.

## Author

**Vishal Wagh**, [GitHub](https://github.com/kali7000) · LinkedIn: https://www.linkedin.com/in/wagh0/
