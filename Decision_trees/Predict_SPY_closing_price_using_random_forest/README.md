# Random Forest From Scratch: Predicting Next-Day SPY Returns

A regression random forest built with only `numpy` and `pandas` (no scikit-learn), trained to predict the **next trading day's return** of SPY from past price features, and evaluated honestly against trivial baselines.

The goal is not a trading edge. It is a leak-free, time-aware pipeline that I understand end to end, and an honest answer to "does this model beat a trivial guess?"

---

## What it does

1. Downloads daily SPY prices from 2010 onward with `yfinance` (split/dividend-adjusted).
2. Builds features from **past data only** and a target from **future data only**.
3. Splits chronologically: oldest 80% for training, newest 20% for testing.
4. Trains a random forest (bootstrap rows, random feature subset at every split, average of all trees).
5. Compares the forest to two baselines on the test set.

### Features (inputs)

| Feature | Meaning |
|---|---|
| `ret_1`, `ret_2`, `ret_5`, `ret_10` | Return over the last 1, 2, 5 and 10 trading days |
| `vol_10` | Standard deviation of daily returns over the last 10 days |
| `ma_ratio` | Close divided by its 20-day moving average |

All features are ratios or returns, never raw prices, so they mean the same thing in 2010 and 2025.

### Target (what is predicted)

`y1 = close.shift(-1) / close - 1`, the return from today's close to tomorrow's close. A prediction above 0 is read as "up," below 0 as "down," and the size of the number is the predicted percent move.

---

## Setup

```bash
pip install numpy pandas yfinance
```

Python 3.9+ is fine.

## How to run

Save the script (e.g. as `random_forest_stock.py`) and run:

```bash
python random_forest_stock.py
```

It will:
- download the data,
- train the forest (progress prints every 10 trees),
- print forest vs. baseline results on the test set.

Example output:

```
Forest   MSE: 8.71e-05  dir acc: 0.546
Baseline (mean) MSE: 8.76e-05
Baseline (always up) dir acc: 0.562
```

### Using the model in your own code

```python
rf = RandomForestRegressor(n_trees=100, max_depth=4, min_leaf=40,
                           feature_fraction=0.33, seed=42)
rf.fit(X_train, Y_train)      # numpy arrays
p = rf.predict(X_test)        # one predicted return per test day
```

`X` must be a **numpy array** (use `df[cols].values`), not a DataFrame. Features and target are kept in separate objects so the target can never leak in as an input.

---

## How it works

| Piece | What it does |
|---|---|
| `best_split` | For each allowed feature, sorts the rows once and uses cumulative sums to score **every** threshold in a single pass. Picks the split with the lowest sum of squared errors. Thresholds are midpoints between neighboring values. |
| `build_tree` | Recursively splits rows into left/right groups until `max_depth` is reached or a node is too small. A leaf stores the **mean target** of its rows. |
| `predict_row` | Walks a tree from the root to a leaf. |
| `RandomForestRegressor.fit` | For each tree: draws a bootstrap sample (rows with replacement) and grows a tree that sees a random subset of features **at every split**. |
| `RandomForestRegressor.predict` | Averages all trees' predictions for each row. |

### Leakage safeguards

- **Chronological split**, never shuffled, because neighboring days are highly similar.
- **A gap at the boundary:** the last training row(s) have labels that reach into the test period, so they are dropped.
- **No normalization**, since trees only ask "above or below a threshold?" and rescaling changes nothing.
- **Features use only data available at that day's close.**

---

## Hyperparameters and how to tune them

| Parameter | Default | Effect |
|---|---|---|
| `n_trees` | 100 | More trees means smoother averages. Gains flatten after ~50-100. |
| `max_depth` | 4 | Deeper trees memorize noise. Keep shallow for financial data. |
| `min_leaf` | 40-50 | Minimum rows per leaf. **The most important knob here.** Small leaves create extreme, unreliable predictions. |
| `feature_fraction` | 0.33-0.5 | Share of features each split may consider. Lower means more diverse trees. |
| `seed` | 42 | Controls bootstrap and feature randomness. |

> If your script still names `min_leaf` as `min_samples`, use that name instead. Make sure `__init__` and `fit` use the same attribute name.

### Tuning rules

1. **Change one thing at a time.** Changing several at once makes it impossible to know what helped.
2. **Do not tune on the test set.** Every setting picked because it looked good on test quietly turns the test set into training data. Carve a **validation slice** off the end of the training data, tune on that, and touch the test set once at the end.
3. **Check seed stability** before trusting any small difference.

### Tuning on a validation slice

```python
val_start = int(len(train) * 0.8)
tr  = train.iloc[:val_start - 1]    # gap row dropped
val = train.iloc[val_start:]

Xtr, ytr = tr[cols].values,  tr['y1'].values
Xv,  yv  = val[cols].values, val['y1'].values

for min_leaf in [20, 40, 80, 160]:
    rf = RandomForestRegressor(n_trees=50, max_depth=4, min_leaf=min_leaf,
                               feature_fraction=0.5, seed=42)
    rf.fit(Xtr, ytr)
    print(min_leaf, mse(yv, rf.predict(Xv)))
```

Repeat for `max_depth` (2, 3, 4, 6) and `feature_fraction` (0.33, 0.5, 0.7).

### Seed stability check

```python
for seed in range(1, 6):
    rf = RandomForestRegressor(n_trees=100, max_depth=4, min_leaf=40,
                               feature_fraction=0.33, seed=seed)
    rf.fit(X_train, Y_train)
    p = rf.predict(X_test)
    signal = p > np.median(p)
    print(seed,
          "MSE", mse(Y_test, p),
          "dir", dir_acc(Y_test, p),
          "spread", Y_test[signal].mean() - Y_test[~signal].mean())
```

If the MSE edge or the spread flips sign across seeds, it is noise.

### Tips

- Start with `n_trees=10` to check speed. Prediction is pure Python, so it scales with trees × rows.
- Use `n_trees=1`, then 10, then 100, and watch test MSE improve and flatten.

---

## Evaluation

| Metric | What it measures | Baseline to beat |
|---|---|---|
| **MSE** | Average squared prediction error | Always predict the training-set mean return |
| **Direction accuracy** | How often `sign(prediction) == sign(actual)` | Always predict "up" (markets drift upward, so this scores above 50% for free) |
| **Median-split spread** | Average actual return on days the forest was relatively bullish minus relatively bearish | Zero |

---

## Results

Test set: 842 trading days (newest 20% of data).

| | Forest (`min_leaf=40`, `feature_fraction=0.33`, seed 1) | Baseline |
|---|---|---|
| MSE | 8.71e-05 | 8.76e-05 (predict the mean) |
| Direction accuracy | 54.6% | 56.2% (always up) |
| Median-split spread | +0.041% per day | 0 |

An earlier run with small leaves gave a similar picture (MSE 8.68e-05, direction accuracy 55.2%).

**Summary:** the forest is roughly tied with the mean baseline on error, slightly worse than "always up" on direction, and shows no statistically detectable ranking skill.

---

## What I learned

**About the data**
- Daily returns are mostly noise. Most of the error (a daily standard deviation near 0.9%) cannot be removed by any model.
- Rows are not independent: neighboring days overlap in information, so random shuffling leaks the future into training. Time order must be preserved.
- Predict returns, not prices. A tree can only output averages of values it has seen, so it cannot extrapolate to new price levels.
- "Always up" is a hard direction baseline (~56%) because markets drift upward. An accuracy of 55% is not good just because it is above 50%.

**About random forests**
- Bootstrap sampling and random feature subsets are what make the trees different. Without them, every tree is identical and averaging does nothing.
- Averaging many trees pulls predictions toward the mean. My predictions had a standard deviation of only ~0.11% against actual returns of ~0.9%. That is hedging, not a bug.
- Small leaves are dangerous. With tiny `min_leaf`, predictions as large as +2.1% and -1.3% appeared, which came from small clusters of freak days. With `min_leaf=40`, **no prediction exceeded 0.5%**.
- Trees need no normalization, because they only compare a value to a threshold.

**About implementation**
- Sorting once and using cumulative sums (`SSE = Σy² − (Σy)²/n`) scores all thresholds at once and is orders of magnitude faster than filtering a DataFrame inside a double loop.
- `DataFrame.apply(axis=1)` is very slow. Numpy arrays and integer column indices are far faster.
- Small bugs can run silently. A misplaced `return` inside a `for` loop made the function check only one feature and still produced output. Toy tests (e.g. `x=[1,2,10,11]`, `y=[0,0,5,5]` should split at 6.0) catch this kind of thing.

**About evaluation**
- Baselines matter more than the model's raw score. Without them, 55% accuracy would look like success.
- A result near the baselines is what a leak-free pipeline should produce. Suspiciously high accuracy (65-70%+) should trigger a hunt for leakage, not celebration.
- A small difference needs a significance check. The +0.041% median-split spread is about 0.7 standard errors, well within chance.
- Looking at test results under many settings slightly contaminates the test set. Tune on validation data instead.

**Bottom line:** simple price and volume features do not predict next-day SPY returns beyond what a constant guess already captures. That is a legitimate, expected finding for a very liquid ETF, and the pipeline is the real deliverable.

---

## Limitations

- One ticker and only six price-based features.
- Results come from a single train/test split. Performance could differ in other market regimes.
- Standard bootstrapping resamples individual days and ignores that neighboring days are correlated.
- No transaction costs or backtest, so direction accuracy says nothing about profitability.
- Fundamentals are excluded because `yfinance` does not provide point-in-time values, and using restated figures would leak the future.
