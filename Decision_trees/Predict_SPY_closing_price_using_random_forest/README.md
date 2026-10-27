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

## Customizing the setup: other horizons and other tickers

The forest code (`best_split`, `build_tree`, `predict_row`, `RandomForestRegressor`) does not change. Only the data-prep block in `__main__` does.

### A. Predict more than one day ahead

Change the shift in the target, and drop **as many rows from the end of train as the horizon**, because the label of row *t* reaches *H* days forward and would otherwise overlap the test period.

```python
HORIZON = 5   # 1 = tomorrow, 2 = two days ahead, 5 = about one trading week

target = pd.DataFrame({'y': close.shift(-HORIZON) / close - 1})
data = pd.concat([features, target], axis=1).dropna()   # also drops the last HORIZON rows

split = int(len(data) * 0.8)
train = data.iloc[:split - HORIZON]    # gap = HORIZON rows
test  = data.iloc[split:]

cols = list(features.columns)
X_train, X_test = train[cols].values, test[cols].values
Y_train, Y_test = train['y'].values, test['y'].values
```

To train several horizons in one run (one forest per horizon):

```python
for h in [1, 2, 5]:
    y = (close.shift(-h) / close - 1).rename('y')
    data = pd.concat([features, y], axis=1).dropna()
    split = int(len(data) * 0.8)
    train, test = data.iloc[:split - h], data.iloc[split:]

    X_train, Y_train = train[cols].values, train['y'].values
    X_test,  Y_test  = test[cols].values,  test['y'].values

    rf = RandomForestRegressor(n_trees=100, max_depth=4, min_leaf=40,
                               feature_fraction=0.33, seed=42)
    rf.fit(X_train, Y_train)
    p = rf.predict(X_test)

    print(f"h={h}  forest MSE {mse(Y_test, p):.2e}  baseline MSE {mse(Y_test, Y_train.mean()):.2e}  "
          f"dir acc {dir_acc(Y_test, p):.3f}  always-up {dir_acc(Y_test, np.ones(len(Y_test))):.3f}")
```

**What changes when the horizon grows**

- **Overlapping labels.** With `H > 1`, neighboring rows share most of their return window (the 5-day returns of Monday and Tuesday overlap by 4 days). Rows are even less independent than before, so the effective sample size is smaller and the test result is less trustworthy than the row count suggests.
- **MSE grows with the horizon.** Return variance scales roughly with the number of days, so a 5-day MSE is not comparable to a 1-day MSE. Only compare the forest to the baseline **at the same horizon**.
- **The always-up baseline changes.** Longer windows have a higher share of positive returns, so recompute it per horizon rather than reusing 56%.
- **Features may need longer lookbacks.** For `H = 5` or more, consider adding `ret_20`, `ret_60` and a longer volatility window.

### B. Use a different stock or asset type

Change the ticker passed to `yf.download`. Any symbol that works on Yahoo Finance works here.

```python
TICKER = "AAPL"
raw = yf.download(TICKER, start="2010-01-01", auto_adjust=True)
```

| Asset type | Example tickers | Notes |
|---|---|---|
| Individual stocks | `AAPL`, `MSFT`, `JPM`, `XOM` | Noisier than SPY (single-company news), so expect higher variance. |
| ETFs | `SPY`, `QQQ`, `IWM`, `GLD`, `TLT` | Very liquid and clean. |
| Indices | `^GSPC`, `^IXIC`, `^VIX` | Not tradable directly. `^VIX` is a volatility level, not a price, so returns behave differently. |
| Crypto | `BTC-USD`, `ETH-USD` | Trades 7 days a week (about 365 rows per year), and history starts around 2014, so adjust `start`. A "10-day window" now covers fewer calendar days. |
| Forex | `EURUSD=X` | Volume is usually 0 or missing, so skip any volume-based feature such as `vol_ratio`. |
| Futures | `GC=F` (gold), `CL=F` (crude oil) | Continuous-contract data can contain roll jumps. |

**Things to check when switching assets**

- **History length:** if the symbol started trading after your `start` date, you simply get fewer rows. Print `data.shape` and make sure the test set is still a few hundred rows.
- **Volume:** if you use `vol_ratio` and volume can be 0, you get `inf` or `NaN`. Drop that feature for those assets.
- **Compare against that asset's own baselines.** A MSE of 8e-05 means nothing for BTC, whose daily moves are several times larger than SPY's.
- **Check the column handling:** `raw.columns.get_level_values(0)` is needed for a single ticker with newer `yfinance`. Download tickers **one at a time** in a loop, not as a list, so the columns stay simple.

### C. Pool several tickers for more training data

Build features **per ticker** (so rolling windows never cross from one stock into another), then stack the results. Use a single **date cutoff** for the split, so no ticker's training period overlaps another ticker's test period (stocks move together on the same days, which would otherwise leak).

```python
def build_dataset(ticker, horizon=1, start="2010-01-01"):
    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    raw.columns = raw.columns.get_level_values(0)
    raw = raw.sort_index()
    close = raw['Close']

    df = pd.DataFrame(index=raw.index)
    df['ret_1']    = close.pct_change(1)
    df['ret_2']    = close.pct_change(2)
    df['ret_5']    = close.pct_change(5)
    df['ret_10']   = close.pct_change(10)
    df['vol_10']   = close.pct_change().rolling(10).std()
    df['ma_ratio'] = close / close.rolling(20).mean()
    df['y']        = close.shift(-horizon) / close - 1
    df['ticker']   = ticker
    return df.dropna()


TICKERS = ["SPY", "AAPL", "MSFT", "JPM", "XOM"]
HORIZON = 1

all_data = pd.concat([build_dataset(t, HORIZON) for t in TICKERS]).sort_index()

dates  = all_data.index.unique().sort_values()
cutoff = dates[int(len(dates) * 0.8)]                    # first test date
gap    = dates[int(len(dates) * 0.8) - HORIZON]          # drop HORIZON dates before the cutoff

train = all_data[all_data.index < gap]
test  = all_data[all_data.index >= cutoff]

cols = [c for c in all_data.columns if c not in ('y', 'ticker')]
X_train, X_test = train[cols].values, test[cols].values
Y_train, Y_test = train['y'].values, test['y'].values
```

Notes for pooled runs:

- Training takes several times longer (about 5x the rows). Start with `n_trees=10`.
- Pool assets that share a trading calendar. Mixing crypto (7 days a week) with stocks (5 days) breaks the "drop `HORIZON` rows" gap logic.
- Different stocks have different volatility, so evaluate per ticker too:

```python
for t in TICKERS:
    m = (test['ticker'] == t).values
    print(t, mse(Y_test[m], p[m]), mse(Y_test[m], Y_train.mean()))
```

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

## Ideas for next steps

1. Run the seed-stability check and report mean and spread across seeds.
2. Tune `min_leaf`, `max_depth` and `feature_fraction` on a validation slice.
3. Add features: volume ratio (`volume / volume.rolling(20).mean()`), RSI, 20-day return, day of week.
4. Report metrics per sub-period (e.g. 2-3 blocks of the test set).
5. Add a 2-day-ahead target and compare difficulty.
6. Pool several tickers (AAPL, MSFT, JPM, XOM) for more rows.
7. Add permutation feature importance (shuffle one column in the test set and measure the rise in MSE).
8. Run a simple backtest (long when the prediction is positive, cash otherwise) with transaction costs against buy-and-hold.
9. Sanity-check against scikit-learn's `RandomForestRegressor` on the same split.
10. Try block bootstrapping, which resamples chunks of consecutive days.

## Disclaimer

This is an educational project. Nothing here is financial advice or a trading strategy.
