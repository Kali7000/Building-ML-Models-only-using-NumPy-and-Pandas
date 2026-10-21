# Ordinal Logistic Regression (From Scratch)

This repository contains a pure Python and NumPy implementation of an Ordinal Logistic Regression model. It is designed to classify data into ordered categories (such as quality ratings) without relying on high-level machine learning libraries like `scikit-learn` or `mord`. 

The model is demonstrated using the [Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset), predicting a wine's rating (3-8) based on 11 chemical properties.

## Mathematical Architecture

While a standard *Multinomial* model calculates independent scores for every category (ignoring natural order), this *Ordinal* model utilizes the "All-Thresholds" approach to respect the sequence of the data ladder.

1. **The Underlying Score:** The model calculates a single continuous score ($z$) for the wine using 11 weights.
   $$z = w_1x_1 + w_2x_2 + ... + w_{11}x_{11}$$
2. **Cumulative Probabilities:** The model compares the single score against 5 distinct thresholds ($t_0$ through $t_4$) to calculate the probability of falling below a specific rung on the ladder using the Sigmoid activation function.
   $$P(y \le k) = \frac{1}{1 + e^{-(t_k - z)}}$$
3. **Bucket Probabilities:** The exact probability for a specific category is found by subtracting the cumulative probabilities of adjacent thresholds.

## Key Features Implemented

* **Stochastic Gradient Descent:** The model trains via row-by-row error calculation and weight updates.
* **Strict Ordinal Constraints:** The algorithm ensures that thresholds $t_0 < t_1 < t_2$ remain strictly ordered during gradient descent.
* **L2 Regularization (Ridge):** A decay penalty (`lambda_val`) is applied to the weights once per epoch to prevent overfitting on the 11 features.
* **Class Weighting:** To combat the heavy class imbalance of the Wine Quality dataset (dominated by 5s and 6s), the error gradients for rare categories (3, 4, 7, 8) are magnified. This forces the model to learn the boundaries for rare events rather than defaulting to the majority class.
* **Data Leakage Prevention:** The `standardize` function isolates the mean and standard deviation of the Training set, applying those exact statistics to the Test set to prevent the model from peeking at future data distributions.

## Evaluation Metrics

Because Ordinal classification implies a ladder, strict accuracy is not the only metric of success. Guessing a quality rating of "5" when the true answer is "6" is a mathematically superior guess than guessing an "8".

The included `test_model` function tracks:
* **Strict Accuracy:** Exact target matches.
* **Off-By-One Accuracy:** The percentage of predictions that fell within 1 quality point of the true answer (typically >95% for this dataset).
* **Confusion Matrix:** A custom 6x6 formatted matrix to visualize prediction drift.

## Requirements
* `pandas`
* `numpy`
* `matplotlib` (Optional, for loss curve visualization)
