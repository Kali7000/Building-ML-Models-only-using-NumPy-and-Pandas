# Decision Tree Regression (From Scratch)

A complete implementation of the CART (Classification and Regression Trees) algorithm built entirely from scratch using Python, NumPy, and Pandas. This project was developed without the use of high-level machine learning libraries like `scikit-learn` to gain a deep mathematical understanding of recursive partitioning and variance reduction.

## 🌳 Project Overview

This model predicts continuous numerical values by building a binary decision tree. It is currently configured to predict daily bike rental demand based on weather, seasonal, and environmental features using the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset).

### Key Features Implemented:

* **Weighted Mean Squared Error (MSE):** The algorithm evaluates the quality of potential splits by calculating the variance of the left and right subsets, weighted by the number of samples in each.
* **Recursive Partitioning:** The tree builds itself dynamically using a nested dictionary structure, continuously splitting the data until a base case is met.
* **Hyperparameter Tuning:** Includes stopping conditions like `max_depth` and a minimum sample size (`len(df) < 15`) to prevent overfitting.
* **Recursive Traversal:** A custom prediction engine that routes unseen data rows through the generated dictionary branches.
* **Visualizer:** A recursive text-based print function that maps out the exact mathematical boundaries the model learned in a readable format.

## 🔬 The Mathematics of the Split

At every node, the model iterates through every unique value of every available feature. It draws a threshold (e.g., `temp <= 15.5`) and splits the data into two branches.

The quality of the split is calculated as:
`Weighted MSE = (N_left / N_total) * Var(Left) + (N_right / N_total) * Var(Right)`

The feature and threshold combination that results in the lowest Weighted MSE is selected as the optimal node, and the process repeats on the resulting subsets.

## 🌲 Tree Visualization Example

Because the model is built from scratch, a custom recursive text visualizer maps out the decision boundaries. Here is a truncated snippet of the actual boundaries the model learned to predict bike rentals:

```text
If temp <= 0.4308:
   |If yr <= 0.0000:
   |   |If season <= 2.0000:
   |   |   |If mnth <= 2.0000:
   |   |   |   |If mnth <= 1.0000:
   |   |   |   |   |If hum <= 0.6864:
   |   |   |   |   |   |If workingday <= 0.0000:
   |   |   |   |   |   |   |--> Predict: 1037.25 bikes
   |   |   |   |   |   |Else (workingday > 0.0000):
   |   |   |   |   |   |   |--> Predict: 1504.80 bikes
...
```

## 🚀 Next Step: The Evolution to Gradient Boosting

While a single CART decision tree is a powerful, highly interpretable algorithm, it has mathematical limitations. A deep tree will overfit and memorize noise, while a shallow tree will underfit and miss complex interactions. 

To achieve state-of-the-art predictive accuracy, this exact decision tree architecture serves as the foundation (the "weak learner") for **Gradient Boosting** (the architecture behind XGBoost and LightGBM). 

### How Gradient Boosting Extends This Project:
1. **Sequential Learning:** Instead of building one massive tree, Gradient Boosting builds hundreds of shallow trees sequentially.
2. **Predicting the Error:** Tree 1 makes a baseline prediction. Tree 2 is then trained *not* on the original target, but on the **residuals** (the errors) of Tree 1. Tree 3 targets the errors of Tree 2, and so on.
3. **Gradients & Hessians:** Instead of evaluating splits using simple Variance/MSE, advanced boosting models evaluate splits using the first-order (Gradient) and second-order (Hessian) derivatives of a custom loss function, allowing the model to optimize for highly complex, non-linear financial or environmental data.
4. **Shrinkage:** Each tree's final prediction is scaled down by a Learning Rate ($\eta$), forcing the ensemble to learn slowly and robustly, completely mitigating the single tree's tendency to overfit.

## 💻 How to Run

### 1. Prerequisites

```bash
pip install pandas numpy
```

### 2. Dataset Setup
Download the `day.csv` file from the UCI Bike Sharing dataset and place it in the same directory as the script.

### 3. Execution
Run the script directly from your terminal:
```bash
python decision_tree.py
```