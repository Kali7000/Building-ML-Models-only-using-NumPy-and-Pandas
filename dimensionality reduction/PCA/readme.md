# Principal Component Analysis (PCA) From Scratch

A pure NumPy implementation of Principal Component Analysis (PCA) built from scratch. This project demonstrates how to perform linear dimensionality reduction without relying on high-level machine learning libraries like `scikit-learn`. 

The algorithm is applied to the **Breast Cancer Wisconsin (Diagnostic) Dataset** to compress the high-dimensional feature space while retaining the maximum amount of mathematical variance (information).

## Algorithmic Pipeline & Mathematics

This script reduces the dataset down to its 7 most important dimensions through the following core linear algebra steps:

### 1. Data Standardization
PCA is highly sensitive to the scale of the data. If one feature is measured in thousands and another in decimals, the larger feature will falsely dominate the variance. 
*   **Action:** Every feature is mean-centered and scaled by its standard deviation so that all features operate on the same uniform scale.

### 2. The Covariance Matrix
*   **Action:** We compute the covariance matrix to measure how every single feature varies in relation to every other feature in the dataset. Because the dataset has 30 features, this results in a symmetric 30x30 matrix.

### 3. Eigendecomposition
*   **Action:** We extract the **Eigenvalues** and **Eigenvectors** from the covariance matrix using NumPy's highly optimized `np.linalg.eigh` function (specifically chosen because our covariance matrix is perfectly symmetric/Hermitian).
    *   **Eigenvectors:** The new "directions" (axes) of our data.
    *   **Eigenvalues:** The magnitude of variance (information) captured in each direction.

### 4. Sorting & Projection
*   **Action:** The eigenvectors are sorted in descending order based on their eigenvalues. We slice the top 7 vectors and compute the dot product with our standardized data, effectively projecting the 30-dimensional dataset into a highly optimized 7-dimensional subspace.

---

## Visualization: The Scree Plot

To understand how much information is retained after dropping 23 dimensions, the script calculates the **Explained Variance Ratio** and generates a Scree Plot. 

![alt text](<Figure 2026-11-14 013758.png>)

*   **Blue Bars (Individual Variance):** Shows the percentage of total dataset variance captured by each individual principal component.
*   **Red Line (Cumulative Variance):** Shows the running total of retained information. As seen in the plot, keeping just the first few components is enough to retain the vast majority of the original dataset's diagnostic patterns.

---

## How to Run

### Prerequisites
*   Python 3.x
*   NumPy
*   Pandas
*   Matplotlib

### Execution
Ensure that your `data.csv` (Breast Cancer Wisconsin Dataset) is located in the same directory as the script, then run:

```bash
python pca_from_scratch.py
```