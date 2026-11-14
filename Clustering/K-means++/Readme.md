# K-Means Clustering from Scratch

This project contains a custom Python implementation of the K-Means clustering algorithm built entirely from scratch using NumPy. It is designed to segment a retail dataset (`Mall_Customers.csv`) into distinct customer groups based on their age, gender, annual income, and spending score.

## Features

* **Custom K-Means Algorithm**: Built without relying on standard machine learning libraries like `scikit-learn` for the core algorithm.

* **K-Means++ Initialization**: Includes a custom implementation of the K-Means++ initialization method to ensure optimal starting positions for centroids, preventing the algorithm from falling into poor local minima.

* **Elbow Method & WCSS**: Contains a custom function to calculate the Within-Cluster Sum of Squares (WCSS). It automatically tests $k$ values from 1 to 10 and plots an Elbow graph to help you visually determine the optimal number of clusters.

  ![alt text](<WCSS graph-1.png>)

* **Data Visualization**: Utilizes `matplotlib` and `seaborn` to generate side-by-side comparative graphs showing how the optimal clusters map across different features (Age vs. Spending, Gender vs. Spending, Income vs. Spending), complete with centroid markers.

## Prerequisites

To run this script, you need Python installed on your machine along with the following standard data science libraries:

* `numpy` (for matrix math and distance calculations)

* `pandas` (for data loading and manipulation)

* `matplotlib` (for plotting graphs and centroids)

* `seaborn` (for color-coding cluster visualizations)

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn
```

## Dataset Requirement

The script requires a dataset named exactly `Mall_Customers.csv` to be located in the same directory as the Python script.

The dataset is expected to have at least the following columns:

* `CustomerID`

* `Gender` (Categorical: 'Male' or 'Female')

* `Age` (Numerical)

* `Annual Income (k$)` (Numerical)

* `Spending Score (1-100)` (Numerical)

## How to Run

1. Ensure `Mall_Customers.csv` is in your working directory.

2. Run the script using your IDE or the command line:

   ```bash
   python your_script_name.py
   ```

3. **First Output**: The script will calculate the WCSS for $k=1$ through $k=10$ and display the **Elbow Method line graph** shown above. *You must close this graph window for the script to continue running.*

4. **Second Output**: The script will automatically run the optimal clustering ($k=5$), print the first 10 rows of the newly labeled dataset to your console, and open a wide figure displaying three color-coded scatter/strip plots mapping your customer segments.

## Customization

* **Number of Clusters**: The final visualizations are currently hardcoded to $k=5$ based on the elbow method results. You can change this by modifying the `best_k = 5` variable in the script.

* **Max Iterations**: The default max iterations for the algorithm to converge is set to 100. You can adjust this by changing the `max_iters` parameter in the `K_means` function definition.