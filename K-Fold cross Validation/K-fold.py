import numpy as np

# 1. Create Dummy Data: y = 2x + 5
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([7, 9, 11, 13, 15, 17, 19, 21, 23, 25])

# 2. Settings
K = 5
lr = 0.01
epochs = 500

# 3. Shuffle the data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# 4. Split into K folds
X_folds = np.array_split(X, K)
y_folds = np.array_split(y, K)

scores = []

# 5. The K-Fold Loop
for i in range(K):
    # a) Set aside the 'i-th' fold as the Validation/Test set
    X_val = X_folds[i]
    y_val = y_folds[i]
    
    # b) Use all OTHER folds for Training
    X_train = np.concatenate([X_folds[j] for j in range(K) if j != i])
    y_train = np.concatenate([y_folds[j] for j in range(K) if j != i])
    
    # c) Train a brand new model from scratch for this fold
    w, b = 0.0, 0.0
    for _ in range(epochs):
        for x_row, y_actual in zip(X_train, y_train):
            y_pred = w * x_row[0] + b
            error = y_pred - y_actual
            w -= lr * error * x_row[0]
            b -= lr * error
            
    # d) Test the model on the Validation set we set aside
    val_errors = []
    for x_row, y_actual in zip(X_val, y_val):
        y_pred = w * x_row[0] + b
        val_errors.append((y_pred - y_actual) ** 2)
    
    fold_mse = np.mean(val_errors)
    scores.append(fold_mse)
    print(f"Fold {i+1} MSE: {fold_mse:.4f}")

# 6. The Final Verdict
print("---")
print(f"Average Cross-Validation MSE: {np.mean(scores):.4f}")