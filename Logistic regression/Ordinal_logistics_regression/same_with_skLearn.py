import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import mord

# 1. Load Data
file_path = r"C:\Users\Kali\OneDrive - purdue.edu\Classes\ML\ML Jourey\Logistic regression\Ordinal_logistics_regression\WineQT.csv"
wine_df = pd.read_csv(file_path)
wine_df = wine_df.drop(columns=["Id"])

# 2. Separate features (X) and target label (y)
X = wine_df.drop('quality', axis=1)
y = wine_df['quality']

# 3. Split 80% training / 20% testing
# random_state=42 ensures we get the exact same shuffle every time
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()

# .fit_transform() learns the stats from the training data AND scales it
X_train_scaled = scaler.fit_transform(X_train)

# .transform() scales the test data using the stats it learned above
X_test_scaled = scaler.transform(X_test)

# Create the model with an L2 penalty
model = mord.LogisticAT(alpha=1.0)

# Train the model (this does all 20,000 epochs of math instantly)
model.fit(X_train_scaled, y_train)

# You can actually look at the weights and thresholds it found!
print("Calculated Weights:", model.coef_)
print("Calculated Thresholds:", model.theta_)

# 1. Make predictions on the hidden test set
predictions = model.predict(X_test_scaled)

# 2. Calculate Strict Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"\nOverall Strict Accuracy: {accuracy * 100:.2f}%")

# 3. Calculate "Off-by-One" Accuracy manually
close_guesses = sum(abs(p - a) <= 1 for p, a in zip(predictions, y_test))
close_accuracy = close_guesses / len(y_test)
print(f"Off-by-One Accuracy: {close_accuracy * 100:.2f}%\n")

# 4. Print the Confusion Matrix
print("--- Confusion Matrix ---")

# We grab the unique categories (3, 4, 5, 6, 7, 8) so the matrix is labeled properly
labels = sorted(y.unique())
matrix = confusion_matrix(y_test, predictions, labels=labels)

# Convert it to a pandas dataframe just so it prints prettily!
matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
matrix_df.index.name = 'Actual'
matrix_df.columns.name = 'Predicted'

print(matrix_df)