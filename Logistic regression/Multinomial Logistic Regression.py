# -*- coding: utf-8 -*-
import math

# Data: (Hours Studied, Category Index)
# Category 0 = Fail, Category 1 = Pass, Category 2 = A+
data = [(1, 0), (2, 0), (4, 1), (5, 1), (7, 2), (8, 2)]

# We now need a weight and bias for EACH of the 3 categories
# w[0] is for Fail, w[1] is for Pass, w[2] is for A+
w = [0.0, 0.0, 0.0]
b = [0.0, 0.0, 0.0]
lr = 0.1

# Upgrade 1: Softmax replaces Sigmoid
def softmax(z_scores):
    # Calculate e^z for every score
    exponentials = [math.exp(z) for z in z_scores]
    total_sum = sum(exponentials)
    
    # Divide each by the total to get probabilities that add up to 1.0
    probabilities = [e / total_sum for e in exponentials]
    return probabilities

# Upgrade 2: Log Loss becomes Cross-Entropy Loss
def calculate_cross_entropy(predicted_probability_of_correct_class):
    # We only penalize based on the probability assigned to the ACTUAL correct answer
    p = max(min(predicted_probability_of_correct_class, 1 - 1e-15), 1e-15) 
    return -math.log(p)

for epoch in range(1000):
    total_loss = 0 
    
    for x, y_actual in data:
        # Step A: Predict
        # Calculate a raw score (z) for all 3 categories
        z_scores = [
            w[0] * x + b[0],
            w[1] * x + b[1],
            w[2] * x + b[2]
        ]
        
        # Turn raw scores into percentages [Fail%, Pass%, A+%]
        y_pred_probs = softmax(z_scores)
        
        # Monitor: Add the loss for the correct category
        prob_of_correct_answer = y_pred_probs[y_actual]
        total_loss += calculate_cross_entropy(prob_of_correct_answer)
        
        # Step B & C: Update the weights for ALL 3 categories
        for category_index in range(3):
            # Is this category the actual correct answer? (1 if yes, 0 if no)
            target = 1 if category_index == y_actual else 0
            
            # The beautiful simplified derivative works here too!
            # It's always: Predicted_Probability - Actual_Target (1 or 0)
            error = y_pred_probs[category_index] - target
            
            w[category_index] = w[category_index] - (lr * error * x)
            b[category_index] = b[category_index] - (lr * error)
            
    if epoch % 200 == 0:
        print(f"Epoch {epoch} | Total Cross-Entropy Loss: {total_loss:.4f}")

# ---------------------------------------------------------
# Let's test the trained model on a new student
# ---------------------------------------------------------
print("\n--- Training Complete ---\n")

test_hours = 6
category_names = ["Fail", "Pass", "A+"]

# Calculate scores for the new student
test_z = [w[0] * test_hours + b[0], 
          w[1] * test_hours + b[1], 
          w[2] * test_hours + b[2]]

final_probabilities = softmax(test_z)

# Find the category with the highest probability
winning_index = final_probabilities.index(max(final_probabilities))
winning_category = category_names[winning_index]

print(f"For {test_hours} hours of study:")
print(f" - Fail probability: {final_probabilities[0]:.2%}")
print(f" - Pass probability: {final_probabilities[1]:.2%}")
print(f" - A+ probability:   {final_probabilities[2]:.2%}")
print(f" - Final Decision:   {winning_category}")