import torch

# torch02.py: Understanding Autograd (Automatic Differentiation)
# This script shows how PyTorch tracks math to calculate a slope (gradient).

print("--- Step 1: Initialization ---")
# We set requires_grad=True so PyTorch starts a "ledger" for this variable.
# This is the "Engine" behind how Neural Networks learn.
myInput = torch.tensor([3.0], requires_grad=True)

print(f"Created myInput: {myInput.item()}")
print(f"Initial Gradient: {myInput.grad} (Empty because no math has happened yet)")

print("\n--- Step 2: The Forward Pass ---")
# Equation: y = x^2 + 5
# In JS: const myY = Math.pow(myInput, 2) + 5;
myY = myInput**2 + 5
print(f"Calculated myY (3^2 + 5): {myY.item()}")

print("\n--- Step 3: The Backward Pass ---")
# This is the "Magic" step. It looks at the equation above and 
# calculates the derivative (the slope) automatically.
myY.backward()
print("Running .backward()... Done.")

print("\n--- Step 4: Final Results ---")
# The derivative of x^2 + 5 is 2x. 
# Since our input (x) is 3, the slope is 2 * 3 = 6.
mySlope = myInput.grad.item()
myStepSize = 0.01  # This is like a "Learning Rate" in ML

print(f"The gradient (slope) at myInput={myInput.item()} is: {mySlope}")

# Demonstrating how the slope predicts the future:
myPredictedChange = mySlope * myStepSize
print(f"Interpretation: If we move myInput by {myStepSize},")
print(f"myY should increase by approximately {myPredictedChange:.4f}")

# Verification:
myNewInput = myInput.item() + myStepSize
myNewY = (myNewInput**2) + 5
print(f"Actual new myY: {myNewY:.4f} (Prediction was very close!)")