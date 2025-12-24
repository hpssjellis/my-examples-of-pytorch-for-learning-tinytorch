import torch

# torch04.py: Learning a Linear Relationship (y = W*x + b)
# Goal: Find the 'Weight' and 'Bias' that turns [1, 2, 3] into [3, 5, 7]
# (Hint: The hidden rule is y = 2x + 1)

print("--- Step 1: Setup Data ---")
myInputs = torch.tensor([[1.0], [2.0], [3.0]])
myTargets = torch.tensor([[3.0], [5.0], [7.0]])

# We start with random guesses for Weight and Bias
myWeight = torch.tensor([0.5], requires_grad=True)
myBias = torch.tensor([0.0], requires_grad=True)
myLearningRate = 0.05

print(f"Starting Weight: {myWeight.item():.2f}, Bias: {myBias.item():.2f}\n")

print("--- Step 2: Training ---")
for myEpoch in range(201):
    # Forward Pass: Predict Y for all inputs at once
    # In tinyTorch, this is a "Linear Layer"
    myPredictions = myInputs * myWeight + myBias
    
    # Calculate Mean Squared Error (Average of all errors)
    myLoss = torch.mean((myPredictions - myTargets)**2)
    
    # Backward Pass
    myLoss.backward()
    
    # Update Weight and Bias
    with torch.no_grad():
        myWeight -= myLearningRate * myWeight.grad
        myBias -= myLearningRate * myBias.grad
        
    # Zero gradients
    myWeight.grad.zero_()
    myBias.grad.zero_()
    
    if myEpoch % 50 == 0:
        print(f"Epoch {myEpoch}: Loss {myLoss.item():.4f} | W: {myWeight.item():.2f} b: {myBias.item():.2f}")

print("\n--- Step 3: Result ---")
print(f"Final Rule: y = {myWeight.item():.2f}x + {myBias.item():.2f}")

# Test it with a new number (Note 10)
myTest = torch.tensor([10.0])
myResult = myTest * myWeight + myBias
print(f"If input is 10, prediction is: {myResult.item():.2f} (Should be 21)")