import torch
import torch.nn as nn

# torch05.py: Using Modules and Optimizers
# This is the "Professional" structure used in both PyTorch and tinyTorch.

# 1. Define the Model Class
# In tinyTorch, your layers will look exactly like this!
class mySimpleBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Linear(input_size, output_size)
        # This automatically creates 'weight' and 'bias' for us.
        self.myLayer = nn.Linear(1, 1)

    def forward(self, x):
        # This defines how data flows through the brain
        return self.myLayer(x)

# 2. Setup Data
myInputs = torch.tensor([[1.0], [2.0], [3.0]])
myTargets = torch.tensor([[3.0], [5.0], [7.0]])

# 3. Initialize the Model, Optimizer, and Loss Function
myModel = mySimpleBrain()
# The Optimizer (SGD) automatically handles the weight updates for us
myOptimizer = torch.optim.SGD(myModel.parameters(), lr=0.05)
myLossFunction = nn.MSELoss()

print("--- Step 1: Initial State ---")
for myName, myParam in myModel.named_parameters():
    print(f"{myName}: {myParam.data}")

print("\n--- Step 2: Training ---")
for myEpoch in range(201):
    # Forward Pass
    myPredictions = myModel(myInputs)
    
    # Calculate Loss
    myLoss = myLossFunction(myPredictions, myTargets)
    
    # Backward Pass (Calculates gradients)
    myLoss.backward()
    
    # The Optimizer does the math we did manually in torch04
    myOptimizer.step()
    
    # Clear gradients for the next turn
    myOptimizer.zero_grad()
    
    if myEpoch % 50 == 0:
        print(f"Epoch {myEpoch}: Loss {myLoss.item():.4f}")

print("\n--- Step 3: Final State ---")
for myName, myParam in myModel.named_parameters():
    print(f"{myName}: {myParam.data}")

# Test the brain
myTestNote = torch.tensor([[10.0]])
print(f"\nPrediction for input 10: {myModel(myTestNote).item():.2f}")