import torch
import torch.nn as nn

# torch06.py: Non-Linearity and Hidden Layers
# Goal: Learn a pattern that isn't a straight line.

# 1. Define a more complex Model
class myDeepBrain(nn.Module):
    def __init__(self):
        super().__init__()
        # We now have two layers!
        # Layer 1: Takes 1 input and expands it to 5 "neurons"
        self.myHiddenLayer = nn.Linear(1, 5)
        # Activation: The "switch" that makes it non-linear
        self.myActivation = nn.ReLU()
        # Layer 2: Shrinks the 5 neurons back down to 1 output
        self.myOutputLayer = nn.Linear(5, 1)

    def forward(self, x):
        myVar = self.myHiddenLayer(x)
        myVar = self.myActivation(myVar)
        myVar = self.myOutputLayer(myVar)
        return myVar

# 2. Setup Data: A pattern that isn't a simple multiply (e.g., Absolute value)
myInputs = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
myTargets = torch.tensor([[2.0], [1.0], [0.0], [1.0], [2.0]]) # Target is |x|

# 3. Setup Model, Optimizer, and Loss
myModel = myDeepBrain()
myOptimizer = torch.optim.Adam(myModel.parameters(), lr=0.01) # Adam is a smarter version of SGD
myLossFunction = nn.MSELoss()

print("--- Training a Deep Brain ---")
for myEpoch in range(501):
    myPredictions = myModel(myInputs)
    myLoss = myLossFunction(myPredictions, myTargets)
    
    myOptimizer.zero_grad()
    myLoss.backward()
    myOptimizer.step()
    
    if myEpoch % 100 == 0:
        print(f"Epoch {myEpoch}: Loss {myLoss.item():.4f}")

# 4. Test it
myTestValue = torch.tensor([[-1.5]])
print(f"\nPrediction for -1.5: {myModel(myTestValue).item():.2f} (Should be 1.50)")