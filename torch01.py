import torch

# 1. Create data: Input (X) and Target (Y)
myInput = torch.tensor([[1.0], [2.0], [3.0]])
myTarget = torch.tensor([[2.0], [4.0], [6.0]]) # The rule is: Target = Input * 2

# 2. Define a simple linear model (1 input -> 1 output)
myModel = torch.nn.Linear(1, 1)

# 3. Define Optimizer and Loss Function
myOptimizer = torch.optim.SGD(myModel.parameters(), lr=0.01)
myLossFunction = torch.nn.MSELoss()

# 4. Training Loop
for myEpoch in range(100):
    myPrediction = myModel(myInput)           # Forward pass
    myLoss = myLossFunction(myPrediction, myTarget) # Calculate error
    
    myOptimizer.zero_grad()                   # Clear old gradients
    myLoss.backward()                         # Backpropagation
    myOptimizer.step()                        # Update weights

# 5. Test the model
print(f"Prediction for 10: {myModel(torch.tensor([10.0])).item():.2f}")