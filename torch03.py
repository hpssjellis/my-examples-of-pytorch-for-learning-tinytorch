import torch

# torch03.py: Smooth Gradient Descent
myInput = torch.tensor([2.0], requires_grad=True)
myTarget = 25.0
myLearningRate = 0.01 

print(f"Target: {myTarget} | Starting Guess: {myInput.item()}\n")

for myEpoch in range(1001):
    # 1. Forward Pass
    myResult = myInput**2
    
    # 2. Calculate Loss (Using Absolute Error for smoothness)
    # In JS: Math.abs(myResult - myTarget)
    myLoss = torch.abs(myResult - myTarget)
    
    # 3. Backward Pass & Update
    myLoss.backward()
    with torch.no_grad():
        myInput -= myLearningRate * myInput.grad
    myInput.grad.zero_()

    # 4. Print progress (Now the print happens AFTER the update)
    if myEpoch % 200 == 0:
        # We recalculate the square here to show the status of the NEW guess
        current_square = myInput.item()**2
        print(f"Epoch {myEpoch}: Guess is {myInput.item():.4f}, Square is {current_square:.4f}")

# Store the final state clearly
final_guess = myInput.item()
final_square = final_guess**2

print("\n--- Final Results ---")
print(f"Final Guess: {final_guess:.4f}")
print(f"Final Square: {final_square:.4f}")