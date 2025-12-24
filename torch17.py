import torch
import torch.nn as nn
import cv2
import numpy as np
import os

# --- 1. The Brain Architecture (Must match torch16.py) ---
class myS3Brain(nn.Module):
    def __init__(self):
        super().__init__()
        # MNIST is 1 channel (grayscale)
        self.myConv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.myConv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.myPool = nn.MaxPool2d(2, 2)
        # 28x28 -> 14x14 -> 7x7
        self.myHidden = nn.Linear(32 * 7 * 7, 64)
        self.myOutput = nn.Linear(64, 10) # 0-9 digits
        self.myActivation = nn.ReLU()

    def forward(self, x):
        x = self.myPool(self.myActivation(self.myConv1(x)))
        x = self.myPool(self.myActivation(self.myConv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.myActivation(self.myHidden(x))
        return self.myOutput(x)

# --- 2. Setup and Loading ---
myModel = myS3Brain()
myWeightFile = 'mnist_weights.pth'

if os.path.exists(myWeightFile):
    myModel.load_state_dict(torch.load(myWeightFile))
    myModel.eval() # Set to expert mode
    print(f"🧠 Brain Loaded: {myWeightFile}")
else:
    print("⚠️ Warning: No weights found! The AI will just guess randomly.")
    print("Run torch16.py first to train the model.")

myCap = cv2.VideoCapture(0)

print("--- torch17.py: Live Digit Reader ---")
print("Place your handwritten digit inside the green box.")
print("Press 'Q' to quit.")

while True:
    myRet, myFrame = myCap.read()
    if not myRet: break
    
    # Create a 200x200 "Scanning Zone" in the center of the screen
    myH, myW, _ = myFrame.shape
    myX1, myY1 = (myW // 2 - 100), (myH // 2 - 100)
    myX2, myY2 = (myW // 2 + 100), (myH // 2 + 100)
    
    # --- 3. THE IMAGE BRIDGE (Pre-processing) ---
    # Crop the box
    myROI = myFrame[myY1:myY2, myX1:myX2]
    
    # Convert to Grayscale
    myGray = cv2.cvtColor(myROI, cv2.COLOR_BGR2GRAY)
    
    # Clean up noise
    myGray = cv2.GaussianBlur(myGray, (5, 5), 0)
    
    # FLIP COLORS: Convert black ink on white paper -> white ink on black background
    # This uses Adaptive Thresholding to handle shadows and different lighting
    myBinary = cv2.adaptiveThreshold(myGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY_INV, 11, 2)
    
    # Shrink to the 28x28 size the brain understands
    mySmall = cv2.resize(myBinary, (28, 28))
    
    # Convert to Tensor (Add batch and channel dimensions: 1x1x28x28)
    myInput = torch.from_numpy(mySmall).float().unsqueeze(0).unsqueeze(0) / 255.0

    # --- 4. PREDICTION ---
    with torch.no_grad():
        myLogits = myModel(myInput)
        myPrediction = torch.argmax(myLogits, 1).item()
        myConfidence = torch.softmax(myLogits, dim=1)[0][myPrediction].item()

    # --- 5. VISUALS ---
    # Draw the scanning box
    cv2.rectangle(myFrame, (myX1, myY1), (myX2, myY2), (0, 255, 0), 2)
    
    # Display the result
    myText = f"Digit: {myPrediction} ({myConfidence*100:.1f}%)"
    cv2.putText(myFrame, myText, (myX1, myY1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show the "Brain View" (The small 28x28 input zoomed in)
    myBrainView = cv2.resize(myBinary, (160, 160))
    cv2.imshow("What the AI Sees", myBrainView)
    cv2.imshow("Main Camera Feed", myFrame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

myCap.release()
cv2.destroyAllWindows()