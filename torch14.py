import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import random

class myS3Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.myConv1 = nn.Conv2d(3, 8, kernel_size=5, stride=4, padding=2)
        self.myConv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.myHidden = nn.Linear(25600, 32)
        self.myActivation = nn.ReLU()
        self.myOutput = nn.Linear(32, 3) 

    def forward(self, x):
        if x.dim() == 3: x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4: x = x.permute(0, 3, 1, 2)
        myFeatures = self.myActivation(self.myConv1(x))
        x = self.myActivation(self.myConv2(myFeatures))
        x = x.reshape(x.size(0), -1) 
        x = self.myActivation(self.myHidden(x))
        return self.myOutput(x), myFeatures

# --- Setup ---
myModel = myS3Brain()
myWeightFile = "torch13-Validated-Model.pth"
if os.path.exists(myWeightFile):
    myModel.load_state_dict(torch.load(myWeightFile))

myClassNames = {0: "BACKGROUND", 1: "HAND", 2: "OBJECT"}
myCap = cv2.VideoCapture(0)
myIntensity = 1.0 # Starting brightness multiplier

print("--- Step 14: Interactive X-Ray ---")
print("CONTROLS:")
print("UP ARROW: Increase Eye Intensity")
print("DOWN ARROW: Decrease Eye Intensity")
print("B, H, P: Train live")
print("Q: Quit")

while True:
    myRet, myFrame = myCap.read()
    if not myRet: break
    myFrame = cv2.resize(myFrame, (320, 320))
    myInput = torch.from_numpy(myFrame).float() / 255.0

    with torch.no_grad():
        myLogits, myFeatures = myModel(myInput)
        myClassID = torch.argmax(myLogits, 1).item()

    # --- INPUT CONTROLS ---
    myKey = cv2.waitKeyEx(10) # Using waitKeyEx for Arrow Keys
    
    # 2490368 is UP, 2621440 is DOWN (Windows/OpenCV standard)
    # Checking against common codes, also mapping standard char keys
    if myKey == 2490368 or myKey == 38: # Up Arrow
        myIntensity += 0.5
    elif myKey == 2621440 or myKey == 40: # Down Arrow
        myIntensity = max(0.1, myIntensity - 0.5)
    
    myChar = chr(myKey & 0xFF).lower() if (myKey & 0xFF) < 256 else ""
    if myChar == 'q': break

    # --- GENERATE FEATURE GRID ---
    myFeatureList = []
    for i in range(8):
        f = myFeatures[0, i].detach().cpu().numpy()
        f = f * myIntensity # Apply the interactive intensity
        f = cv2.normalize(f, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        f = cv2.resize(f, (160, 160))
        # Add Label to each filter
        cv2.putText(f, f"Filter {i}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        myFeatureList.append(f)

    top_row = np.hstack(myFeatureList[0:4])
    bot_row = np.hstack(myFeatureList[4:8])
    myFullGrid = np.vstack([top_row, bot_row])

    # --- UI DISPLAY ---
    # Put class name and intensity on the main camera feed
    cv2.putText(myFrame, f"Target: {myClassNames[myClassID]}", (10, 30), 1, 1.5, (0, 255, 0), 2)
    cv2.putText(myFrame, f"Eye Intensity: {myIntensity:.1f}x", (10, 310), 1, 1.0, (255, 255, 255), 1)

    cv2.imshow('Teacher View (Input)', myFrame)
    cv2.imshow('ESP32-S3 View (Internal Filters)', myFullGrid)

myCap.release()
cv2.destroyAllWindows()