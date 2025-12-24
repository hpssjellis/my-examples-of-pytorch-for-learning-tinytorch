import torch
import torch.nn as nn
import cv2
import random
import os

# torch10.py: Persistent S3-Brain with Save/Load
# This version "remembers" its training across sessions.

class myS3Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.myConv1 = nn.Conv2d(3, 8, kernel_size=5, stride=4, padding=2)
        self.myConv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.myHidden = nn.Linear(25600, 32)
        self.myActivation = nn.ReLU()
        self.myOutput = nn.Linear(32, 1)
        self.mySigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
            
        x = self.myActivation(self.myConv1(x))
        x = self.myActivation(self.myConv2(x))
        x = x.view(x.size(0), -1) 
        x = self.myActivation(self.myHidden(x))
        return self.mySigmoid(self.myOutput(x))

# --- Setup ---
myModel = myS3Brain()
myOptimizer = torch.optim.Adam(myModel.parameters(), lr=0.0005)
myLossFunc = nn.BCELoss()
myWeightFile = "torch10-Hand-Model.pth"

# NEW: Check for existing memory
if os.path.exists(myWeightFile):
    myModel.load_state_dict(torch.load(myWeightFile))
    print(f"--- Loaded existing brain from {myWeightFile} ---")
else:
    print("--- Starting with a fresh, empty brain ---")

myHandFrames = []
myBackFrames = []
myMaxMemory = 15 

myCap = cv2.VideoCapture(0)

while True:
    myRet, myFrame = myCap.read()
    if not myRet: break
    myFrame = cv2.resize(myFrame, (320, 320))
    myInput = torch.from_numpy(myFrame).float() / 255.0

    with torch.no_grad():
        myProb = myModel(myInput).item()

    myKey = cv2.waitKey(10) & 0xFF
    
    # 1. COLLECTION
    myLearnedSomething = False
    if myKey == ord('b'):
        myBackFrames.append(myInput)
        if len(myBackFrames) > myMaxMemory: myBackFrames.pop(0)
        myLearnedSomething = True
    elif myKey == ord('h'):
        myHandFrames.append(myInput)
        if len(myHandFrames) > myMaxMemory: myHandFrames.pop(0)
        myLearnedSomething = True

    # 2. TRAINING (If we have both data types)
    if myLearnedSomething and len(myHandFrames) > 2 and len(myBackFrames) > 2:
        h_sample = random.choice(myHandFrames)
        b_sample = random.choice(myBackFrames)
        
        myOptimizer.zero_grad()
        loss = myLossFunc(myModel(h_sample), torch.tensor([[1.0]])) + \
               myLossFunc(myModel(b_sample), torch.tensor([[0.0]]))
        loss.backward()
        myOptimizer.step()
        
        # SAVE THE BRAIN: This ensures we don't lose progress if it crashes
        torch.save(myModel.state_dict(), myWeightFile)
        print(f"Brain saved to {myWeightFile} | Loss: {loss.item():.4f}")

    # 3. VISUALS
    myLabel = "HAND" if myProb > 0.5 else "BACKGRND"
    myColor = (0, 255, 0) if myProb > 0.5 else (0, 0, 255)
    cv2.putText(myFrame, f"{myLabel} {myProb:.2f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, myColor, 2)
    cv2.imshow('S3 Persistent Trainer', myFrame)

    if myKey == ord('q'): break

myCap.release()
cv2.destroyAllWindows()