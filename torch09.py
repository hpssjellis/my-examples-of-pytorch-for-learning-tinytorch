import torch
import torch.nn as nn
import cv2
import random

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
        # Handle both single images and batches
        if x.dim() == 3:
            x = x.permute(2, 0, 1).unsqueeze(0)
        elif x.dim() == 4:
            x = x.permute(0, 3, 1, 2)
            
        x = self.myActivation(self.myConv1(x))
        x = self.myActivation(self.myConv2(x))
        x = x.view(x.size(0), -1) 
        x = self.myActivation(self.myHidden(x))
        return self.mySigmoid(self.myOutput(x))

myModel = myS3Brain()
# Lower learning rate (0.0005) helps prevent "slamming" the weights
myOptimizer = torch.optim.Adam(myModel.parameters(), lr=0.0005)
myLossFunc = nn.BCELoss()

# Memory Buffers
myHandFrames = []
myBackFrames = []
myMaxMemory = 15 # Store 15 examples of each

myCap = cv2.VideoCapture(0)
print("--- Buffered S3 Training ---")
print("Collect ~10 of each! 'b'=Background, 'h'=Hand, 'q'=Quit")

while True:
    myRet, myFrame = myCap.read()
    if not myRet: break
    myFrame = cv2.resize(myFrame, (320, 320))
    myInput = torch.from_numpy(myFrame).float() / 255.0

    with torch.no_grad():
        myProb = myModel(myInput).item()

    myKey = cv2.waitKey(10) & 0xFF
    
    # COLLECTION LOGIC
    if myKey == ord('b'):
        myBackFrames.append(myInput)
        if len(myBackFrames) > myMaxMemory: myBackFrames.pop(0)
        print(f"Stored Background ({len(myBackFrames)}/{myMaxMemory})")
    elif myKey == ord('h'):
        myHandFrames.append(myInput)
        if len(myHandFrames) > myMaxMemory: myHandFrames.pop(0)
        print(f"Stored Hand ({len(myHandFrames)}/{myMaxMemory})")

    # TRAINING LOGIC (Only if we have examples of both)
    if len(myHandFrames) > 2 and len(myBackFrames) > 2:
        # Create a small batch for training
        h_sample = random.choice(myHandFrames)
        b_sample = random.choice(myBackFrames)
        
        # Train on Hand
        p_h = myModel(h_sample)
        loss_h = myLossFunc(p_h, torch.tensor([[1.0]]))
        
        # Train on Background
        p_b = myModel(b_sample)
        loss_b = myLossFunc(p_b, torch.tensor([[0.0]]))
        
        myOptimizer.zero_grad()
        (loss_h + loss_b).backward()
        myOptimizer.step()

    # Visuals
    myLabel = "HAND" if myProb > 0.5 else "BACKGRND"
    myColor = (0, 255, 0) if myProb > 0.5 else (0, 0, 255)
    cv2.putText(myFrame, f"{myLabel} {myProb:.2f}", (10, 30), 1, 2, myColor, 2)
    cv2.imshow('S3 Buffered Trainer', myFrame)

    if myKey == ord('q'): break

myCap.release()
cv2.destroyAllWindows()