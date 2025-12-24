import torch
import torch.nn as nn
import cv2
import random
import os

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
        x = self.myActivation(self.myConv1(x))
        x = self.myActivation(self.myConv2(x))
        x = x.reshape(x.size(0), -1) 
        x = self.myActivation(self.myHidden(x))
        return self.myOutput(x)

# --- Setup ---
myModel = myS3Brain()
myOptimizer = torch.optim.Adam(myModel.parameters(), lr=0.0005)
myLossFunc = nn.CrossEntropyLoss()
myWeightFile = "torch13-Validated-Model.pth"

# Training vs Validation Buffers
myTrainData = {0: [], 1: [], 2: []}
myValData = {0: [], 1: [], 2: []}
myClassNames = {0: "BACKGROUND", 1: "HAND", 2: "OBJECT"}

myCap = cv2.VideoCapture(0)
myValAccuracy = 0.0

print("--- Step 13: Validation (The Final Exam) ---")

while True:
    myRet, myFrame = myCap.read()
    if not myRet: break
    myFrame = cv2.resize(myFrame, (320, 320))
    myInput = torch.from_numpy(myFrame).float() / 255.0

    with torch.no_grad():
        myLogits = myModel(myInput)
        myClassID = torch.argmax(myLogits, 1).item()

    myRawKey = cv2.waitKey(10) & 0xFF
    myKey = chr(myRawKey).lower() if myRawKey < 256 else ""
    
    myTargetID = None
    if myKey == 'b': myTargetID = 0
    elif myKey == 'h': myTargetID = 1
    elif myKey == 'p': myTargetID = 2

    if myTargetID is not None:
        # 80/20 Split: 20% of the time, put data in the "Exam" buffer
        if random.random() < 0.2:
            myValData[myTargetID].append(myInput)
            if len(myValData[myTargetID]) > 10: myValData[myTargetID].pop(0)
            print(f"Added to VALIDATION set: {myClassNames[myTargetID]}")
        else:
            myTrainData[myTargetID].append(myInput)
            if len(myTrainData[myTargetID]) > 15: myTrainData[myTargetID].pop(0)

        # TRAINING STEP
        if all(len(myTrainData[i]) > 0 for i in range(3)):
            batch_input = torch.stack([random.choice(myTrainData[i]) for i in range(3)])
            myOptimizer.zero_grad()
            loss = myLossFunc(myModel(batch_input), torch.tensor([0, 1, 2]))
            loss.backward()
            myOptimizer.step()

        # VALIDATION STEP (The Exam)
        if all(len(myValData[i]) > 0 for i in range(3)):
            myModel.eval() # Set to evaluation mode
            with torch.no_grad():
                val_input = torch.stack([random.choice(myValData[i]) for i in range(3)])
                val_out = myModel(val_input)
                val_preds = torch.argmax(val_out, 1)
                # Compare predictions to truth [0, 1, 2]
                correct = (val_preds == torch.tensor([0, 1, 2])).sum().item()
                myValAccuracy = (correct / 3.0) * 100
            myModel.train() # Set back to training mode

    # Visuals
    myColor = (255, 0, 0) if myClassID == 0 else (0, 255, 0) if myClassID == 1 else (0, 255, 255)
    cv2.putText(myFrame, f"{myClassNames[myClassID]}", (10, 35), 1, 2, myColor, 2)
    # New: Display the "Exam Score"
    cv2.putText(myFrame, f"Exam Score: {myValAccuracy:.1f}%", (10, 70), 1, 1.2, (255, 255, 255), 2)
    cv2.imshow('S3 Validation Trainer', myFrame)

    if myKey == 's' or myKey == 'q':
        torch.save(myModel.state_dict(), myWeightFile)
        if myKey == 'q': break

myCap.release()
cv2.destroyAllWindows()