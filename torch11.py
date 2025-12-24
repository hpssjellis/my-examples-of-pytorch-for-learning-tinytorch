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
myWeightFile = "torch11-Multi-Model.pth"

if os.path.exists(myWeightFile):
    myModel.load_state_dict(torch.load(myWeightFile))
    print("--- Memory Loaded ---")

myBuffers = {0: [], 1: [], 2: []} 
myClassNames = {0: "BACKGROUND", 1: "HAND", 2: "OBJECT"}
myMaxMemory = 15 

myCap = cv2.VideoCapture(0)

print("\n--- INSTRUCTIONS ---")
print("1. Capture ALL THREE to start training: B, H, and P")
print("2. Press 'S' to Save the brain manually")
print("3. Press 'Q' to Save and Quit\n")

while True:
    myRet, myFrame = myCap.read()
    if not myRet: break
    myFrame = cv2.resize(myFrame, (320, 320))
    myInput = torch.from_numpy(myFrame).float() / 255.0

    with torch.no_grad():
        myLogits = myModel(myInput)
        myProbs = torch.softmax(myLogits, dim=1)
        myConf, myClassID = torch.max(myProbs, 1)
        myClassID = myClassID.item()
        myConf = myConf.item()

    myRawKey = cv2.waitKey(10) & 0xFF
    myKey = chr(myRawKey).lower() if myRawKey < 256 else ""
    
    # 1. COLLECTION
    myTargetID = None
    if myKey == 'b': myTargetID = 0
    elif myKey == 'h': myTargetID = 1
    elif myKey == 'p': myTargetID = 2

    if myTargetID is not None:
        myBuffers[myTargetID].append(myInput)
        if len(myBuffers[myTargetID]) > myMaxMemory: myBuffers[myTargetID].pop(0)
        print(f"Captured {myClassNames[myTargetID]}. Buffers: B:{len(myBuffers[0])} H:{len(myBuffers[1])} P:{len(myBuffers[2])}")

    # 2. TRAINING (The "Stall" Fix: providing feedback if not ready)
    if all(len(myBuffers[i]) > 0 for i in range(3)):
        samples = [random.choice(myBuffers[i]) for i in range(3)]
        labels = torch.tensor([0, 1, 2])
        batch_input = torch.stack(samples)
        
        myOptimizer.zero_grad()
        loss = myLossFunc(myModel(batch_input), labels)
        loss.backward()
        myOptimizer.step()
    elif myTargetID is not None:
        print("Waiting for all 3 classes to be captured before learning...")

    # 3. MANUAL SAVE
    if myKey == 's':
        torch.save(myModel.state_dict(), myWeightFile)
        print(f"!!! Brain Manually Saved to {myWeightFile} !!!")

    # 4. VISUALS
    myLabel = f"{myClassNames[myClassID]} ({myConf:.2f})"
    myColor = (255, 0, 0) if myClassID == 0 else (0, 255, 0) if myClassID == 1 else (0, 255, 255)
    
    cv2.putText(myFrame, myLabel, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, myColor, 2)
    cv2.putText(myFrame, "B|H|P=Train  S=Save  Q=Quit", (10, 310), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('S3 Multi-Gesture Trainer', myFrame)

    if myKey == 'q':
        torch.save(myModel.state_dict(), myWeightFile)
        print("Saving and Exiting...")
        break

myCap.release()
cv2.destroyAllWindows()