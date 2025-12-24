import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. The Blueprint (Brain structure)
class myS3Brain(nn.Module):
    def __init__(self):
        super().__init__()
        self.myConv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.myConv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.myPool = nn.MaxPool2d(2, 2)
        self.myHidden = nn.Linear(32 * 7 * 7, 64)
        self.myOutput = nn.Linear(64, 10) # 10 digits
        self.myActivation = nn.ReLU()

    def forward(self, x):
        x = self.myPool(self.myActivation(self.myConv1(x)))
        x = self.myPool(self.myActivation(self.myConv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.myActivation(self.myHidden(x))
        return self.myOutput(x)

# 2. Setup Data
myTransform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
myTrainData = datasets.MNIST(root='./data', train=True, download=True, transform=myTransform)
myTrainLoader = torch.utils.data.DataLoader(myTrainData, batch_size=64, shuffle=True)

myModel = myS3Brain()
myOptimizer = optim.Adam(myModel.parameters(), lr=0.001)
myLossFunc = nn.CrossEntropyLoss()

# 3. Training Loop
print("🚀 Training starting... this might take a minute.")
myModel.train()
for myBatchIdx, (myData, myTarget) in enumerate(myTrainLoader):
    myOptimizer.zero_grad()
    myResult = myModel(myData)
    myLoss = myLossFunc(myResult, myTarget)
    myLoss.backward()
    myOptimizer.step()
    
    if myBatchIdx % 200 == 0:
        print(f"Progress: {myBatchIdx}/{len(myTrainLoader)} batches...")

# --- THE FIX: SAVE THE WEIGHTS ---
torch.save(myModel.state_dict(), 'mnist_weights.pth')
print("✅ Success! Weights saved as 'mnist_weights.pth'")