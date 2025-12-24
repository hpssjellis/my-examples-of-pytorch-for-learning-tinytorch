import torch
import cv2
import time

# torch08.py: Digital Tripwire (Bulletproof Quitting Version)
print("--- Step 1: Initialize Camera ---")
myCap = cv2.VideoCapture(0)

# Initial setup
myRet, myOldFrame = myCap.read()
if not myRet:
    print("Error: Could not access the webcam.")
    exit()

myOldGray = cv2.cvtColor(myOldFrame, cv2.COLOR_BGR2GRAY)
myOldTensor = torch.from_numpy(myOldGray).float() / 255.0

# higher myThreshold for less sensitive
myThreshold = 0.003     
myLastStatus = "STILL" 

# Create the window first so we can set its properties
cv2.namedWindow('Motion Sensor')

print("System Armed.")
print("CRITICAL: You must CLICK the video window before pressing 'q'.")

while True:
    myRet, myFrame = myCap.read()
    if not myRet: break

    # 1. Convert current frame to Tensor
    myGray = cv2.cvtColor(myFrame, cv2.COLOR_BGR2GRAY)
    myCurrentTensor = torch.from_numpy(myGray).float() / 255.0

    # 2. CALCULATE LOSS
    myDifference = torch.mean((myCurrentTensor - myOldTensor)**2)
    myLossValue = myDifference.item()

    # 3. LOGIC
    if myLossValue > myThreshold:
        myCurrentStatus = "MOVEMENT!"
        myColor = (0, 0, 255) 
    else:
        myCurrentStatus = "STILL"
        myColor = (0, 255, 0) 

    # 4. TERMINAL LOGGING (Only on change)
    if myCurrentStatus != myLastStatus:
        print(f"Log: {myCurrentStatus} (Loss: {myLossValue:.6f})")
        myLastStatus = myCurrentStatus

    # 5. Display on Screen
    cv2.putText(myFrame, f"Status: {myCurrentStatus}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, myColor, 2)
    cv2.imshow('Motion Sensor', myFrame)

    # 6. UPDATE MEMORY
    myOldTensor = myCurrentTensor

    # 7. THE FIX: Longer wait time and checking specifically for 'q'
    # 30ms is standard for 30fps video
    myKey = cv2.waitKey(30)
    
    # Check if 'q' was pressed OR if the window was closed with the [X] button
    if myKey & 0xFF == ord('q') or cv2.getWindowProperty('Motion Sensor', cv2.WND_PROP_VISIBLE) < 1:
        print("\n'q' detected or window closed. Exiting...")
        break

# Clean up
myCap.release()
cv2.destroyAllWindows()
# Final "nudge" to the OS to close windows
for i in range(1, 10):
    cv2.waitKey(1)