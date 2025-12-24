import torch
import cv2
import numpy as np

# torch07.py (Revised): WebCam to Tensor with Proof of Life
print("--- Step 1: Initialize Camera ---")
myCap = cv2.VideoCapture(0)

if not myCap.isOpened():
    print("Error: Could not open webcam. Is another app using it?")
    exit()

print("Camera is LIVE. Press 'q' to quit.")

while True:
    myRet, myFrame = myCap.read()
    if not myRet:
        break

    # 1. Convert to Tensor (The "Brain" format)
    mySmallFrame = cv2.resize(myFrame, (100, 100)) # Smaller for faster processing
    myTensor = torch.from_numpy(mySmallFrame).permute(2, 0, 1).float() / 255.0

    # 2. PROOF OF GRABBING PIXELS:
    # Calculate the average brightness of the whole image
    myBrightness = torch.mean(myTensor).item()
    
    # Use '\r' to overwrite the same line in the terminal (cleaner output)
    print(f"Current Avg Brightness: {myBrightness:.4f} ", end='\r')

    # 3. Show the image window
    cv2.imshow('My WebCam Feed', myFrame)

    # 4. HANDLE KEYBOARD (The logic fix)
    # We store the keypress in 'myKey'
    myKey = cv2.waitKey(1) & 0xFF
    
    if myKey == ord('q'):
        print("\n'q' pressed. Closing...")
        break
    elif myKey == ord('s'):
        print(f"\nManual Snapshot! Tensor Shape: {myTensor.shape}")

# Clean up
myCap.release()
cv2.destroyAllWindows()
print("Camera closed successfully.")