# my-examples-of-pytorch-for-learning-tinytorch
# ESP32-S3 Vision AI: From Math to Microcontroller

This project follows a step-by-step curriculum to understand Deep Learning using PyTorch, specifically designed for eventual deployment to an ESP32-S3 microcontroller using the **tinyTorch** framework.

---

## 🛠 Setup Instructions

### 1. Create the environment
Open your terminal inside your project folder and run:

**Windows:**
`python -m venv .myVenv`

**Mac/Linux:**
`python3 -m venv .myVenv`

### 2. Activate the environment
I have included an `activate.bat` file for Windows users. Simply run:
`activate` (or `.myVenv\Scripts\activate`)

**Mac/Linux:**
`source .myVenv/bin/activate`

*Note: You'll know it's working because `(.myVenv)` will appear next to your command prompt.*

### 3. Installation
Once activated, install the necessary AI and Vision libraries:
```bash
pip install torch torchvision torchaudio opencv-python
