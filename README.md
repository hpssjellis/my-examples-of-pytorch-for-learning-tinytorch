# my-examples-of-pytorch-for-learning-tinytorch
# ESP32-S3 Vision AI: From Math to Microcontroller

This project follows a step-by-step curriculum to understand Deep Learning using PyTorch, specifically designed for eventual deployment to an ESP32-S3 microcontroller using the **tinyTorch** framework.

---

## 🛠 Setup Instructions

### 1. Create the environment
Open your terminal inside your project folder and run:

**Windows:**
```python -m venv .myVenv```

**Mac/Linux:**
```python3 -m venv .myVenv```

### 2. Activate the environment
I have included an `activate.bat` file for Windows users. Simply run:
```activate``` (or ```.myVenv\Scripts\activate```)

**Mac/Linux:**
```source .myVenv/bin/activate```

*Note: You'll know it's working because `(.myVenv)` will appear next to your command prompt.*

### 3. Installation
Once activated, install the necessary AI and Vision libraries:
```bash
pip install torch torchvision torchaudio opencv-python

```




## 🚀 The Curriculum (Script Progress)

| File | Title | Concept Learned |
| :--- | :--- | :--- |
| **torch01** | Scalar Tensors | Creating basic numbers in PyTorch and understanding the "Tensor" data type. |
| **torch02** | Autograd Math | Learning how PyTorch tracks math history to calculate derivatives automatically. |
| **torch03** | Linear Regression | Teaching the computer to find the slope of a line ($y = mx + b$). |
| **torch04** | The First Neuron | Building a single-layer neural network to solve numeric patterns. |
| **torch05** | Hidden Layers | Adding "inner thoughts" to the network to solve non-linear problems. |
| **torch06** | Optimizers | Comparing SGD and Adam to see how the computer "walks" toward the answer. |
| **torch07** | OpenCV Integration | Opening the webcam and displaying a live feed at **320x320** resolution. 'q' to quit |
| **torch08** | Motion Detection | Using "Frame Differencing" to detect change between the current and last frame. |
| **torch09** | Binary Classifier | Using a CNN to distinguish between a 'h' **Hand** and the 'b' **Background**. |
| **torch10** | Persistent Memory | Saving weights to a `.pth` file so the brain "remembers" after 'q' quitting. |
| **torch11** | Multiclass Logic | Teaching the brain to recognize 3 distinct states: 'b' **Background**,'h' **Hand**, and 'p' **Object**. |
| **torch12** | Data Augmentation | Making the AI robust by randomly rotating and flipping images during training. |




## 🚀 The Curriculum (Script Progress)

| File | Title | Concept Learned | tinyTorch Chapter |
| :--- | :--- | :--- | :--- |
| **torch01** | Scalar Tensors | Creating basic numbers and the "Tensor" data type. | **1.1** The Tensor Class |
| **torch02** | Autograd Math | How PyTorch tracks math history for derivatives. | **1.2** Automatic Differentiation |
| **torch03** | Linear Regression | Solving the $y = mx + b$ line equation. | **2.1** Linear Regression |
| **torch04** | The First Neuron | Building a single-layer neural network. | **2.2** Neurons & Perceptrons |
| **torch05** | Hidden Layers | Adding "inner thoughts" for non-linear problems. | **3.1** Multi-Layer Perceptrons |
| **torch06** | Optimizers | Using SGD and Adam to "walk" toward the answer. | **3.2** Optimization Algorithms |
| **torch07** | OpenCV Integration | Opening the webcam at **320x320** resolution. | **4.1** Data Loading & Input |
| **torch08** | Motion Detection | Using "Frame Differencing" for change detection. | **4.2** Basic Image Operations |
| **torch09** | Binary Classifier | Distinguishing between **Hand** and **Background**. | **5.1** Binary Classification |
| **torch10** | Persistent Memory | Saving weights to a `.pth` file. | **6.1** Model Serialization |
| **torch11** | Multiclass Logic | Recognizing **Background**, **Hand**, and **Object**. | **5.2** Softmax & Cross Entropy |
| **torch12** | Data Augmentation | Making AI robust with rotations and flips. | **4.3** Data Augmentation |
