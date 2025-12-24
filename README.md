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





## 🚀 The Full Curriculum (Math to Microcontroller)

| File | Title | Concept Learned | tinyTorch Chapter |
| :--- | :--- | :--- | :--- |
| **torch01** | Scalar Tensors | Creating basic numbers and the "Tensor" data type. | **1.1** The Tensor Class |
| **torch02** | Autograd Math | How PyTorch tracks math history for derivatives. | **1.2** Automatic Differentiation |
| **torch03** | Linear Regression | Solving the $y = mx + b$ line equation. | **2.1** Linear Regression |
| **torch04** | The First Neuron | Building a single-layer neural network. | **2.2** Neurons & Perceptrons |
| **torch05** | Hidden Layers | Adding "inner thoughts" for non-linear problems. | **3.1** Multi-Layer Perceptrons |
| **torch06** | Optimizers | Using SGD and Adam to "walk" toward the answer. | **3.2** Optimization Algorithms |
| **torch07** | OpenCV Intro | Opening the webcam and handling image frames. | **4.1** Data Loading |
| **torch08** | Motion Detect | Using frame differencing to see movement. | **4.2** Basic Image Ops |
| **torch09** | Binary Classifier | Training the brain to see "Hand" vs "Background". | **5.1** Binary Classification |
| **torch10** | Persistent Memory | Saving and Loading weights using .pth files. | **6.1** Model Serialization |
| **torch11** | Multiclass Logic | Expanding to 3+ classes (Hand, Object, Background). | **5.2** Softmax & Entropy |
| **torch12** | Data Augment | Rotating and flipping images to make AI smarter. | **4.3** Augmentation |
| **torch13** | Validation Logic | Using 20% of data to "test" the brain's honesty. | **7.1** Evaluation |
| **torch14** | Feature Maps | Visualizing the Conv1 filters (The Brain's Eyes). | **8.1** Visualization |
| **torch15** | Export Master | Converting weights to C++ Header (.h) arrays. | **9.1** Deployment Bridge |
| **torch16** | MNIST Trainer | Training on 60,000 professional digit samples. | **10.1** Standard Datasets |
| **torch17** | Live Digit Reader | Live webcam OCR using Adaptive Thresholding. | **10.2** Real-world Inference |
