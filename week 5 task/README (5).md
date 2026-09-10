# Week 5 – Deep Learning Application in Data Science

## Project
**Handwritten Digit Classification using a Convolutional Neural Network (CNN)**

This project fulfills the Week 5 requirement to design, train, and evaluate a neural network using a publicly available dataset. The MNIST handwritten-digit dataset is used with TensorFlow/Keras.

## Objectives
- Select a suitable public dataset.
- Define a classification problem.
- Preprocess image data.
- Design and justify a CNN architecture.
- Train and validate the network.
- Evaluate the model using accuracy, loss, a classification report, and a confusion matrix.
- Address overfitting using dropout and early stopping.
- Save the trained model and generated evaluation figures.

## Dataset
MNIST contains grayscale images of handwritten digits from 0 to 9. Each image is 28×28 pixels. TensorFlow/Keras provides a direct public loader, so a separate manual download is not required.

## Architecture
1. Input: 28×28×1
2. Conv2D: 32 filters, 3×3, ReLU
3. MaxPooling2D: 2×2
4. Conv2D: 64 filters, 3×3, ReLU
5. MaxPooling2D: 2×2
6. Flatten
7. Dense: 128 neurons, ReLU
8. Dropout: 30%
9. Output: 10 neurons, Softmax

Adam is used as the optimizer and sparse categorical cross-entropy as the loss function.

## How to run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python week5_deep_learning.py
```

The script creates an `outputs/` folder containing:
- `confusion_matrix.png`
- `accuracy_curve.png`
- `loss_curve.png`
- `sample_predictions.png`
- `mnist_cnn.keras`

## Reproducibility
Random seeds are fixed at 42 for NumPy and TensorFlow. Training uses a held-out validation set and early stopping.

## Limitations and improvements
The model is intentionally compact and suitable for a student/internship project. Performance can be improved through data augmentation, batch normalization, learning-rate scheduling, hyperparameter tuning, or a deeper CNN. CPU-only environments may require additional training time.

## Files
- `week5_deep_learning.py` – complete implementation
- `requirements.txt` – Python dependencies
- `README.md` – project instructions
- `Week_5_Deep_Learning_Report.docx` – detailed submission report
