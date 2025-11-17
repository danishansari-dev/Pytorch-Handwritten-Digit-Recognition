# PyTorch Handwritten Digit Recognition

A compact, easy-to-follow example that trains a Convolutional Neural Network (CNN) on the MNIST dataset using PyTorch. It demonstrates dataset loading, model definition, training/testing loops, and how to run inference on single images.

🚀 Project owner / maintainer: **Mohammad Danish Ansari**

This project is inspired by the open-source machine learning community.

---

## ⭐ Features

- Clean, minimal CNN for digit classification
- Training and evaluation scripts with GPU support (if available)
- Data loaders using the official MNIST dataset
- Example code for running predictions on single images
- Easy-to-follow README with reproduction steps

---

## 🧭 Technologies

- Python 3.8+ (recommended)
- PyTorch
- torchvision
- NumPy
- Matplotlib (optional, for visualizations)

---

## 📚 Dataset — MNIST

The project uses the classic MNIST dataset of handwritten digits:

- 60,000 training images (28x28, grayscale)
- 10,000 test images
- Dataset provided by `torchvision.datasets.MNIST` (downloaded automatically to `data/MNIST/raw`)

---

## 🏗️ Model Architecture (Overview)

The example model is a simple CNN suitable for MNIST-level tasks:

- Input: 1 x 28 x 28 grayscale image
- Conv2d(1 -> 10, kernel=5) -> ReLU -> MaxPool(2)
- Conv2d(10 -> 20, kernel=5) -> Dropout2d -> ReLU -> MaxPool(2)
- Flatten
- Linear(320 -> 50) -> ReLU -> Dropout
- Linear(50 -> 10) -> Softmax logits

This architecture balances clarity and performance for educational purposes. You can replace it with deeper networks (ResNet, LeNet variants, etc.) for higher accuracy.

---

## ⚙️ Installation

1. Clone the repository (or download the files):

```bash
git clone https://github.com/danishansari-dev/Pytorch-Handwritten-Digit-Recognition.git
cd Pytorch-Handwritten-Digit-Recognition
```

2. (Optional) Create a virtual environment and activate it:

Windows (cmd.exe):

```cmd
python -m venv venv
venv\\Scripts\\activate
```

3. Install required packages:

```cmd
pip install -r requirements.txt
```

If a `requirements.txt` is not present, install the minimum packages:

```cmd
pip install torch torchvision numpy matplotlib
```

---

## ▶️ How to Run (Quick Start)

1. Ensure you have the MNIST data (the code will download it automatically):

```cmd
python train.py --epochs 5 --batch-size 64
```

2. Evaluate the model (after training or using a saved checkpoint):

```cmd
python evaluate.py --checkpoint checkpoints/model.pt
```

3. Run a single-image prediction (example):

```cmd
python predict.py --image examples/one_digit.png --checkpoint checkpoints/model.pt
```

Notes:

- Training script arguments (`train.py`) typically include `--epochs`, `--batch-size`, `--lr`, and `--device`.
- The project contains small example scripts; adapt paths and arguments as needed.

---

## 🛠️ How to Train (Conceptual)

Typical training loop used in this project:

1. Create `DataLoader` for train and test splits (shuffling train set).
2. Instantiate the model and move it to device (`cpu` or `cuda`).
3. Use `CrossEntropyLoss()` and an optimizer (e.g., `Adam`).
4. For each epoch:
   - Set model to `train()` and iterate batches
   - Compute logits, loss, `loss.backward()` and `optimizer.step()`
   - Periodically log training loss and (optionally) save checkpoints
5. After each epoch, evaluate using `model.eval()` and `torch.no_grad()`.

Example (pseudo):

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(1, epochs+1):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    evaluate(model, val_loader)
```

---

## ✅ How to Test Predictions

To run single-image inference:

```python
from PIL import Image
import torchvision.transforms as T

img = Image.open('examples/one_digit.png').convert('L')
transform = T.Compose([T.Resize((28,28)), T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
tensor = transform(img).unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    logits = model(tensor)
    pred = logits.argmax(dim=1).item()
    print('Predicted digit:', pred)
```

Placeholders for `predict.py` and `evaluate.py` are available in the repository root (or create them following these examples).

---

## 🖼️ Screenshots

Add screenshots or GIFs inside the `assets/` or `examples/` folder and reference them here. Example placeholder:

![Training curve placeholder](assets/training_curve.png)

If you don't have images yet, create `assets/placeholder.png` and add it to the repo.

---

## 🔮 Future Improvements

- Replace the small CNN with a stronger baseline (e.g., ResNet18) for higher accuracy
- Add training scripts that log to TensorBoard / Weights & Biases
- Add unit tests for data loading and small integration tests
- Provide a simple Flask/FastAPI demo to serve the model for inference
- Hyperparameter search (Optuna, Ray Tune)

---

## 📄 License & Credits

This project is released under the MIT License. See `LICENSE` for details.

**Credits:** This project is inspired by the open-source ML community.

---

## 👤 Author / Maintainer / Contact

- **Author:** Mohammad Danish Ansari
- **Maintainer:** Mohammad Danish Ansari
- **Contact:** danishansari-dev (GitHub) — open issues or reach me at `danishansari-dev` on GitHub

If you'd like to contribute or report issues, please open a GitHub issue or submit a PR.

---
