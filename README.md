# 🐶 Dog Breed Identification with CNN (TensorFlow + Keras)

This project uses Convolutional Neural Networks (CNNs) to identify dog breeds from images. Built using **TensorFlow** and **Keras**, the model is trained on the [Kaggle Dog Breed Identification dataset](https://www.kaggle.com/competitions/dog-breed-identification).

---

## 📁 Project Structure

```
Dog-Breed-Identification/
├── data_loader.py         # Data loading and preprocessing
├── src/
│   ├── model.py           # CNN model architecture
│   ├── train.py           # Training script
│   ├── evaluate.py        # Single image prediction
│   └── utils.py           # (Optional) Helper functions
├── sample_data/           # (Optional) Small subset for testing
├── notebooks/             # Jupyter notebooks for EDA/training
├── dog_breed_model.h5     # Saved Keras model (post-training)
├── requirements.txt       # Required Python packages
├── .gitignore
└── README.md
```

---

## 📦 Dataset

- 🐾 **Source**: [Kaggle - Dog Breed Identification](https://www.kaggle.com/competitions/dog-breed-identification/data)
- 🔄 Contains 120 dog breeds and 10,000+ labeled images

### 📥 Download Instructions

To use the dataset:

1. Install [Kaggle CLI](https://github.com/Kaggle/kaggle-api):
   ```bash
   pip install kaggle
   ```
2. Authenticate by placing your `kaggle.json` key file in the correct location.
3. Download the data:
   ```bash
   python data_loader.py
   ```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

### 3. Predict a breed from an image

```bash
python src/evaluate.py --image_path sample_data/golden_retriever.jpg
```

(Ensure the model is saved as `dog_breed_model.h5` after training.)

---

## 🧠 Model Architecture

- Input size: `224x224x3` images
- Layers:
  - 2x Conv2D + MaxPooling
  - Flatten + Dense + Dropout
  - Output Dense layer with `softmax` activation
- Loss: `categorical_crossentropy`
- Optimizer: `Adam`

---

## 📈 Evaluation

- Uses 80/20 train-validation split via `ImageDataGenerator`.
- Accuracy and loss tracked during training.
- You can add confusion matrix, classification report, and Top-5 accuracy metrics.

---

## 📊 Sample Output

```
Prediction: golden_retriever
Confidence: 97.5%
```

---

## ✨ Future Improvements

- Use a pretrained model (MobileNetV2, ResNet50)
- Add data augmentation
- Implement early stopping and model checkpoints
- Deploy with Streamlit or Flask

---

## 📝 License

MIT License. See `LICENSE` file for details.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📬 Contact

Created by [Aditya Gautam](https://github.com/Aditya-gautam21)  
Feel free to reach out for collaboration or questions!