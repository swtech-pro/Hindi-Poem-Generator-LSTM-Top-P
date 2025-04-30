# 📜 Hindi Poem Generator (LSTM + Top-p Sampling)

This project is a deep learning-based Hindi poem generator built using TensorFlow and Keras. It uses LSTM layers and top-p (nucleus) sampling to generate fluent and diverse Hindi poetry based on a seed prompt.

## 🚀 Features
- Custom LSTM model built from scratch
- Top-p sampling for creative text generation
- Tokenization and sequence generation using Keras
- Trained on Hindi poem lines (Kaggle dataset or your own)
- Save and reuse model

## 🛠 Requirements
Install the dependencies using:
```bash
pip install -r requirements.txt
```

## 📁 File Structure
```
hindi_poem_lstm_top_p/
├── hindipoems.txt               # Hindi poem corpus
├── train_lstm.py                # Model training script
├── generate_poem.py             # Poem generation script
├── hindi_poem_lstm_top_p.ipynb  # Colab-compatible notebook
├── lstm_hindi_poem.h5           # Saved trained model
├── requirements.txt             # Dependencies
├── license.txt                  # MIT License
├── .gitignore
└── README.md
```

## 🧠 How to Train the Model
Run the following command:
```bash
python train_lstm.py
```

## 📝 How to Generate Poem
```bash
python generate_poem.py "प्रकृति की"
```

## ✅ Output Example
`प्रकृति की गोद में बिखरे हैं रंग और सुरों की मिठास...`

## 👨‍💻 Author
Developed as part of an AI Poetry Generation initiative.
