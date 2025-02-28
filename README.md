# Hate Speech Detection using MuRIL and BERT Transformers

## 📌 Project Overview
This project implements **Hate Speech Detection** using **MuRIL (Multilingual Representations for Indian Languages) and BERT transformers**. The models classify social media posts into three categories: **Profane, Hate, and Neutral**.

## 📂 Dataset
- **Labeled Data:** Contains tweets labeled as `Profane`, `Hate`, or `Neutral`.
- **Profanity List:** A CSV file containing a list of Hinglish (Roman Hindi) profane words.
- **Hinglish to English Dictionary:** JSON file mapping Roman Hindi words to their English equivalents.

## 🚀 Model Architecture
- **Pretrained Models:** 
  - `google/muril-base-cased`
  - `bert-base-uncased`
- **Custom Classifier:** 
  - A fully connected layer with ReLU activation.
  - Dropout layer for regularization.
  - Final classification layer.

## 🛠 Installation & Setup
### **1. Install Dependencies**
```bash
pip install numpy pandas torch transformers scikit-learn tqdm matplotlib
```

### **2. Download Pretrained Models**
The models are loaded using `transformers` library:
```python
from transformers import AutoTokenizer, AutoModel
MODEL_NAME = 'google/muril-base-cased'
BERT_MODEL_NAME = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

model = AutoModel.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(BERT_MODEL_NAME)
```

## 📊 Data Preprocessing
- **Remove Missing/Duplicate Values** from labeled dataset.
- **Convert Roman Hindi Words to English** using the Hinglish dictionary.
- **Tokenization** using MuRIL and BERT tokenizers.
- **Handle Out-of-Vocabulary (OOV) words**.

## 🎯 Training Process
- **Dataset Splitting:**
  - **Train (70%)**
  - **Validation (15%)**
  - **Test (15%)**
- **Batch Size:** 16
- **Learning Rate:** 1e-5
- **Epochs:** 5
- **Optimizer:** Adam
- **Loss Function:** CrossEntropyLoss with class weighting

### **Train the Models**
```python
for epoch in range(EPOCHS):
    train_model(epoch)  # Train MuRIL model
    train_bert_model(epoch)  # Train BERT model
```

## 🧪 Evaluation
- **Metrics:** Accuracy, Loss, Precision, Recall, F1-score
- **Model Testing:**
```python
evaluate_muril_model()
evaluate_bert_model()
```

## 🎯 Classification of Media Posts
The models can classify a social media post as **Profane, Hate, or Neutral**.
```python
text = "kharab dost harami idiot"
classify_media_post(text)
```

## 📈 Model Performance
- **Final MuRIL Training Accuracy:** ~88.64%
- **Final MuRIL Test Accuracy:** ~83.10%
- **Final BERT Training Accuracy:** ~91.52%
- **Final BERT Test Accuracy:** TBD

## 📊 Visualization of Training Progress
The script generates **loss and accuracy plots** after training:
```python
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()
```

## 💾 Model Saving & Loading
To save the trained models:
```python
torch.save(muril_model.state_dict(), 'muril_model_final.pt')
torch.save(bert_model.state_dict(), 'bert_model_final.pt')
```
To load the models later:
```python
muril_model.load_state_dict(torch.load('muril_model_final.pt'))
bert_model.load_state_dict(torch.load('bert_model_final.pt'))
```

## 📌 Future Improvements
- Use **BERT-based models** like `XLM-R` for better multilingual representation.
- Implement **attention mechanisms** for interpretability.
- Fine-tune on a **larger labeled dataset** for better generalization.

## 🤝 Contributing
Feel free to submit issues or pull requests to improve the model! 🚀

