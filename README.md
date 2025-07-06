# 📧 Email Spam Detection Model

A machine learning project to classify emails as **Spam** or **Not Spam** using natural language processing (NLP) techniques and supervised learning algorithms.


## 🚀 Project Overview

This project builds a spam detection system using Python, scikit-learn, and NLP techniques. It uses a dataset of labeled emails and processes them to train a machine learning model that can accurately classify unseen emails.


## 🧠 Features

- Preprocessing of raw email text (removal of stopwords, stemming, etc.)
- Feature extraction using TF-IDF vectorization
- Model training using classifiers like:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machines (SVM)
- Evaluation with accuracy, precision, recall, F1-score
- GUI support (optional, e.g., using Tkinter or PyQt)
- Save/load trained model using `joblib` or `pickle`

---

## 📁 Project Structure

```
📦email-spam-detector/
 ┣ 📂data/
 ┃ ┗ spam.csv
 ┣ 📂models/
 ┃ ┗ spam_classifier.pkl
 ┣ 📂notebooks/
 ┃ ┗ SpamDetection.ipynb
 ┣ 📂gui/ (optional)
 ┃ ┗ spam_gui.py
 ┣ train_model.py
 ┣ predict.py
 ┗ README.md
```

---

## 🔧 Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

**Main Libraries:**
- numpy
- pandas
- scikit-learn
- nltk
- matplotlib
- seaborn
- joblib

---

## 🧪 Model Training

To train the model:

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Train a classifier
- Save the trained model to `models/`

---

## 🔍 Model Evaluation

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

## 📈 Sample Results

| Metric     | Score     |
|------------|-----------|
| Accuracy   | 98.5%     |
| Precision  | 97.3%     |
| Recall     | 96.8%     |
| F1-Score   | 97.0%     |

---

## 🖼 GUI (Optional)

To run the GUI app (if implemented):

```bash
python gui/spam_gui.py
```

Enter the email content and get real-time classification!

---

## 🗂 Dataset

Used the popular **Spam Collection Dataset (SMS Spam Collection)** or any custom dataset in `.csv` format with two columns:
- `label` (spam/ham)
- `message` (email text)

Example:

```csv
label,message
ham,"Hey there, how are you?"
spam,"Congratulations! You won a free ticket..."
```

---

## 💡 Future Improvements

- Use deep learning models (e.g., LSTM, BERT)
- Multi-language spam detection
- Web deployment with Flask or FastAPI
- Email scraping and real-time inbox classification

---

## 🤝 Contribution

Feel free to fork this repo, raise issues or submit PRs to improve the system!

---

## 🙋‍♂️ Author

**Padala Lakshmi Sai Lahari**  
[GitHub](https://github.com/padala-lahari07) | [LinkedIn](https://www.linkedin.com/in/padala-lakshmi-sai-lahari-b08b59259/)
