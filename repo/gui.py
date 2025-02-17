import tkinter as tk
from tkinter import messagebox, scrolledtext
import joblib
import string
from nltk.corpus import stopwords
import nltk

# Load the trained model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Download stopwords if not already present
nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words) if words else "emptytext"

def predict_message():
    user_input = entry.get("1.0", "end-1c")  # Get text from the input field
    if not user_input.strip():
        messagebox.showwarning("Warning", "Please enter a message to classify.")
        return
    
    cleaned_input = clean_text(user_input)
    vectorized_text = vectorizer.transform([cleaned_input]).toarray()
    prediction = model.predict(vectorized_text)
    result = "Spam" if prediction[0] == 1 else "Ham"
    
    label_result.config(text=f"Result: {result}", fg="red" if result == "Spam" else "green")

# GUI Setup
root = tk.Tk()
root.title("Spam Classifier")
root.geometry("500x400")
root.configure(bg="#f0f0f0")

# Heading Label
label_title = tk.Label(root, text="Spam Message Classifier", font=("Arial", 16, "bold"), bg="#f0f0f0", fg="#333")
label_title.pack(pady=10)

# Text Entry Field with Scrollbar
entry = scrolledtext.ScrolledText(root, height=5, width=50, font=("Arial", 12))
entry.pack(pady=10)

# Predict Button with Styling
button_predict = tk.Button(root, text="Check Message", command=predict_message, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5)
button_predict.pack(pady=10)

# Result Label
label_result = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f0f0f0")
label_result.pack(pady=10)

# Run GUI
root.mainloop()
