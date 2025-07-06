import joblib

# Load the saved model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to predict spam/ham
def predict_message(text):
    from nltk.corpus import stopwords
    import string
    
    # Define text cleaning function
    def clean_text(text):
        text = text.lower()  # Convert to lowercase
        text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
        words = text.split()
        words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
        return " ".join(words)

    # Preprocess text
    text = clean_text(text)
    
    # Convert to numerical format
    vectorized_text = vectorizer.transform([text]).toarray()
    
    # Predict
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Ham"

# Test messages
print(predict_message("You have won $1000! Click the link to claim."))
print(predict_message("Hey, let's meet for lunch tomorrow."))
