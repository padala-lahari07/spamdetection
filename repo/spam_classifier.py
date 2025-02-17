import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset correctly
df = pd.read_csv("spam.csv", encoding="latin-1")

# Rename columns correctly
df = df[['v1', 'v2']]  # Keep only relevant columns
df.columns = ['label', 'message']  # Rename columns

# Convert labels to numeric format (0 = ham, 1 = spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop any empty rows
df.dropna(inplace=True)

# Download stopwords if not already present
nltk.download('stopwords')

# Define text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    cleaned_text = " ".join(words)

    # If the message is empty after cleaning, return a placeholder
    return cleaned_text if cleaned_text.strip() else "emptytext"

# Apply text cleaning
df['clean_message'] = df['message'].apply(clean_text)

# Convert text into numerical features
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 words
X = vectorizer.fit_transform(df['clean_message']).toarray()  # Convert to array
y = df['label'].values  # Labels (0 = ham, 1 = spam)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print classification report
print(classification_report(y_test, y_pred))

# Function to predict spam/ham for a given message
def predict_message(text):
    text = clean_text(text)  # Preprocess the text
    vectorized_text = vectorizer.transform([text]).toarray()  # Convert to numerical form
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Ham"

# Example Test Messages
print(predict_message("Congratulations! You've won a free iPhone. Click here to claim!"))
print(predict_message(
    "Hey, are you coming to the meeting tomorrow?"))

# Save model
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Load model later (example)
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")
