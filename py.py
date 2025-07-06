import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset correctly with tab separation
df = pd.read_csv("spam.csv", sep="\t", encoding="latin-1", names=["label", "message"])

# Drop any empty rows
df.dropna(inplace=True)

# Convert labels to numeric format (0 = ham, 1 = spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Download stopwords if not already present
nltk.download('stopwords')

# Define text cleaning function
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = "".join([char for char in text if char not in string.punctuation])  # Remove punctuation
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]  # Remove stopwords
    return " ".join(words)

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



# Save model
joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# Load model later (example)
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

#path:cd E:\Projects\email_spam_detection 
#venv\Scripts\activate
#pip install numpy pandas scikit-learn nltk matplotlib seaborn pyqt6-tools
#to run use:python spam_gui.py