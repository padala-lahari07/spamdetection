import sys
import joblib
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTextEdit, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout)
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt

# Load trained spam classifier model & vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

class SpamClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Email Spam Classifier")
        self.setGeometry(100, 100, 600, 450)
        self.setStyleSheet("background-color: #2C2F33; color: white;")

        # Main Layout
        layout = QVBoxLayout()

        # Title Label
        self.title_label = QLabel("📧 Email Spam Detector", self)
        self.title_label.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("color: #00A8E8;")
        layout.addWidget(self.title_label)

        # Input Text Box
        self.text_input = QTextEdit(self)
        self.text_input.setPlaceholderText("Enter your email message here...")
        self.text_input.setFont(QFont("Arial", 12))
        self.text_input.setStyleSheet("background-color: #40444B; color: white; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.text_input)

        # Button Layout
        button_layout = QHBoxLayout()
        
        # Classify Button
        self.classify_button = QPushButton("🔍 Classify", self)
        self.classify_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.classify_button.setStyleSheet("background-color: #008CBA; color: white; padding: 10px; border-radius: 5px;")
        self.classify_button.clicked.connect(self.classify_message)
        button_layout.addWidget(self.classify_button)
        
        # Clear Button
        self.clear_button = QPushButton("🧹 Clear", self)
        self.clear_button.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.clear_button.setStyleSheet("background-color: #FF4500; color: white; padding: 10px; border-radius: 5px;")
        self.clear_button.clicked.connect(self.clear_text)
        button_layout.addWidget(self.clear_button)
        
        layout.addLayout(button_layout)
        
        # Result Label
        self.result_label = QLabel("Prediction: ", self)
        self.result_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("color: #00FF00;")
        layout.addWidget(self.result_label)

        # Set Layout to Central Widget
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def classify_message(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            self.result_label.setText("⚠️ Please enter a message!")
            self.result_label.setStyleSheet("color: #FF0000;")
            return

        # Preprocess and classify
        vectorized_text = vectorizer.transform([text]).toarray()
        prediction = model.predict(vectorized_text)
        result = "🚨 Spam" if prediction[0] == 1 else "✅ Ham"
        
        # Display result
        self.result_label.setText(f"Prediction: {result}")
        self.result_label.setStyleSheet("color: #FF0000;" if result == "🚨 Spam" else "color: #00FF00;")
    
    def clear_text(self):
        self.text_input.clear()
        self.result_label.setText("Prediction: ")
        self.result_label.setStyleSheet("color: white;")

# Run the App
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpamClassifierApp()
    window.show()
    sys.exit(app.exec())

