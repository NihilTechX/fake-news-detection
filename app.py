import os
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from flask import Flask, render_template, request, jsonify

# Download NLTK resources (if not already downloaded)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

app = Flask(__name__)

# Load the model and vectorizer
model_dir = os.path.dirname(os.path.abspath(__file__))
model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
vectorizer_files = [f for f in os.listdir(model_dir) if f.endswith('_vectorizer.pkl')]

if not model_files or not vectorizer_files:
    print("Model files not found. Please run fake_news_detector.py first to train the models.")
    model = None
    vectorizer = None
else:
    # Use the first model found (should be the best one)
    model_path = os.path.join(model_dir, model_files[0])
    vectorizer_path = os.path.join(model_dir, vectorizer_files[0])
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print(f"Loaded model from {model_path}")
    print(f"Loaded vectorizer from {vectorizer_path}")

def preprocess_text(text):
    """Preprocess text before prediction (same as in training)"""
    if isinstance(text, str):
        try:
            # Basic text cleaning
            text = text.lower()
            text = re.sub(r'\W', ' ', text)  # Remove all non-word characters
            text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
            text = re.sub(r'\d', '', text)    # Remove digits
            
            # Get stopwords and initialize stemmer
            stop_words = set(stopwords.words('english'))
            stemmer = PorterStemmer()
            
            # Simple tokenization as fallback if NLTK tokenization fails
            try:
                tokens = nltk.word_tokenize(text)
            except LookupError:
                # Fallback to simple splitting if tokenization fails
                tokens = text.split()
                
            # Apply stemming and remove stopwords
            tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Warning: Error in text preprocessing: {e}")
            # Return a cleaned version without advanced NLP if there's an error
            return re.sub(r'\W+', ' ', text.lower()).strip()
    else:
        return ''

def predict_news(text):
    """Predict if news is real or fake"""
    if model is None or vectorizer is None:
        return {"error": "Model not loaded. Please train the model first."}, 0
    
    try:
        # Preprocess the text
        clean_text = preprocess_text(text)
        
        # Vectorize the text
        vectorized_text = vectorizer.transform([clean_text])
        
        # Make prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Get confidence score
        # The way to get confidence depends on the model type
        try:
            # For LinearSVC or SVC
            confidence = abs(model.decision_function(vectorized_text)[0])
        except:
            # For other models that don't have decision_function
            try:
                # For models with predict_proba
                probs = model.predict_proba(vectorized_text)[0]
                confidence = max(probs)
            except:
                # If all else fails
                confidence = 1.0
        
        # Return result
        if prediction == 1:
            result = {"prediction": "REAL", "probability": float(confidence)}
        else:
            result = {"prediction": "FAKE", "probability": float(confidence)}
            
        return result, confidence
    except Exception as e:
        return {"error": str(e)}, 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    news_text = data.get('text', '')
    
    if not news_text:
        return jsonify({"error": "No text provided"}), 400
    
    # Make prediction
    result, confidence = predict_news(news_text)
    
    # Return result
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
