import joblib
import sys
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d', '', text)
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def predict_news(text, model_name='PAC'):
    try:
        # Get the directory where the script is located
        model_dir = os.path.dirname(os.path.abspath(__file__))
        # Look for models in the same directory as the script
        model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
        vectorizer_path = os.path.join(model_dir, f'{model_name}_vectorizer.pkl')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"Error: Model files not found at:")
            print(f"  - {model_path}")
            print(f"  - {vectorizer_path}")
            print("Please run fake_news_detector.py first to train and save the models.")
            return None, None
        
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        preprocessed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([preprocessed_text])
        
        prediction = model.predict(vectorized_text)[0]
        confidence = abs(model.decision_function(vectorized_text)[0])
        
        if prediction == 1:
            return "REAL", confidence
        else:
            return "FAKE", confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_news.py \"<news article text>\" [model_name]")
        return
    
    news_text = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'PAC'
    
    result, confidence = predict_news(news_text, model_name)
    
    if result:
        print(f"Prediction: The news is {result}")
        print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()
