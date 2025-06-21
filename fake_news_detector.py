import pandas as pd
import numpy as np
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Download all needed NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

def load_datasets(true_path, fake_path):
    # Load the datasets with better error handling
    try:
        true_news = pd.read_csv(true_path)
        fake_news = pd.read_csv(fake_path)
        
        # Print data info to help with debugging
        print(f"\nTrue news dataset shape: {true_news.shape}")
        print(f"Fake news dataset shape: {fake_news.shape}")
        print(f"True news columns: {true_news.columns.tolist()}")
        print(f"Fake news columns: {fake_news.columns.tolist()}")
        
        # Check if 'text' column exists, if not, look for similar columns
        text_column_candidates = ['text', 'content', 'article', 'news', 'body']
        
        # Find text column in true_news
        true_text_col = None
        for col in text_column_candidates:
            if col in true_news.columns:
                true_text_col = col
                break
        
        if true_text_col is None:
            print("Warning: Could not find text column in true_news. Using the first column.")
            true_text_col = true_news.columns[0]
        
        # Find text column in fake_news
        fake_text_col = None
        for col in text_column_candidates:
            if col in fake_news.columns:
                fake_text_col = col
                break
                
        if fake_text_col is None:
            print("Warning: Could not find text column in fake_news. Using the first column.")
            fake_text_col = fake_news.columns[0]
            
        print(f"Using '{true_text_col}' as text column for true news")
        print(f"Using '{fake_text_col}' as text column for fake news")
        
        # Create standard column names
        true_news = true_news.rename(columns={true_text_col: 'text'})
        fake_news = fake_news.rename(columns={fake_text_col: 'text'})
        
        # Add labels
        true_news['label'] = 1
        fake_news['label'] = 0
        
        # Combine datasets
        all_news = pd.concat([true_news, fake_news])
        
        # Quick check for meaningful text
        print(f"\nSample text from true news: {all_news[all_news['label']==1]['text'].iloc[0][:100]}...")
        print(f"Sample text from fake news: {all_news[all_news['label']==0]['text'].iloc[0][:100]}...")
        
        return all_news
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        sys.exit(1)

def preprocess_text(text):
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

def vectorize_text(text_data):
    # Using fewer features for faster processing
    vectorizer = TfidfVectorizer(max_features=1000)  # Reduced from 5000 to 1000
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:")
    print(cm)
    
    return accuracy, f1, cm

def train_pac(X_train, y_train):
    pac = PassiveAggressiveClassifier(max_iter=1000, random_state=42)
    pac.fit(X_train, y_train)
    return pac

def train_svm(X_train, y_train):
    # Using faster configurations for SVM
    # LinearSVC is much faster than SVC with linear kernel
    from sklearn.svm import LinearSVC
    svm = LinearSVC(random_state=42, max_iter=1000, dual=False)
    svm.fit(X_train, y_train)
    return svm

def plot_comparison(models_results):
    models = list(models_results.keys())
    accuracies = [models_results[model]['accuracy'] for model in models]
    f1_scores = [models_results[model]['f1'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, accuracies, width, label='Accuracy')
    rects2 = ax.bar(x + width/2, f1_scores, width, label='F1 Score')
    
    ax.set_ylabel('Scores')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')

def plot_confusion_matrices(models_results):
    fig, axes = plt.subplots(1, len(models_results), figsize=(15, 5))
    
    for i, (model_name, results) in enumerate(models_results.items()):
        sns.heatmap(results['cm'], annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'{model_name} Confusion Matrix')
        axes[i].set_xlabel('Predicted labels')
        axes[i].set_ylabel('True labels')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')

def save_model(model, vectorizer, model_name):
    import joblib
    # Get the directory where the script is located
    model_dir = os.path.dirname(os.path.abspath(__file__))
    # Save models in the same directory as the script
    model_path = os.path.join(model_dir, f'{model_name}_model.pkl')
    vectorizer_path = os.path.join(model_dir, f'{model_name}_vectorizer.pkl')
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"{model_name} model saved to: {model_path}")
    print(f"{model_name} vectorizer saved to: {vectorizer_path}")

def main():
    try:
        print("\n=== Fake News Detection System ===\n")
        
        # Default path (you can modify this to match your local structure)
        default_dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'archive (2)', 'News _dataset'))
        
        # Alternative paths to try if default doesn't work
        alternative_paths = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'News _dataset')),  # Without archive (2)
            os.path.abspath('News _dataset'),  # Direct path
        ]
        
        # Check if custom paths are provided via command line arguments
        if len(sys.argv) > 2:
            true_path = sys.argv[1]
            fake_path = sys.argv[2]
        else:
            # Try default path
            true_path = os.path.join(default_dataset_dir, 'True.csv')
            fake_path = os.path.join(default_dataset_dir, 'Fake.csv')
            
            # If default path doesn't exist, try alternatives
            if not os.path.exists(true_path) or not os.path.exists(fake_path):
                found = False
                for alt_path in alternative_paths:
                    alt_true = os.path.join(alt_path, 'True.csv')
                    alt_fake = os.path.join(alt_path, 'Fake.csv')
                    if os.path.exists(alt_true) and os.path.exists(alt_fake):
                        true_path = alt_true
                        fake_path = alt_fake
                        found = True
                        print(f"Found dataset at alternative location: {alt_path}")
                        break
                
                if not found:
                    print(f"Using default dataset paths. To specify custom paths, run:")
                    print(f"python {sys.argv[0]} <path_to_true_news.csv> <path_to_fake_news.csv>")
        
        # Validate that files exist
        if not os.path.exists(true_path) or not os.path.exists(fake_path):
            print(f"Error: Dataset files not found at the specified paths.")
            print(f"True news path: {true_path}")
            print(f"Fake news path: {fake_path}")
            print(f"Please check the file paths and try again.")
            sys.exit(1)
        
        print(f"Using dataset files:")
        print(f"- True news: {true_path}")
        print(f"- Fake news: {fake_path}")
        
        print("\nLoading datasets...")
        df = load_datasets(true_path, fake_path)
        
        # Reduce dataset size for much faster processing
        # Using only 5% of the data for quick testing
        sample_fraction = 0.05  # Use only 5% of the data
        print(f"\nReducing dataset size to {sample_fraction*100}% for faster processing...")
        print(f"Original dataset size: {len(df)} articles")
        df = df.sample(frac=sample_fraction, random_state=42)
        print(f"Reduced dataset size: {len(df)} articles")
        
        print("\nPreprocessing text data...")
        df['clean_text'] = df['text'].apply(preprocess_text)
        
        print("Vectorizing text data...")
        X, vectorizer = vectorize_text(df['clean_text'])
        y = df['label']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("Training Passive Aggressive Classifier...")
        pac_model = train_pac(X_train, y_train)
        
        print("Training SVM...")
        svm_model = train_svm(X_train, y_train)
        
        print("\nEvaluating models:")
        pac_accuracy, pac_f1, pac_cm = evaluate_model(pac_model, X_test, y_test, "Passive Aggressive Classifier")
        svm_accuracy, svm_f1, svm_cm = evaluate_model(svm_model, X_test, y_test, "Support Vector Machine")
        
        models_results = {
            'PAC': {'accuracy': pac_accuracy, 'f1': pac_f1, 'cm': pac_cm},
            'SVM': {'accuracy': svm_accuracy, 'f1': svm_f1, 'cm': svm_cm}
        }
        
        print("\nPlotting comparison charts...")
        plot_comparison(models_results)
        plot_confusion_matrices(models_results)
        
        best_model = pac_model if pac_accuracy > svm_accuracy else svm_model
        best_model_name = "PAC" if pac_accuracy > svm_accuracy else "SVM"
        
        print(f"\nSaving the best model ({best_model_name})...")
        save_model(best_model, vectorizer, best_model_name)
        
        print("\nFake news detection model training complete!")
        
    except LookupError as e:
        print(f"\nError: NLTK resource not found. {str(e)}")
        print("Try running the following commands to download required resources:")
        print("import nltk")
        print("nltk.download('punkt')")
        print("nltk.download('stopwords')")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
    
    print("Vectorizing text data...")
    X, vectorizer = vectorize_text(df['clean_text'])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Passive Aggressive Classifier...")
    pac_model = train_pac(X_train, y_train)
    
    print("Training SVM...")
    svm_model = train_svm(X_train, y_train)
    
    print("\nEvaluating models:")
    pac_accuracy, pac_f1, pac_cm = evaluate_model(pac_model, X_test, y_test, "Passive Aggressive Classifier")
    svm_accuracy, svm_f1, svm_cm = evaluate_model(svm_model, X_test, y_test, "Support Vector Machine")
    
    models_results = {
        'PAC': {'accuracy': pac_accuracy, 'f1': pac_f1, 'cm': pac_cm},
        'SVM': {'accuracy': svm_accuracy, 'f1': svm_f1, 'cm': svm_cm}
    }
    
    print("\nPlotting comparison charts...")
    plot_comparison(models_results)
    plot_confusion_matrices(models_results)
    
    best_model = pac_model if pac_accuracy > svm_accuracy else svm_model
    best_model_name = "PAC" if pac_accuracy > svm_accuracy else "SVM"
    
    print(f"\nSaving the best model ({best_model_name})...")
    save_model(best_model, vectorizer, best_model_name)

if __name__ == "__main__":
    main()
