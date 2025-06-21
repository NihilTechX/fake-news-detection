# Fake News Detection Web Application

A machine learning application that detects whether news articles are real or fake using NLP and classification models.

## Features

- Machine learning-based detection of fake news articles
- Modern, responsive web interface
- Real-time analysis with visual feedback
- Utilizes NLP techniques for text preprocessing
- Compares multiple classification models (Passive Aggressive Classifier and SVM)

## Technologies Used

- Python
- Flask
- scikit-learn
- NLTK
- HTML/CSS/JavaScript
- Pandas & NumPy

## Getting Started

### Prerequisites

- Python 3.7+
- Pip package manager

### Installation

1. Clone the repository
   ```
   git clone https://github.com/yourusername/fake-news-detection.git
   cd fake-news-detection
   ```

2. Create a virtual environment and activate it
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install dependencies
   ```
   pip install -r requirements.txt
   ```

### Dataset

The system requires two datasets:
- `True.csv`: Contains real news articles
- `Fake.csv`: Contains fake news articles

Place these files in a directory called `News _dataset` in the parent directory.

You can download suitable datasets from:
- [Kaggle Fake News Dataset](https://www.kaggle.com/c/fake-news/data)
- [Kaggle Real or Fake News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

### Usage

1. Train the model:
   ```
   python fake_news_detector.py
   ```

2. Start the web application:
   ```
   python app.py
   ```

3. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

- `fake_news_detector.py`: Main script for training and evaluating models
- `app.py`: Flask web application
- `predict_news.py`: CLI script for predictions
- `templates/`: HTML templates
- `static/`: Static files (images, etc.)
- `requirements.txt`: Project dependencies

## Model Performance

The system compares two machine learning models:
- Passive Aggressive Classifier (PAC)
- Support Vector Machine (SVM)

Performance metrics include accuracy, F1 score, and confusion matrices.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Tamizhan Skills for the project inspiration
- [NLTK](https://www.nltk.org/) for natural language processing tools
- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
