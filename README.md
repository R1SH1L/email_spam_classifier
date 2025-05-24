# Email Spam Classifier

A machine learning-based email spam classifier built with Python, scikit-learn, and Streamlit. This project uses Natural Language Processing (NLP) techniques to classify emails as spam or ham (not spam).

## Features

- **Text Preprocessing**: Removes URLs, punctuation, stop words, and applies lemmatization
- **Machine Learning**: Uses TF-IDF vectorization with Naive Bayes classifier
- **Web Interface**: Interactive Streamlit web application for real-time predictions
- **High Accuracy**: Achieves 97% accuracy on the test dataset

## Project Structure

```
email-spam-classifier/
│
├── app/
│   └── streamlit_app.py          # Streamlit web application
│
├── data/
│   └── spam.csv                  # Dataset (SMS Spam Collection)
│
├── model/
│   ├── train_model.py            # Model training script
│   └── model.pkl                 # Trained model (generated after training)
│
├── preprocessing/
│   ├── __init__.py               # Makes it a Python package
│   └── preprocess.py             # Text preprocessing functions
│
├── venv/                         # Virtual environment
├── requirements.txt              # Project dependencies
└── README.md                     # Project documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd email-spam-classifier
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (if not already downloaded)
   ```python
   import nltk
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## Usage

### 1. Train the Model

First, ensure you have the spam dataset in the `data/` directory, then run:

```bash
python model/train_model.py
```

This will:
- Load and preprocess the dataset
- Train a Naive Bayes classifier
- Evaluate the model performance
- Save the trained model to `model/model.pkl`

### 2. Run the Web Application

Launch the Streamlit web interface:

```bash
streamlit run app/streamlit_app.py
```

The application will open in your default web browser.

### 3. Using the Web Interface

1. Enter email text in the text area
2. Click the "Classify" button
3. View the prediction (Spam ❌ or Not Spam ✅)

## Quick Test

You can test the classifier with these sample emails:

**Spam Example:**
```
URGENT! You've won $1000000! Click here immediately to claim your prize! Limited time offer! Call now!
```

**Ham Example:**
```
Hi John, hope you're doing well. Let's schedule our meeting for next week. Best regards, Sarah.
```

## Model Performance

The trained model achieves the following performance metrics:

- **Overall Accuracy**: 97%
- **Ham (Not Spam)**: 96% precision, 100% recall
- **Spam**: 100% precision, 74% recall

## Technical Details

### Preprocessing Pipeline

1. **Text Cleaning**:
   - Convert to lowercase
   - Remove URLs and web links
   - Remove punctuation
   - Tokenization

2. **NLP Processing**:
   - Remove stop words
   - Lemmatization using WordNet

### Machine Learning Pipeline

1. **Feature Extraction**: TF-IDF Vectorization
2. **Classification**: Multinomial Naive Bayes
3. **Evaluation**: Classification report with precision, recall, and F1-score

## Dependencies

- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning library
- `nltk`: Natural Language Processing
- `streamlit`: Web application framework
- `joblib`: Model serialization

## Dataset

This project uses the SMS Spam Collection Dataset, which contains:
- 5,574 SMS messages
- Binary classification (spam/ham)
- Publicly available dataset