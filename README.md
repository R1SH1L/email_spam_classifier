# Email Spam Classifier

A machine learning-based email spam classifier built with Python and Streamlit.

## Features

- Text preprocessing and feature extraction using TF-IDF
- Logistic Regression model with balanced class weights
- Interactive web interface using Streamlit
- Real-time spam detection with confidence scores

## Setup

1. Create virtual environment and install dependencies:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Download the SMS Spam Collection dataset from Kaggle and place it as `spam.csv` in the `data/` folder.

## Usage

1. Train the model:
```bash
python model/train_model.py
```

2. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

## Model Performance

- Accuracy: ~98.5%
- Precision (Spam): ~95%
- Recall (Spam): ~94%

## Project Structure

```
email-spam-classifier/
├── app/
│   └── streamlit_app.py      # Streamlit web interface
├── model/
│   ├── train_model.py        # Model training script
│   └── model.pkl             # Trained model (generated)
├── preprocessing/
│   └── preprocess.py         # Text preprocessing utilities
├── data/
│   └── spam.csv             # Dataset (download separately)
├── requirements.txt          # Python dependencies
└── README.md                # Project documentation
```

## Technical Details

### Preprocessing
- Text normalization (lowercase)
- HTML tag removal
- URL and email removal
- Non-alphabetic character removal
- Whitespace normalization

### Model
- TF-IDF vectorization with n-grams (1-3)
- Logistic Regression with balanced class weights
- Cross-validation for optimal hyperparameters

