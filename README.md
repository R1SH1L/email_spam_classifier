# Email Spam Classifier

A machine learning-based email spam classifier built with Python and Streamlit.

## Features

- Text preprocessing and feature extraction using TF-IDF
- Logistic Regression model with balanced class weights
- Interactive web interface using Streamlit
- Real-time spam detection with confidence scores

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/email-spam-classifier.git
cd email-spam-classifier
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset

Download the SMS Spam Collection dataset and place it as `spam.csv` in the `data/` folder.
The dataset should have two columns: `v1` (ham/spam labels) and `v2` (text content).

Dataset source: [UCI ML Repository - SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

## Usage

1. Train the model:
```bash
python model/train_model.py
```

2. Run the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

3. Open your browser and navigate to http://localhost:8501

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
├── data/
│   └── spam.csv             # Dataset (download separately)
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore rules
```

## Technical Details

### Preprocessing
- Text normalization (lowercase)
- URL and email removal
- Selective punctuation retention for spam indicators
- Whitespace normalization

### Model
- TF-IDF vectorization with n-grams (1-3)
- Logistic Regression with balanced class weights
- Cross-validation for optimal hyperparameters

## License

MIT License