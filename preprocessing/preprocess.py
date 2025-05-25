import re
import string

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    
    text = re.sub(r'<[^>]+>', '', text)
    
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    text = re.sub(r'\S+@\S+', '', text)
    
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    text = ' '.join(text.split())
    
    return text
