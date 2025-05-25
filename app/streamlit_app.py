import streamlit as st
import joblib
import re
from pathlib import Path

# Configure page
st.set_page_config(page_title="Email Spam Classifier", page_icon="🚀")

# FIXED: Match the training preprocessing exactly
def preprocess_text(text):
    if not text:
        return ""
    
    text = str(text).lower()
    
    # Remove URLs and emails but keep other patterns
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Keep important punctuation and numbers for spam detection
    # Only remove excessive punctuation - SAME AS TRAINING
    text = re.sub(r'[^\w\s!$%]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Load model
@st.cache_resource
def load_model():
    try:
        parent_dir = Path(__file__).parent.parent
        model_path = parent_dir / 'model' / 'model.pkl'
        if model_path.exists():
            return joblib.load(model_path)
        else:
            # Try current directory
            if Path('model.pkl').exists():
                return joblib.load('model.pkl')
            st.error(f"Model not found at: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# UI
st.title("🚀 Email Spam Classifier")

email_text = st.text_area("Enter Email Content", height=150)

if st.button("🔍 Classify Email"):
    if email_text.strip():
        if model is None:
            st.error("❌ Model not found! Train the model first.")
        else:
            try:
                cleaned_text = preprocess_text(email_text)
                
                if not cleaned_text.strip():
                    st.warning("⚠️ No text left after preprocessing")
                else:
                    prediction = model.predict([cleaned_text])[0]
                    probability = model.predict_proba([cleaned_text])[0]
                    
                    # Debug info
                    st.write(f"**Debug:** Original: `{email_text[:100]}...`")
                    st.write(f"**Debug:** Cleaned: `{cleaned_text[:100]}...`")
                    st.write(f"**Debug:** Prediction: {prediction}, Probabilities: {probability}")
                    
                    if prediction == 1:
                        st.error("🚨 **SPAM DETECTED**")
                        confidence = probability[1]
                    else:
                        st.success("✅ **LEGITIMATE EMAIL**")
                        confidence = probability[0]
                    
                    st.metric("Confidence", f"{confidence:.1%}")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    else:
        st.warning("⚠️ Please enter email text.")

# Test examples moved to bottom
st.markdown("---")
with st.expander("📝 Test Examples"):
    st.write("**Spam Examples:**")
    st.code("URGENT! You've won $1000000! Click here NOW! Limited time offer!")
    st.code("FREE money! Call now! Act fast!")
    st.code("Congratulations! You have won $1000! Claim now!")
    st.write("**Ham Examples:**")
    st.code("Hi, can we schedule a meeting for tomorrow at 3pm?")
    st.code("Thanks for the document, I'll review it today.")
