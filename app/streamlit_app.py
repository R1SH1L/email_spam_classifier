import streamlit as st
import joblib
import sys
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="🚀",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Add the parent directory to the Python path
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

try:
    from preprocessing.preprocess import preprocess_text
except ImportError:
    st.error("❌ Preprocessing module not found. Please check your project structure.")
    st.stop()

# Load model with better path handling
@st.cache_resource
def load_model():
    """Load the trained model with caching for better performance"""
    model_path = parent_dir / 'model' / 'model.pkl'
    
    if not model_path.exists():
        st.error("❌ Model file not found! Please train the model first.")
        st.info("Run: `python model/train_model.py` to train the model")
        st.stop()
    
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load model
model = load_model()

# Main UI
st.title("🚀 Email Spam Classifier")
st.markdown("""
**Classify emails as Spam or Ham (Not Spam) using Machine Learning**

Simply paste your email content below and click classify!
""")

# Input section
email_text = st.text_area(
    "Enter Email Content",
    placeholder="Paste your email text here...",
    height=200,
    help="Enter the email content you want to classify"
)

# Classification section
if st.button("🔍 Classify Email", type="primary"):
    if email_text.strip():
        try:
            with st.spinner("🔄 Analyzing email..."):
                # Preprocess text
                cleaned_text = preprocess_text(email_text)
                
                # Make prediction
                prediction = model.predict([cleaned_text])[0]
                probability = model.predict_proba([cleaned_text])[0]
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 **SPAM DETECTED**")
                        confidence = probability[1]
                    else:
                        st.success("✅ **LEGITIMATE EMAIL**")
                        confidence = probability[0]
                
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Additional info
                st.markdown("---")
                with st.expander("📊 View Prediction Details"):
                    st.write(f"**Spam Probability:** {probability[1]:.3f}")
                    st.write(f"**Ham Probability:** {probability[0]:.3f}")
                    st.write(f"**Processed Text Length:** {len(cleaned_text)} characters")
                    
        except Exception as e:
            st.error(f"❌ Error during classification: {str(e)}")
    else:
        st.warning("⚠️ Please enter some email text to classify.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <small>Built with Streamlit | Email Spam Classification using ML</small>
    </div>
    """, 
    unsafe_allow_html=True
)
