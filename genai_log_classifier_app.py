import streamlit as st
import joblib
import re
import os
from google import genai
from google.genai.errors import APIError

# --- Configuration & Setup ---

# TEMPORARY KEY HANDLING: Replace this placeholder with your actual, valid key.
# This eliminates environment variable issues during development/testing.
TEMP_HARDCODED_KEY = "AIzaSyC9G-MXywx5UNDCN5HKrofS3W0NO0T3e9w"

# Try to get the key from the environment first (best practice), 
# but fall back to the hardcoded key if the environment variable is not set.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", TEMP_HARDCODED_KEY) 

MODEL_PATH = "log_classifier.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

# --- Model & Client Initialization ---
@st.cache_resource
def load_ml_assets():
    """Loads the trained ML model and vectorizer."""
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    except FileNotFoundError:
        st.error(f"**Critical Error:** Model files ({MODEL_PATH}, {VECTORIZER_PATH}) not found.")
        st.warning("Please run **`train_model.py`** first to create the necessary files.")
        st.stop()

model, vectorizer = load_ml_assets()

# Initialize the GenAI Client (must be done outside of Streamlit's main execution flow)
def initialize_genai_client(api_key):
    """Initializes the GenAI Client."""
    # Check if the key is the placeholder or missing
    if not api_key or api_key == "YOUR_ACTUAL_VALID_KEY_HERE":
        return None
    try:
        # The client automatically picks up the key from the environment variable or the provided string
        return genai.Client(api_key=api_key)
    except Exception:
        return None

genai_client = initialize_genai_client(GEMINI_API_KEY)

# --- Utility Functions ---

def clean_text(text):
    """Cleans log message (must match training script's preprocessing)."""
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def generate_remediation(log_message, prediction):
    """Calls Gemini API using the initialized client."""
    system_prompt = (
        "You are an expert DevOp engineer assistant. Your task is to analyze a system log and "
        "provide a concise, professional explanation of the issue and a clear, three-step remediation plan. "
        "Do not include any greeting or conversational filler. Output should be formatted using bolding and bullet points."
    )
    
    user_query = f"Analyze the following log message which was classified as a {prediction} and provide an explanation and remediation steps: '{log_message}'"

    try:
        response = genai_client.models.generate_content(
            model='gemini-2.5-flash-preview-09-2025',
            contents=user_query,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[{"google_search": {}}]
            )
        )
        return response.text
        
    except APIError as e:
        # Specific handling for 403 error for better feedback
        if '403' in str(e):
             return "GenAI API Error (403 Forbidden): Please check your API Key's validity and permissions."
        return f"An API Error occurred: {e}"
    except Exception as e:
        return f"An unexpected error occurred during GenAI call: {e}"

# --- Streamlit Main App ---

def main():
    st.set_page_config(
        page_title="GenAI Log Classification System", 
        page_icon="🤖", 
        layout="wide" # Use 'wide' layout for a professional look
    )

    st.title("GenAI Log Classification System")
    st.markdown("---")

    # Display API Key status in a sidebar warning/success area
    with st.sidebar:
        st.subheader("Configuration")
        if not genai_client:
            st.error(
                f"""
                **⚠️ Authentication Failed** (403 likely)!
                Please check your API Key in the `TEMP_HARDCODED_KEY` variable.
                """
            )
        else:
            st.success("✅ Gemini API Client Initialized.")
        
        st.markdown("---")
        st.info("This system uses **Naïve Bayes** for classification and **Gemini 2.5 Flash** for remediation.")


    # Color mapping for UI elements
    color_map = {
        "ERROR": ("#FEE2E2", "#B91C1C"), # Light Red / Dark Red
        "WARNING": ("#FEF3C7", "#92400E"), # Light Yellow / Dark Orange
        "INFO": ("#D1FAE5", "#065F46"),   # Light Green / Dark Green
        "DEBUG": ("#DBEAFE", "#1D4ED8")   # Light Blue / Dark Blue
    }

    # Use columns to position the input area
    col1, col2 = st.columns([3, 1])

    with col1:
        log_input = st.text_area(
            "Enter Log Message to Analyze:", 
            placeholder="e.g., Error: database connection pool is exhausted",
            height=200
        )
    
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True) # Vertical spacing
        if st.button("🚀 Analyze Log Message", type="primary", use_container_width=True):
            if not log_input.strip():
                st.warning("Please enter a log message.")
                return
            st.session_state['run_analysis'] = True
        
        # Reset button
        if st.button("🔄 Clear Results", use_container_width=True):
            if 'run_analysis' in st.session_state:
                del st.session_state['run_analysis']
            st.rerun()

    
    # Check if analysis button was pressed
    if st.session_state.get('run_analysis'):
        
        # --- Multi-step processing for improved professionalism ---
        
        # 1. Core ML Classification
        with st.spinner("Step 1/2: Running ML Classification..."):
            cleaned_log = clean_text(log_input)
            
            if not cleaned_log:
                prediction = "INFO"
            else:
                log_vector = vectorizer.transform([cleaned_log])
                prediction = model.predict(log_vector)[0]
        
        # Display Results in organized tabs
        tab_ml, tab_genai, tab_raw = st.tabs(["📊 ML Classification", "💡 Remediation Plan", "📄 Raw Input"])
        
        # --- Tab 1: ML Classification ---
        with tab_ml:
            bg_color, text_color = color_map.get(prediction, ("#E5E7EB", "#1F2937"))
            
            st.markdown(f"""
                <div style="background-color: {bg_color}; 
                            padding: 25px; 
                            border-radius: 12px; 
                            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); 
                            border-left: 8px solid {text_color}; margin-top: 10px;">
                    <h3 style="color: {text_color}; margin: 0;">
                        Classification Result: 
                        <span style="font-weight: 800; font-size: 2.2rem; margin-left: 15px;">{prediction}</span>
                    </h3>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional context based on prediction
            if prediction in ["ERROR", "WARNING"]:
                st.warning("A critical log was detected. Proceeding to GenAI for detailed remediation...")
            else:
                st.info("Log classified as non-critical. Remediation analysis is optional.")

        # --- Tab 2: GenAI Remediation Plan ---
        with tab_genai:
            if prediction in ["ERROR", "WARNING"]:
                if not genai_client:
                    st.error("Cannot generate remediation plan. API Client failed to initialize.")
                else:
                    with st.spinner(f"Step 2/2: Generating Remediation Plan for **{prediction}**..."):
                        genai_output = generate_remediation(log_input, prediction)
                    
                    st.success("Analysis Complete!")
                    st.subheader("Expert DevOp Assistant Analysis")
                    st.markdown(genai_output)
                
            else:
                st.info("GenAI analysis is typically reserved for ERROR or WARNING logs.")

        # --- Tab 3: Raw Input ---
        with tab_raw:
            st.code(log_input, language='text')
            st.subheader("Pre-processed Log Text")
            st.code(cleaned_log, language='text')


if __name__ == '__main__':
    main()
