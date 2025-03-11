import streamlit as st
import lizard
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the complexity model
@st.cache_resource
def load_complexity_model():
    return tf.keras.models.load_model('code_complexity_model.h5')

# Load the bug prediction model
@st.cache_resource
def load_bug_model():
    return tf.keras.models.load_model('bug_prediction.keras')

# Function to compute cyclomatic complexity
def get_cyclomatic_complexity(code):
    analysis = lizard.analyze_file.analyze_source_code("temp.java", code)
    function_complexities = [func.cyclomatic_complexity for func in analysis.function_list]
    return max(function_complexities) if function_complexities else 1

# Prediction function for bug detection
def predict_bug(snippet, model, tokenizer, max_len=100):  # Default MAX_LEN=100
    sequence = tokenizer.texts_to_sequences([snippet])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0]  # [P(Fixed), P(Buggy)]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    label = "Buggy" if predicted_class == 1 else "Fixed"
    st.write(f"Debug: Raw Probabilities: Fixed={prediction[0]:.4f}, Buggy={prediction[1]:.4f}")
    return label, confidence

# Streamlit app
def main():
    st.title("Java Code Complexity and Bug Analyzer")

    # Input area for Java code
    code = st.text_area("Paste your Java code here:", height=200)

    # Load tokenizer (assuming it's predefined or loaded; adjust as needed)
    # For now, I'll mock it - replace with your actual tokenizer loading logic
    try:
        from tensorflow.keras.preprocessing.text import Tokenizer
        tokenizer = Tokenizer(num_words=10000)  # Mock tokenizer; replace with your actual one
        tokenizer.fit_on_texts([code])  # Fit on sample data; load your trained tokenizer instead
    except Exception as e:
        st.error(f"Tokenizer setup failed: {str(e)}")
        return

    if st.button("Analyze Code"):
        if code.strip() == "":
            st.warning("Please enter some Java code to analyze.")
        else:
            # Load models
            complexity_model = load_complexity_model()
            bug_model = load_bug_model()

            # --- Complexity and Readability Analysis ---
            # Compute cyclomatic complexity
            raw_complexity = get_cyclomatic_complexity(code)

            # Normalize and reshape input for complexity model
            max_complexity = 100
            normalized_complexity = np.array([raw_complexity], dtype=np.float32) / max_complexity
            normalized_complexity = normalized_complexity.reshape(-1, 1, 1)

            # Make complexity predictions
            complexity_prediction = complexity_model.predict(normalized_complexity)
            readability_score = complexity_prediction[3][0][0]  # 4th output is readability
            readability = "Readable" if readability_score >= 0.5 else "Unreadable"

            # --- Bug Prediction ---
            bug_label, bug_confidence = predict_bug(code, bug_model, tokenizer, max_len=100)

            # Display results
            st.subheader("Analysis Results:")
            st.write(f"1. **Raw Cyclomatic Complexity:** {raw_complexity}")
            st.write(f"2. **Normalized Complexity Score:** {normalized_complexity[0][0][0]:.4f}")
            st.write(f"3. **Predicted Readability Score:** {readability_score:.4f}")
            st.write(f"4. **Predicted Readability:** {readability}")
            st.write(f"5. **Bug Prediction:** {bug_label} (Confidence: {bug_confidence:.4f})")

if __name__ == "__main__":
    main()