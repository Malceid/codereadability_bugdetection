import streamlit as st
import tensorflow as tf
import numpy as np
import lizard
import joblib
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Constants for Bug Prediction
BUG_MODEL_PATH = "bug_prediction.keras"
BUG_TOKENIZER_PATH = "tokenizer_bugprediction.json"
MAX_CHARS = 5000
MAX_LEN = 200

# Constants for Complexity Prediction
COMPLEXITY_MODEL_PATH = "code_complexity_model.h5"
LABEL_ENCODER_PATH = "label_encoder.joblib"       # Updated to .joblib
BIG_O_ENCODER_PATH = "big_o_encoder.joblib"       # Updated to .joblib
BIG_O_LABEL_ENCODER_PATH = "big_o_label_encoder.joblib"  # Updated to .joblib
MAX_COMPLEXITY_PATH = "max_complexity.txt"

# Cache loading functions
@st.cache_resource
def load_bug_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Bug prediction model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Bug model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading bug model: {e}")
        return None

@st.cache_resource
def load_bug_tokenizer(tokenizer_path):
    try:
        with open(tokenizer_path, 'r') as f:
            data = json.load(f)
            tokenizer = tokenizer_from_json(data)
        st.success("Bug prediction tokenizer loaded successfully!")
        return tokenizer
    except FileNotFoundError:
        st.error(f"Error: Bug tokenizer file not found at {tokenizer_path}")
        return None
    except Exception as e:
        st.error(f"Error loading bug tokenizer: {e}")
        return None

@st.cache_resource
def load_complexity_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        st.success("Complexity model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Error: Complexity model file not found at {model_path}")
        return None
    except Exception as e:
        st.error(f"Error loading complexity model: {e}")
        return None

@st.cache_resource
def load_complexity_resources():
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        big_o_encoder = joblib.load(BIG_O_ENCODER_PATH)
        big_o_label_encoder = joblib.load(BIG_O_LABEL_ENCODER_PATH)
        with open(MAX_COMPLEXITY_PATH, 'r') as f:
            max_complexity = float(f.read())
        st.success("Complexity resources loaded successfully!")
        return label_encoder, big_o_encoder, big_o_label_encoder, max_complexity
    except FileNotFoundError as e:
        st.error(f"Error: Complexity resource file not found - {e}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading complexity resources: {e}")
        return None, None, None, None

# Prediction functions
def predict_bug(snippet, model, tokenizer, max_len=MAX_LEN):
    sequence = tokenizer.texts_to_sequences([snippet])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0]  # [P(Fixed), P(Buggy)]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]
    label = "Buggy" if predicted_class == 1 else "Fixed"
    return label, confidence, prediction

def get_cyclomatic_complexity(code):
    analysis = lizard.analyze_file.analyze_source_code("temp.java", code)
    function_complexities = [func.cyclomatic_complexity for func in analysis.function_list]
    return max(function_complexities) if function_complexities else 1

def predict_complexity(code, model, max_complexity, label_encoder, big_o_encoder, big_o_label_encoder):
    raw_complexity = get_cyclomatic_complexity(code)
    example_complexity = np.array([raw_complexity]) / max_complexity
    example_complexity = example_complexity.reshape(-1, 1, 1)
    prediction = model.predict(example_complexity)
    return raw_complexity, example_complexity, prediction

# Streamlit app
def main():
    st.title("Code Analysis App")

    # Load bug prediction resources
    bug_model = load_bug_model(BUG_MODEL_PATH)
    bug_tokenizer = load_bug_tokenizer(BUG_TOKENIZER_PATH)

    # Load complexity prediction resources
    complexity_model = load_complexity_model(COMPLEXITY_MODEL_PATH)
    label_encoder, big_o_encoder, big_o_label_encoder, max_complexity = load_complexity_resources()

    # Check if loading was successful
    if any(x is None for x in [bug_model, bug_tokenizer, complexity_model, label_encoder, big_o_encoder, big_o_label_encoder, max_complexity]):
        st.error("Application cannot proceed due to loading errors.")
        st.stop()

    # Text area for code input
    code_snippet = st.text_area("Enter your Java code snippet:", height=200, placeholder="e.g., public static int sum(int a, int b) { return a + b; }")

    # Prediction button
    if st.button("Analyze"):
        if code_snippet.strip():
            try:
                # Bug prediction
                bug_label, bug_confidence, bug_probs = predict_bug(code_snippet, bug_model, bug_tokenizer)
                
                # Complexity prediction
                raw_complexity, norm_complexity, complexity_pred = predict_complexity(
                    code_snippet, complexity_model, max_complexity, label_encoder, big_o_encoder, big_o_label_encoder
                )

                # Display Bug Prediction Results
                st.subheader("Bug Prediction Results:")
                st.write(f"Prediction: {bug_label} (Confidence: {bug_confidence:.4f})")
                st.write(f"Raw Probabilities: Fixed={bug_probs[0]:.4f}, Buggy={bug_probs[1]:.4f}")

                # Display Complexity Prediction Results
                st.subheader("Complexity Prediction Results:")
                st.write(f"Raw Cyclomatic Complexity Score: {raw_complexity}")
                st.write(f"Normalized Complexity Score: {norm_complexity[0][0][0]:.4f}")
                st.write("Predicted Complexity Level:", label_encoder.inverse_transform([np.argmax(complexity_pred[0])])[0])
                st.write("Predicted Big-O Notation:", big_o_encoder.inverse_transform([np.argmax(complexity_pred[1])])[0])
                st.write("Predicted Big-O Label:", big_o_label_encoder.inverse_transform([np.argmax(complexity_pred[2])])[0])
                st.write("Predicted Readability:", "Readable" if complexity_pred[3][0] < 0.5 else "Unreadable")
            except Exception as e:
                st.error(f"Error during analysis: {e}")
        else:
            st.warning("Please enter a valid Java code snippet.")

if __name__ == "__main__":
    main()