import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ------------------------------
# Load saved resources
# ------------------------------
@st.cache_resource
def load_resources():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_resources()

# ------------------------------
# Predict next word
# ------------------------------
def predict_next_word(model, tokenizer, text, max_len):
    sequence = tokenizer.texts_to_sequences([text])[0]

    sequence = pad_sequences(
        [sequence],
        maxlen=max_len - 1,
        padding="pre"
    )

    prediction = model.predict(sequence, verbose=0)

    predicted_index = np.argmax(prediction)

    return tokenizer.index_word.get(predicted_index, "")


# ------------------------------
# Generate text
# ------------------------------
def generate_text(model, tokenizer, seed_text, max_len, n_words):
    generated_text = seed_text

    for _ in range(n_words):

        next_word = predict_next_word(
            model,
            tokenizer,
            generated_text,
            max_len
        )

        if next_word == "":
            break

        generated_text += " " + next_word

    return generated_text


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🧠 Next Word Prediction using LSTM")

st.markdown("""
This application predicts the next word(s) from a given text using a **Long Short-Term Memory (LSTM)** neural network.

### 📖 About the Project
- Dataset: **3,038 inspirational quotes**
- Task: **Next Word Prediction**
- Best Model: **LSTM**
- Vocabulary Size: **10,000 words**

### ✨ How to Use
1. Enter the beginning of a sentence.
2. Choose how many words to generate.
3. Click **Generate Text**.
""")

with st.expander("📚 Click to see example prompts"):
    st.markdown("""
    - The world as we have created it is a process of our thinking it cannot be changed without changing our thinking
    - It is our choices Harry that show what we truly are far more than our abilities
    - A day without sunshine is like you know night
    - If we have no peace it is because we have forgotten that we belong to each other
    - If no one cares for you at all do you even really exist
""")


st.set_page_config(
    page_title="Next Word Prediction",
    page_icon="🧠",
    layout="centered"
)

st.write(
    "Enter some starting text and the model will generate the next words."
)

user_input = st.text_input(
    "✍️ Enter seed text",
    placeholder="Example: once upon a time"
)

num_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=30,
    value=10
)

if st.button("🚀 Generate Text"):

    if not user_input.strip():
        st.warning("Please enter some text.")
    else:

        generated = generate_text(
            model,
            tokenizer,
            user_input,
            max_len,
            num_words
        )

        st.success("Generated Text")

        st.write(generated)


# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("LSTM-based Text Generation using Streamlit")