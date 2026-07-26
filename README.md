# 🧠 Next Word Prediction using LSTM & SimpleRNN

A Deep Learning project that predicts the next word in a sentence using **SimpleRNN** and **LSTM**. The model is trained on a collection of inspirational quotes and deployed using **Streamlit** for interactive text generation.

---

## 📌 Project Overview

The objective of this project is to build a language model capable of predicting the next word given an input sequence. Two recurrent neural network architectures were implemented and compared:

- SimpleRNN
- LSTM (Long Short-Term Memory)

Both models were trained on the same dataset, and the **LSTM model achieved better performance**, making it the final model used in the application.

---

## 🚀 Features

- Next word prediction
- Generate multiple words from a seed sentence
- Compare SimpleRNN and LSTM
- Interactive Streamlit web application
- Trained tokenizer and reusable model

---

## 📊 Dataset

The dataset contains **3,038 inspirational quotes**.

Example quotes:

> A lady's imagination is very rapid; it jumps from admiration to love...

> There are only two ways to live your life...

> Try not to become a man of success...

---

## 🧹 Data Preprocessing

Before training, the dataset was cleaned by:

- Converting all text to lowercase
- Removing punctuation
- Tokenizing every sentence into integer word IDs

Example:

Original sentence

```
Try not to become a man of success.
```

After preprocessing

```
try not to become a man of success
```

Tokenized

```
[15, 89, 7, 231, 5, 44, 92]
```

---

## 🔄 Creating Training Samples

Instead of using an entire sentence as one training sample, every sentence was divided into multiple **input-output pairs**.

```python
for seq in sequence:
    for i in range(1, len(seq)):
        input_seq = seq[:i]
        output_seq = seq[i]
        X.append(input_seq)
        y.append(output_seq)
```

Example:

```
Sentence

i love ai
```

Training samples:

| Input | Target |
|--------|--------|
| i | love |
| i love | ai |

This process expanded the dataset from **3,038 quotes** to **85,271 training samples**.

---

## 📏 Padding

Every training sequence has a different length.

To make all sequences the same size, left padding was applied.

Example

```
[1]

↓

[0 0 0 ... 0 1]
```

```
[1 2]

↓

[0 0 ... 1 2]
```

Maximum sequence length after preprocessing:

```
745
```

Final input shape

```
(85271, 745)
```

---

## 🎯 One-Hot Encoding

The target word was converted into a one-hot encoded vector using

```python
to_categorical(y, num_classes=10000)
```

Output shape

```
(85271,10000)
```

where **10,000** represents the vocabulary size.

---

# 🧠 Model Architecture

Both models use the same architecture except for the recurrent layer.

### Embedding Layer

```python
Embedding(
    input_dim=10000,
    output_dim=50,
    input_length=max_len
)
```

- Vocabulary Size = **10,000**
- Embedding Dimension = **50**

The embedding layer converts every word ID into a dense vector of **50 features**.

Example

```
Word ID

15

↓

[-0.23, 0.41, ..., 0.72]
```

After embedding

```
(85271,745)

↓

(85271,745,50)
```

---

## 🔹 SimpleRNN Model

```python
rnn_model = Sequential()

rnn_model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=50,
        input_length=max_len
    )
)

rnn_model.add(SimpleRNN(units=128))

rnn_model.add(Dense(
    units=vocab_size,
    activation="softmax"
))
```

### Explanation

- Embedding converts every word into a 50-dimensional vector.
- SimpleRNN processes one word at a time.
- Hidden layer contains **128 neurons**.
- The final hidden state is passed to the Dense layer.
- Dense predicts the probability of every word in the vocabulary.

---

## 🔹 LSTM Model

```python
lstm_model = Sequential()

lstm_model.add(
    Embedding(
        input_dim=vocab_size,
        output_dim=50,
        input_length=max_len
    )
)

lstm_model.add(LSTM(units=128))

lstm_model.add(Dense(
    units=vocab_size,
    activation="softmax"
))
```

LSTM uses memory cells and gates that help preserve long-term context, making it more effective than SimpleRNN for text generation.

---

## ⚙️ Model Compilation

Both models were compiled using

```python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

- **Adam** optimizes the network weights.
- **Categorical Crossentropy** measures prediction error.
- **Accuracy** evaluates prediction performance.

---

## 💾 Saving the Model

The trained tokenizer and model were saved for deployment.

```python
lstm_model.save("lstm_model.h5")

pickle.dump(tokenizer, open("tokenizer.pkl","wb"))

pickle.dump(max_len, open("max_len.pkl","wb"))
```

---

# 🔮 Prediction Pipeline

For prediction, the tokenizer converts user text into integer IDs.

```
what are you

↓

[45, 18, 92]
```

The sequence is padded before being passed to the LSTM model.

```python
pred = model.predict(seq)

pred_index = np.argmax(pred)
```

The predicted index is converted back into a word using

```python
index_to_word[index]
```

To generate multiple words, the predicted word is appended back to the input repeatedly.

```
Input

what are you

↓

what are you doing

↓

what are you doing today

↓

...
```

---

# 📈 Understanding the Shapes

Training Data

```
(85271,745)
```

After Embedding

```
(85271,745,50)
```

After LSTM

```
(85271,128)
```

Dense Output

```
(85271,10000)
```

---

## 💡 Important Note

A common misconception is that the model contains

```
85271 × 745 × 50 × 128
```

weights.

This is **incorrect**.

The shape above represents the **training data**, not the model parameters.

The actual trainable parameters depend only on the model architecture.

For this project:

```
Embedding

10000 × 50

Input → Hidden

50 × 128

Hidden → Hidden

128 × 128

Hidden → Output

128 × 10000
```

The same weights are **shared across all 85,271 training samples** and **every time step**, which is why the number of parameters remains fixed regardless of the dataset size.

---

## 🖥️ Streamlit Application

The trained LSTM model is deployed using Streamlit.

Users can:

- Enter a seed sentence
- Choose the number of words to generate
- Predict the next word(s) interactively

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pickle
- Streamlit
- Scikit-learn

---

## ▶️ Run Locally

```bash
git clone <repository-url>

cd next_word_prediction

pip install -r requirements.txt

streamlit run app.py
```

---

## 🌟 Future Improvements

- Bidirectional LSTM
- GRU
- Attention Mechanism
- Transformer Models
- Temperature Sampling
- Top-k Sampling
- Larger Dataset

---

## 👨‍💻 Author

**Samadhan Mane**

B.Tech Computer Engineering Student

MIT Academy of Engineering (MITAOE)

Deep Learning | Machine Learning | Full Stack Development