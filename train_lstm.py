import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("hindipoems.txt", encoding="utf-8") as f:
    data = f.read()

lines = [line.strip() for line in data.split("\n") if line.strip()]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in lines:
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        input_sequences.append(tokens[:i+1])

max_seq_len = max(len(seq) for seq in input_sequences)
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_seq_len, padding="pre"))
X, y = input_sequences[:,:-1], tf.keras.utils.to_categorical(input_sequences[:,-1], num_classes=total_words)

model = Sequential([
    Embedding(total_words, 64, input_length=max_seq_len-1),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(total_words, activation="softmax")
])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(X, y, epochs=50, verbose=2)
model.save("lstm_hindi_poem.h5")
