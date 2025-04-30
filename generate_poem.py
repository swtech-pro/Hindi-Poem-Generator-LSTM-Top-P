import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
import random

with open("hindipoems.txt", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
total_words = len(tokenizer.word_index) + 1

max_seq_len = max([len(tokenizer.texts_to_sequences([line])[0]) for line in lines])

model = load_model("lstm_hindi_poem.h5")

def top_p_sampling(preds, p=0.9):
    sorted_indices = np.argsort(preds)[::-1]
    cumulative = np.cumsum(preds[sorted_indices])
    cutoff = cumulative > p
    if np.any(cutoff):
        sorted_indices = sorted_indices[:np.argmax(cutoff)+1]
    sorted_probs = preds[sorted_indices]
    sorted_probs /= sorted_probs.sum()
    return np.random.choice(sorted_indices, p=sorted_probs)

def generate_poem(seed, num_words=20, top_p_value=0.9):
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding="pre")
        preds = model.predict(token_list, verbose=0)[0]
        next_index = top_p_sampling(preds, p=top_p_value)
        next_word = tokenizer.index_word.get(next_index, "")
        seed += " " + next_word
    return seed

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else "प्रकृति की"
    print("📜", generate_poem(prompt))
