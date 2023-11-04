import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Sample training data
X = ['Berlin London', 'cow cat', 'pink yellow']
y = ['capitals', 'animals', 'colors']

# Tokenize the input text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_sequences = tokenizer.texts_to_sequences(X)

# Get word index and calculate the maximum sequence length
word_index = tokenizer.word_index
max_sequence_length = max([len(sentence.split()) for sentence in X])

# Pad sequences to the same length
X_padded = pad_sequences(X_sequences, maxlen=max_sequence_length)

# Load pre-trained word embeddings (GloVe)
embeddings_index = {}
with open('glove.6B.50d.txt', 'r', encoding='utf-8') as file:
    for line in file:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

# Create the embedding matrix
embedding_dim = 50
embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Create a simple feedforward neural network model
model = Sequential()
model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length, trainable=False))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes (capitals, animals, colors)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical

# Transform labels into one-hot encoded format
label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y)
y_categorical = to_categorical(y_one_hot)

model.fit(X_padded, y_categorical, epochs=10, batch_size=1)

# New, previously unseen words
test_X = ['dog', 'red', 'Madrid']
test_sequences = tokenizer.texts_to_sequences(test_X)
test_padded = pad_sequences(test_sequences, maxlen=max_sequence_length)

# Predict the categories
predictions = model.predict(test_padded)
predicted_labels = [y[np.argmax(pred)] for pred in predictions]

print(predicted_labels)
