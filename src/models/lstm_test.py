import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Example data: sequences and labels
sequences = [
    [1, 2, 3],
    [4, 5],
    [6, 7, 8, 9],
    [10]
]

labels = [0, 1, 0, 1]  # Binary classification

# Convert sequences to a fixed length by padding or truncating
max_seq_length = max(len(seq) for seq in sequences)
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_seq_length)

# Convert labels to one-hot encoding
num_classes = len(set(labels))
labels = to_categorical(labels, num_classes=num_classes)

# Define the LSTM model
model = Sequential()
model.add(Embedding(input_dim=10, output_dim=32, input_length=max_seq_length))  # Assuming vocabulary size is 10
model.add(LSTM(64, return_sequences=False))
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
model.fit(sequences, labels, epochs=10, batch_size=2)

# Evaluate the model
loss, accuracy = model.evaluate(sequences, labels)
print(f'Final Loss: {loss}')
print(f'Final Accuracy: {accuracy}')
