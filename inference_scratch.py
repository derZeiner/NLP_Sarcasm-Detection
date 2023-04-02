import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

# Load the trained model from disk
model = tf.keras.models.load_model('train-CNN-model.h5')

# Define some input text data
input_text = "my link to muhammad  ali through parkinson's"


# Tokenize the input text using the previously defined tokenizer
tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
tokenizer.fit_on_texts([input_text])
input_sequence = tokenizer.texts_to_sequences([input_text])
padded_input_sequence = pad_sequences(input_sequence, maxlen=100, padding='post')

# Make predictions using the loaded model
prediction = model.predict(padded_input_sequence)[0][0]

print(prediction)

# Convert the prediction to a sentiment label
sentiment_label = 'positive' if prediction >= 0.5 else 'negative'

# Print the prediction
print(f'The model predicts that the input text has a {sentiment_label} sentiment.')