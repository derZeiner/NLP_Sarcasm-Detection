import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


def predict_label(input_text):
    # Tokenize the input text using the previously defined tokenizer
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts([input_text])
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=200, padding='post')

    # Make predictions using the loaded model
    prediction = model.predict(padded_input_sequence)[0][0]

    print("--------------------------------------------")
    print(prediction)

    # Convert the prediction to a sentiment label
    sentiment_label = 'positive' if prediction >= 0.5 else 'negative'
    print(input_text)
    # Print the prediction
    print(f'The model predicts that the input text has a {sentiment_label} sentiment.')
    print("--------------------------------------------")


# Test the model with some example inputs
statements = [
    "Oh, great! Another meeting that could have been an email.",
    "Team collaboration is essential for success.",
    "I just love it when people talk loudly on their phones in public.",
    "Respecting others' personal space promotes a positive atmosphere.",
    "I can't wait to try the latest fad diet that's definitely going to work this time.",
    "Maintaining a balanced diet and regular exercise contributes to a healthy lifestyle.",
    "I'm so glad my computer decided to update right in the middle of my presentation.",
    "Regular software updates help keep our devices secure and efficient.",
    "Sure, I'd love to hear your opinion on a topic you know nothing about.",
    "Seeking expert advice ensures informed decision-making.",
    "Isn't it just wonderful when people don't use their turn signals?",
    "Using turn signals enhances road safety and communication between drivers.",
]

# Define the path to the trained model
model_path = '../../models/scratch_sarcasm_model.h5'

# Load the trained model from disk
model = tf.keras.models.load_model(model_path)

for string in statements:
    # Predict the label of the input string
    predict_label(string)
