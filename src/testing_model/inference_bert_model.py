import transformers
import numpy as np
import tensorflow as tf


def convert_string_to_bert_input(_string):
    # ************************************************************
    # This function is adapted from the following source:
    # Define a function to convert a single string input to BERT input features
    # ************************************************************
    # Tokenize the input string
    tokens = tokenizer.tokenize(_string)
    # Define the maximum sequence length
    MAX_SEQ_LENGTH = 20
    # Truncate the sequence to the maximum length
    if len(tokens) > MAX_SEQ_LENGTH - 2:
        tokens = tokens[0: (MAX_SEQ_LENGTH - 2)]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    ids = tokenizer.convert_tokens_to_ids(tokens)
    masks = [1] * len(ids)
    # Zero-pad up to the sequence length.
    while len(ids) < MAX_SEQ_LENGTH:
        ids.append(0)
        masks.append(0)
    # Convert to numpy arrays
    encoded = np.array([ids, masks])
    return encoded


# Define a function to predict the label of a single string input
def predict_label(_string, cl):
    bert_input = convert_string_to_bert_input(_string)
    prediction = model.predict([bert_input[0:1], bert_input[1:2]])
    label = 'Sarcastic' if prediction > 0.5 else 'Not sarcastic'
    class_label = 'Sarcastic' if cl == 1 else 'Not sarcastic'
    print("--------------------------------------------")
    print(f"Input: {_string}")
    print(f"Prediction: {label} ({prediction[0][0]:.4f})")
    print(f"Correct label: {class_label} ({cl})\n")
    print("--------------------------------------------")


# Test the model with some example inputs
statements = [
    "Oh, great! Another meeting that could have been an email.",
    "Team collaboration is essential for success.",
    "I just love it when people talk loudly on their phones in public.",
    "Respecting others' personal space promotes a positive atmosphere.",
    "I'm so glad my computer decided to update right in the middle of my presentation.",
    "Regular software updates help keep our devices secure and efficient.",
    "Sure, I'd love to hear your opinion on a topic you know nothing about.",
    "Seeking expert advice ensures informed decision-making.",
    "Isn't it just wonderful when people don't use their turn signals?",
    "Using turn signals enhances road safety and communication between drivers.",
]

headlines = [
    "man delays exit from burning house to avoid small talk with neighbors",
    "gunfire and explosions as sudan army and paramilitary fight",
    "conservatives boycott computers after noticing keyboard can be used to type 'trans'",
    "athlete emerges after 500 days living in cave",
    "thousands of beef ribs fall from sky onto empty plates of texans who strapped on bib, prayed for dinner",
    "jerusalem christians say attacks on the rise",
    "federal reserve calls for more poverty",
    "man 'eaten alive' by bed bugs in US jail",
    "aging rock musician realizes it time to grow up and get a real job as jazz musician",
    "japan pm evacuated after apparent smoke bomb blast"
]

classes = [1,0,1,0,1,0,1,0,1,0]

# Load the tokenizer and model
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# define custom object scope for TFDistilBertModel
custom_objects = {'TFDistilBertModel': transformers.TFDistilBertModel}

# path to the saved model
model_path = '../../models/pretrained_bert_model.h5'

# Load the trained model from disk
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# Predict the label of the example inputs
for i in range(len(statements)):
    predict_label(statements[i], classes[i])

for i in range(len(headlines)):
    predict_label(headlines[i], classes[i])
