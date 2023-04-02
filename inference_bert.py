import transformers
import numpy as np
import tensorflow as tf

# Load the tokenizer and model
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# define custom object scope for TFDistilBertModel
custom_objects = {'TFDistilBertModel': transformers.TFDistilBertModel}

# Load the trained model from disk
model = tf.keras.models.load_model('train-BERT-model.h5', custom_objects=custom_objects)

# Define the maximum sequence length
MAX_SEQ_LENGTH = 20


# Define a function to convert a single string input to BERT input features
def convert_string_to_bert_input(string):
    tokens = tokenizer.tokenize(string)

    if len(tokens) > MAX_SEQ_LENGTH - 2:
        tokens = tokens[0: (MAX_SEQ_LENGTH - 2)]
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    ids = tokenizer.convert_tokens_to_ids(tokens)
    masks = [1] * len(ids)

    # Zero-pad up to the sequence length.
    while len(ids) < MAX_SEQ_LENGTH:
        ids.append(0)
        masks.append(0)

    encoded = np.array([ids, masks])
    return encoded


# Define a function to predict the label of a single string input
def predict_label(string):
    bert_input = convert_string_to_bert_input(string)
    prediction = model.predict([bert_input[0:1], bert_input[1:2]])
    label = 'Sarcastic' if prediction > 0.5 else 'Not sarcastic'
    print(f"Input: {string}")
    print(f"Prediction: {label} ({prediction[0][0]:.4f})\n")


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

for string in statements:
    predict_label(string)
