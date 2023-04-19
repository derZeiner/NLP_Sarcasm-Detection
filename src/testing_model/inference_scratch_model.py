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


def predict_label(_string, cl):
    # Tokenize the input text using the previously defined tokenizer
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts([_string])
    input_sequence = tokenizer.texts_to_sequences([_string])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=200, padding='post')

    # Make predictions using the loaded model
    prediction = model.predict(padded_input_sequence)[0][0]

    label = 'Sarcastic' if prediction > 0.5 else 'Not sarcastic'
    class_label = 'Sarcastic' if cl == 1 else 'Not sarcastic'
    print("--------------------------------------------")
    print(f"Input: {_string}")
    print(f"Prediction: {label} ({prediction:.4f})")
    print(f"Correct label: {class_label} ({cl})\n")
    print("--------------------------------------------")
    return prediction


def run_inference():
    prediction_statements = []
    prediction_headlines = []
    # Predict the label of the example inputs
    for i in range(len(statements)):
        prediction_statements.append(predict_label(statements[i], classes[i]))
    for i in range(len(headlines)):
        prediction_headlines.append(predict_label(headlines[i], classes[i]))
    # compare the predictions with the correct labels and calculate the accuracy
    correct_statements = 0
    correct_headlines = 0
    correct = 0
    for i in range(len(prediction_statements)):
        if classes[i] == 0:
            if prediction_statements[i] < 0.5:
                correct += 1
                correct_statements += 1
        else:
            if prediction_statements[i] > 0.5:
                correct += 1
                correct_statements += 1
    for i in range(len(prediction_headlines)):
        if classes[i] == 0:
            if prediction_headlines[i] < 0.5:
                correct += 1
                correct_headlines += 1
        else:
            if prediction_headlines[i] > 0.5:
                correct += 1
                correct_headlines += 1
    # calculate accuracy seperate for headlines and statements
    accuracy_statements = correct_statements / len(prediction_statements)
    accuracy_headlines = correct_headlines / len(prediction_headlines)
    print(f"Accuracy statements: {accuracy_statements:.4f}")
    print(f"Accuracy headlines: {accuracy_headlines:.4f}")
    # calculate the overall accuracy
    accuracy = (accuracy_statements + accuracy_headlines) / 2
    print(f"Accuracy overall: {accuracy:.4f}")


# Test the model with some example inputs - created with chatGPT 4
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
    # https://www.theonion.com/man-delays-exit-from-burning-house-to-avoid-small-talk-1850313309
    "half of congressional republicans want abortion pill ruling to stand",
    # https://www.huffpost.com/entry/republicans-congress-abortion-pill-supreme-court_n_643f0e0ce4b03c1b88c35c76
    "conservatives boycott computers after noticing keyboard can be used to type 'trans'",
    # https://www.theonion.com/conservatives-boycott-computers-after-noticing-keyboard-1850337910
    "at least 1 dead, 5 injured in nyc parking garage collapse",
    # https://www.huffpost.com/entry/new-york-city-parking-garage-collapse_n_643f0e12e4b04997b56de791
    "mormon whishes he drank beer so he could boycott bud light",
    # https://babylonbee.com/news/mormon-wishes-he-drank-beer-so-he-could-boycott-bud-light
    "jerusalem christians say attacks on the rise",  # https://www.bbc.com/news/world-65204037
    "missouri now requiring all residents to have license, permit to operate doorbell",
    # https://www.theonion.com/missouri-now-requiring-all-residents-to-have-license-p-1850348783
    "man 'eaten alive' by bed bugs in US jail",  # https://www.bbc.com/news/world-us-canada-65267971
    "'enough with blood, we want skin!': the american red cross has announced that while they loved collecting your blood, they are more interested in skin now",
    # https://clickhole.com/enough-with-blood-we-want-skin-the-american-red-cross-has-announced-that-while-they-loved-collecting-your-blood-they-are-more-interested-in-skin-now/
    "former bud light drinkers say 'too little, too late' after brand tries to make amends with pro-america ad"
    # https://www.foxnews.com/us/former-bud-light-drinkers-say-little-late-brand-tries-make-amends-pro-america-ad#&_intcmp=fnhpbt4
]

classes = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# Define the path to the trained model
model_path = '../../models/scratch_sarcasm_model.h5'

# Load the trained model from disk
model = tf.keras.models.load_model(model_path)

run_inference()
