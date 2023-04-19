# NLP_Sarcasm-Detection
authors: Paul Hassanpour, Michael Zeiner

## Description
This is a nlp project for sarcasm detection. We use the sarcasm dataset from kaggle, which includes headlines from huffingtonpost and the onion. Based on this data we want to predict if a headline uses sarcasm.

## How to use
1. extract the data from the zip file
2. install the requirements with `pip install -r requirements.txt`
3. run either train_pretrained_bert_model.py, train_scratch_kfold.py or train_scratch_sarcasm_model.py or all together/after each other
4. The model will be placed in /models or download the models from https://1drv.ms/f/s!AiGCrp9BWsG_h6xHpLljIFUPf83bmg?e=kL27Lz
5. After training your model you can run the inference files for each specific model in /testing_model
   1. inference uses 10 statements created with chatgpt and 10 headlines not from the dataset
   2. both split 50/50 between sarcastic and not sarcastic
   3. it will run through all strings and print the prediction for each one
   4. at the end you will get the results for the whole inference set