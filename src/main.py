import train_model.train_pretrained_bert_model as pbm
import train_model.train_scratch_sarcasm_model as ssm
import train_model.train_scratch_kfold as sskm


# ***********************************
# Function: Main function
# ***********************************
# which model do you want to train?
train_bpm = True
train_ssm = True
train_sskm = True

if __name__ == '__main__':
    if train_ssm:
        ssm.run()
    if train_sskm:
        sskm.run()
    if train_bpm:
        pbm.run()


        
