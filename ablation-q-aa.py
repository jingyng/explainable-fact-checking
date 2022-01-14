#!/usr/bin/env python
# coding: utf-8

import wandb

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="3"

import random
import numpy as np
import argparse
import time, os
import copy
import pickle
from random import sample
import torchvision
from sklearn.model_selection import train_test_split
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader, sampler
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import logging

from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio
from tqdm import tqdm
import pandas as pd


# seed = 42
# os.environ['PYTHONHASHSEED'] = str(seed)
# # Torch RNG
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# # Python RNG
# np.random.seed(seed)
# random.seed(seed)
# # # In[ ]:


from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead


class MyDataset(Dataset):
    def __init__(self, df, args):

        self.tokenizer1 = AutoTokenizer.from_pretrained(args.bert1)
        
        sent2 = list(zip(df.answer_text, df.answer_gold))
        
        
        self.cq_input = self.tokenizer1.batch_encode_plus(
            batch_text_or_text_pairs=df['question'].to_list(),  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = args.max_c_len,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )
        
        self.aa_input = self.tokenizer1.batch_encode_plus(
            batch_text_or_text_pairs=sent2,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = args.max_a_len,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )
        
        self.y_data = torch.tensor(df['label'].tolist(), dtype = torch.long)
        
        
    def __getitem__(self, index):
        return self.cq_input['input_ids'][index], self.cq_input['attention_mask'][index], self.aa_input['input_ids'][index], self.aa_input['attention_mask'][index], self.y_data[index]
    
    def __len__(self):
        return len(self.y_data)



def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()



class BERT_Fusion(nn.Module):
    def __init__(self, args):
        super(BERT_Fusion, self).__init__()
        self.bert1 = AutoModel.from_pretrained(args.bert1)
        
        for param in self.bert1.base_model.parameters():
            param.requires_grad = True
            
            
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1',  nn.Linear(2*self.bert1.config.hidden_size, 2))


    def forward(self, input_ids_cq, mask_cq, input_ids_aa, mask_aa):
        
        outputs_cq = self.bert1(input_ids_cq, attention_mask = mask_cq)
        outputs_aa = self.bert1(input_ids_aa, attention_mask = mask_aa)

        feature_cq = outputs_cq[1]
        feature_aa = outputs_aa[1]


        cqaa = torch.cat((feature_cq, feature_aa), 1)

        output = self.classifier(cqaa)
     
        return output

    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=None, type=str, required=True, 
                       help="run name for the experiment")
    parser.add_argument("--bert1", default='microsoft/mpnet-base', type=str, required=False, 
                       help="bert model name for claim and questions")
    parser.add_argument("--lr", default= 2e-5, type=float, required=False, 
                       help="learning rate")
    parser.add_argument("--bsz", default= 16, type=int, required=False, 
                       help="batch size")
    parser.add_argument("--data_path", default= './', type=str, required=False, 
                       help="input data path")
    parser.add_argument("--max_c_len", default= 128, type=int, required=False, 
                       help="maximum token length for claim and questions")
    parser.add_argument("--max_a_len", default= 128, type=int, required=False, 
                       help="maximum token length for answers")
    parser.add_argument("--num_workers", default= 4, type=int, required=False, 
                       help="number of workers")
    parser.add_argument("--save_model_path", default= './model/', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--epochs", default= 5, type=int, required=False, 
                       help="number of epochs")
    
    
    args = parser.parse_args()

    wandb.init(name = args.run_name, project='icassp2022', entity='fakejing', \
              tags = ['gold','final','electra'], config = args, save_code = True)
    

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)
    
    df_train_set = pd.read_pickle(args.data_path + 'qa-fool-me-twice-train-oria-electra.pkl')
    df_dev_set  = pd.read_pickle(args.data_path + 'qa-fool-me-twice-dev-oria-electra.pkl')
    df_test_set = pd.read_pickle(args.data_path + 'qa-fool-me-twice-test-oria-electra.pkl')

    df_train_set = df_train_set.sample(frac=1, random_state=1)

#     df_train_set = df_train_set[-50:]
#     df_dev_set = df_dev_set[-50:]
#     df_test_set = df_test_set[-50:]


    train_set = MyDataset(df_train_set, args)

    dev_set = MyDataset(df_dev_set, args)

    test_set = MyDataset(df_test_set, args)


    train_loader = DataLoader(dataset=train_set, batch_size=args.bsz, shuffle=False, num_workers = args.num_workers)
    val_loader = DataLoader(dataset=dev_set, batch_size=args.bsz, num_workers = args.num_workers, shuffle=False)


    model = BERT_Fusion(args)
    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    wandb.watch(model)


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                     lr= args.lr)


    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_loss = 100
    best_validate_dir = ''

    for epoch in range(args.epochs):

        model.train()


        optimizer.lr = args.lr

        start_time = time.time()
        cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []

        for i, (train_cq_input_ids, train_cq_attention_mask, train_aa_input_ids, train_aa_attention_mask, train_labels) in enumerate(train_loader):

            train_labels = to_var(train_labels)
            
            optimizer.zero_grad()

            train_ids_cq = to_var(train_cq_input_ids)
            train_mask_cq = to_var(train_cq_attention_mask)

            train_ids_aa = to_var(train_aa_input_ids)
            train_mask_aa = to_var(train_aa_attention_mask)


            output = model(train_ids_cq, train_mask_cq, train_ids_aa, train_mask_aa)
            loss = criterion(output, train_labels)
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(output, 1)

            cross_entropy = True

            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

            torch.cuda.empty_cache() 


        model.eval()
        validate_acc_vector_temp = []
        for i, (val_cq_input_ids, val_cq_attention_mask, val_aa_input_ids, val_aa_attention_mask, val_labels) in enumerate(val_loader):

            validate_labels = to_var(val_labels)

            validate_ids_cq = to_var(val_cq_input_ids)
            validate_mask_cq = to_var(val_cq_attention_mask)

            validate_ids_aa = to_var(val_aa_input_ids)
            validate_mask_aa = to_var(val_aa_attention_mask)

            with torch.no_grad():
                validate_output = model(validate_ids_cq, validate_mask_cq, validate_ids_aa, validate_mask_aa)
                _, validate_argmax = torch.max(validate_output, 1)
                vali_loss = criterion(validate_output, validate_labels)

            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.item())
            validate_acc_vector_temp.append(validate_accuracy.item())

            torch.cuda.empty_cache() 

            validate_acc = np.mean(validate_acc_vector_temp)
            valid_acc_vector.append(validate_acc)

        print ('Epoch [%d/%d],  Loss: %.4f, validate loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
                % (
                epoch + 1, args.epochs,  np.mean(cost_vector), np.mean(vali_cost_vector), np.mean(acc_vector),  validate_acc ))

        wandb.log({'epoch':epoch, "train_loss": np.mean(cost_vector), "val_loss":np.mean(vali_cost_vector), 'train_acc':np.mean(acc_vector), 'val_acc':validate_acc})


        if validate_acc > best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(args.save_model_path):
                os.mkdir(args.save_model_path)
            best_validate_dir = args.save_model_path + str(epoch + 1) + '-'+ args.run_name +'.pt'


        duration = time.time() - start_time

    torch.save(model.state_dict(), best_validate_dir)


    # test the model


    model = BERT_Fusion(args)
    model.load_state_dict(torch.load(best_validate_dir))
    #    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
    model.eval()


    test_loader = DataLoader(dataset=test_set, batch_size=args.bsz, num_workers = args.num_workers, shuffle=False)


    print('testing model')
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_cq_input_ids, test_cq_attention_mask, test_aa_input_ids, test_aa_attention_mask, test_labels) in enumerate(test_loader):
        test_labels = to_var(test_labels)

        test_ids_cq = to_var(test_cq_input_ids)
        test_mask_cq = to_var(test_cq_attention_mask)

        test_ids_aa = to_var(test_aa_input_ids)
        test_mask_aa = to_var(test_aa_attention_mask)

        with torch.no_grad():
            test_output = model(test_ids_cq, test_mask_cq, test_ids_aa, test_mask_aa)
            _, test_argmax = torch.max(test_output, 1)
        if i == 0:
            test_score = to_np(test_output.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_output.squeeze())), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

        torch.cuda.empty_cache()

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')

    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')

    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    wandb.log({'test_acc': test_accuracy, 'test_auc': test_aucroc})

    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred, digits=4)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))

    print('Saving results')


if __name__ == "__main__":
    main()