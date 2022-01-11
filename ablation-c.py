#!/usr/bin/env python
# coding: utf-8

# In[1]:

import wandb

# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import argparse
import time, os
import copy
import pickle
import random
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
from typing import Optional, Tuple

import torchvision.datasets as dsets
import torchvision.transforms as transforms

import logging

from sklearn import metrics
from sklearn.preprocessing import label_binarize
import scipy.io as sio
from tqdm import tqdm
import pandas as pd


from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead

# seed = 42
# os.environ['PYTHONHASHSEED'] = str(seed)
# # Torch RNG
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# # Python RNG
# np.random.seed(seed)
# random.seed(seed)

class MyDataset(Dataset):
    def __init__(self, df, args):

        self.tokenizer1 = AutoTokenizer.from_pretrained(args.bert1)
#         self.tokenizer2 = AutoTokenizer.from_pretrained(pretrain_model_path2)
        

#         a2 = list(zip(df.answer_text, df.answer_retrieved))
        
        claim_input = self.tokenizer1.batch_encode_plus(
            batch_text_or_text_pairs=df.text.to_list(),  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length = args.max_c_len,  # maximum length of a sentence
            pad_to_max_length=True,  # Add [PAD]s
            return_attention_mask = True,  # Generate the attention mask
            return_tensors = 'pt',  # ask the function to return PyTorch tensors
        )
        
        self.c = torch.stack((claim_input['input_ids'], claim_input['attention_mask']))
        self.y_data = torch.tensor(df['label'].tolist(), dtype = torch.long)

        
    def __getitem__(self, index):
        return self.c[:,index], self.y_data[index]
    
    def __len__(self):
        return len(self.y_data)


# In[5]:


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
        self.classifier.add_module('c_fc1',  nn.Linear(self.bert1.config.hidden_size, 2))


    def forward(self, c):
        
        outputs_c = self.bert1(c[:,0], attention_mask = c[:,1])[1]
        
        outputs_c = torch.unsqueeze(outputs_c, 1) 

        output = self.classifier(outputs_c)
        output = torch.squeeze(output, 1)
     
        return output


    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=None, type=str, required=True, 
                       help="run name for the experiment")
    parser.add_argument("--bert1", default='microsoft/mpnet-base', type=str, required=False, 
                       help="bert model name for claim and questions")
#     parser.add_argument("--bert2", default=None, type=str, required=True, 
#                        help="bert model name for answers/answer pairs")
    parser.add_argument("--lr", default= 2e-5, type=float, required=False, 
                       help="learning rate")
    parser.add_argument("--bsz", default= 32, type=int, required=False, 
                       help="batch size")
    parser.add_argument("--data_path", default= './', type=str, required=False, 
                       help="input data path")
    parser.add_argument("--max_c_len", default= 32, type=int, required=False, 
                       help="maximum token length for claim and questions")
    parser.add_argument("--max_a_len", default= 32, type=int, required=False, 
                       help="maximum token length for answers")
    parser.add_argument("--num_workers", default= 4, type=int, required=False, 
                       help="number of workers")
    parser.add_argument("--save_model_path", default= './model/', type=str, required=False, 
                       help="path to save model")
    parser.add_argument("--epochs", default= 5, type=int, required=False, 
                       help="number of epochs")
    parser.add_argument("--num_questions", default= 10, type=int, required=False, 
                       help="number of questions")
    
    args = parser.parse_args()
    

    wandb.init(name = args.run_name, project='fack-check-qa', entity='fakejing', \
              tags = ['final','claim-only'], config = args, save_code = True)


    df_train_set = pd.read_pickle(args.data_path + 'qa-fool-me-twice-train-nocat-cqaa-electra.pkl')
    df_dev_set  = pd.read_pickle(args.data_path + 'qa-fool-me-twice-dev-nocat-cqaa-electra.pkl')
    df_test_set = pd.read_pickle(args.data_path + 'qa-fool-me-twice-test-nocat-cqaa-electra.pkl')


    df_train_set = df_train_set.sample(frac=1, random_state=1)


    # In[12]:


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
    #     model = torch.nn.DataParallel(model)
        model.cuda()

    wandb.watch(model)

    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                     lr= args.lr)

    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_loss = 100
    best_validate_dir = ''
#     output_file = 'output/'

    for epoch in range(args.epochs):

        model.train()

        optimizer.lr = args.lr
        #rgs.lambd = lambd

        start_time = time.time()
        cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []

        for i, (train_c, train_labels) in enumerate(train_loader):

            train_labels = to_var(train_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            train_c = to_var(train_c)

            output = model(train_c)
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
        for i, (val_c, val_labels) in enumerate(val_loader):

            validate_labels = to_var(val_labels)

            validate_c = to_var(val_c)

            with torch.no_grad():
                validate_output = model(validate_c)
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


    model = BERT_Fusion(args)
    model.load_state_dict(torch.load(best_validate_dir))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()


    test_loader = DataLoader(dataset=test_set, batch_size=args.bsz, num_workers = args.num_workers, shuffle=False)


    print('testing model')
    test_score = []
    test_pred = []
    test_true = []
    for i, (test_c, test_labels) in enumerate(test_loader):
        test_labels = to_var(test_labels)

        test_c = to_var(test_c)
       
        with torch.no_grad():
            test_output= model(test_c)
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
