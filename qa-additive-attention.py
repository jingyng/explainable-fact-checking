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
        
        qs_input = []
        a2s_input = []
        for i in range(args.num_questions):

            qs = list(np.array(df['qa'+str(i+1)].to_list())[:,0])
            acs = list(np.array(df['qa'+str(i+1)].to_list())[:,1])
            ars = list(np.array(df['qa'+str(i+1)].to_list())[:,2])
            apairs = list(zip(acs, ars))

            qs_input.append(self.tokenizer1.batch_encode_plus(
                batch_text_or_text_pairs=qs,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = args.max_c_len,  # maximum length of a sentence
                pad_to_max_length=True,  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            ))

            a2s_input.append(self.tokenizer1.batch_encode_plus(
                batch_text_or_text_pairs=apairs,  # the sentence to be encoded
                add_special_tokens=True,  # Add [CLS] and [SEP]
                max_length = args.max_a_len,  # maximum length of a sentence
                pad_to_max_length=True,  # Add [PAD]s
                return_attention_mask = True,  # Generate the attention mask
                return_tensors = 'pt',  # ask the function to return PyTorch tensors
            ))
        
        
        self.c = torch.stack((claim_input['input_ids'], claim_input['attention_mask']))
        self.y_data = torch.tensor(df['label'].tolist(), dtype = torch.long)
        
        self.qs_input_ids = qs_input[0]['input_ids'].unsqueeze(0)
        self.qs_masks = qs_input[0]['attention_mask'].unsqueeze(0)
        
        self.a2s_input_ids = a2s_input[0]['input_ids'].unsqueeze(0)
        self.a2s_masks = a2s_input[0]['attention_mask'].unsqueeze(0)

        for i in range(1, args.num_questions):
            self.qs_input_ids = torch.cat((self.qs_input_ids, qs_input[i]['input_ids'].unsqueeze(0)))
#             print(self.qs_input_ids.shape)
            self.qs_masks = torch.cat((self.qs_masks, qs_input[i]['attention_mask'].unsqueeze(0)))
            
            self.a2s_input_ids = torch.cat((self.a2s_input_ids, a2s_input[i]['input_ids'].unsqueeze(0)))
            self.a2s_masks = torch.cat((self.a2s_masks, a2s_input[i]['attention_mask'].unsqueeze(0)))
        
    def __getitem__(self, index):
        return self.c[:,index], self.qs_input_ids[:,index], \
                self.qs_masks[:,index], self.a2s_input_ids[:,index],    \
                self.a2s_masks[:,index], self.y_data[index]
    
    def __len__(self):
        return len(self.y_data)


# In[5]:


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


# In[6]:


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# In[7]:


class AdditiveAttention(nn.Module):
    """
     Applies a additive attention (bahdanau) mechanism on the output features from the decoder.
     Additive attention proposed in "Neural Machine Translation by Jointly Learning to Align and Translate" paper.
     Args:
         hidden_dim (int): dimesion of hidden state vector
     Inputs: query, value
         - **query** (batch_size, q_len, hidden_dim): tensor containing the output features from the decoder.
         - **value** (batch_size, v_len, hidden_dim): tensor containing features of the encoded input sequence.
     Returns: context, attn
         - **context**: tensor containing the context vector from attention mechanism.
         - **attn**: tensor containing the alignment from the encoder outputs.
     Reference:
         - **Neural Machine Translation by Jointly Learning to Align and Translate**: https://arxiv.org/abs/1409.0473
    """
    def __init__(self, hidden_dim: int) -> None:
        super(AdditiveAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
#         print('score:', score)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn


# In[9]:


class BERT_Fusion(nn.Module):
    def __init__(self, args):
        super(BERT_Fusion, self).__init__()
        self.question_num = args.num_questions
        self.bert1 = AutoModel.from_pretrained(args.bert1)
#         self.bert2 = AutoModel.from_pretrained(pretrain_model_path2)

#         self.attention = ScaledDotProductAttention(self.bert1.config.hidden_size)
        self.attention = AdditiveAttention(self.bert1.config.hidden_size)
        
        
        for param in self.bert1.base_model.parameters():
            param.requires_grad = True
            
#         for param in self.bert2.base_model.parameters():
#             param.requires_grad = True
            
#         for param in self.attention.parameters():
#             param.requires_grad = True
            
        self.classifier = nn.Sequential()
        self.classifier.add_module('c_fc1',  nn.Linear(self.bert1.config.hidden_size, 2))
#         self.classifier.add_module('c_softmax', nn.Softmax(dim=1))


    def forward(self, c, q_input_ids, q_masks, a2_input_ids, a2_masks):
        
        outputs_c = self.bert1(c[:,0], attention_mask = c[:,1])[1]
        
        outputs_q = self.bert1(q_input_ids[:,0], attention_mask = q_masks[:,0])[1]
#         print(outputs_q.shape)
        outputs_q = outputs_q.unsqueeze(1)
#         print(outputs_q.shape)
        outputs_a2 = self.bert1(a2_input_ids[:,0], attention_mask = a2_masks[:,0])[1].unsqueeze(1)
        
        for i in range(1, self.question_num):
            outputs_q = torch.cat((outputs_q, self.bert1(q_input_ids[:,i], attention_mask = q_masks[:,i])[1].unsqueeze(1)), dim=1)
#             print(outputs_q.shape)
            outputs_a2 = torch.cat((outputs_a2, self.bert1(a2_input_ids[:,i], attention_mask = a2_masks[:,i])[1].unsqueeze(1)), dim=1)
        
#         outputs_ar = self.bert2(ar[:,0], attention_mask = ar[:,1])
#         print(len(outputs_cq))
#         print((outputs_aa[0].shape))
#         print((outputs_aa[1].shape))

#         feature_news = max_pooling(outputs_news, mask_news)
#         feature_tweets = max_pooling(outputs_tweets, mask_tweets)
#         feature_c = outputs_c[1]
#         feature_q = outputs_q[1]

#         feature_c = outputs_c.last_hidden_state[:,0]
#         feature_q = outputs_q.last_hidden_state[:,0]

#         feature_ac = mean_pooling(outputs_ac, ac[:,1])
#         feature_ar = mean_pooling(outputs_ar, ar[:,1])

#         feature_a2 = outputs_a2[1]
#         feature_ar = outputs_ar[1]

#         print(feature_cq.shape)
#         print(feature_a2.shape)
    
        outputs_c = torch.unsqueeze(outputs_c, 1)
#         feature_q = torch.unsqueeze(feature_q, 1)
#         feature_a2 = torch.unsqueeze(feature_a2, 1)        

        
#         cqaa = torch.cat((feature_c, feature_q, feature_ac, feature_ar), 1)
        
#         print(cqaa.shape)
#         print('a2:',outputs_a2.shape)
#         print('c:',outputs_c.shape)    
        
        output_att, attn = self.attention(outputs_c, outputs_q, outputs_a2)
          
#         print('output_att:',output_att.shape)
        
        ### Class
        output = self.classifier(output_att)
        output = torch.squeeze(output, 1)
#         print(output.shape)
        ## Domain
#         reverse_feature = ReverseLayerF.apply(news_tweets, lambd)
#         domain_output = self.domain_classifier(reverse_feature)
     
        return output, attn


    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default=None, type=str, required=True, 
                       help="run name for the experiment")
    parser.add_argument("--bert1", default=None, type=str, required=True, 
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
    
    # pretrain_model_path = 'deepset/sentence_bert'
#     pretrain_model_path1 = 'sentence-transformers/LaBSE'
#     pretrain_model_path2 = 'sentence-transformers/all-MiniLM-L6-v2'
    # pretrain_model_path = 'bert-base-multilingual-cased'
    # pretrain_model_path = 'allenai/longformer-base-4096'
    # pretrain_model_path = 'bert-base-multilingual-uncased'
#     pretrain_model_path1 = args.bert1
#     pretrain_model_path2 = args.bert2

#     run_name = 'nocat-c-q-aa-retrieved'

    wandb.init(name = args.run_name, project='fack-check-qa', entity='fakejing', \
              tags = ['1-bert-nocat', 'gold', 'final','electra'], config = args, save_code = True)

# pretrain_model_path1 = 'roberta-large'
# pretrain_model_path2 = 'roberta-base'
# pretrain_model_path1 = 'sentence-transformers/LaBSE'
# pretrain_model_path1 = 'sentence-transformers/multi-qa-mpnet-base-dot-v1'
# pretrain_model_path2 = 'sentence-transformers/all-MiniLM-L6-v2'
# pretrain_model_path2 = 'sentence-transformers/paraphrase-mpnet-base-v2'
# pretrain_model_path = 'bert-base-multilingual-cased'
# pretrain_model_path = 'allenai/longformer-base-4096'
# pretrain_model_path = 'bert-base-multilingual-uncased'
# max_claim_len = 32
# max_q_len = 32
# max_a_len = 32

    logger = logging.getLogger("wandb")
    logger.setLevel(logging.ERROR)

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

        for i, (train_c, train_qs_input_ids, train_qs_masks,  \
                train_a2s_input_ids, train_a2s_masks, train_labels) in enumerate(train_loader):

            train_labels = to_var(train_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            train_c = to_var(train_c)
            train_qs_input_ids = to_var(train_qs_input_ids)
            train_qs_masks = to_var(train_qs_masks)

            train_a2s_input_ids = to_var(train_a2s_input_ids)
            train_a2s_masks = to_var(train_a2s_masks)


            output, _ = model(train_c, train_qs_input_ids, train_qs_masks, train_a2s_input_ids, train_a2s_masks)
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
        for i, (val_c, val_qs_input_ids, val_qs_masks, \
                val_a2s_input_ids, val_a2s_masks, val_labels) in enumerate(val_loader):

            validate_labels = to_var(val_labels)

            validate_c = to_var(val_c)
            validate_qs_input_ids = to_var(val_qs_input_ids)
            validate_qs_masks = to_var(val_qs_masks)

            validate_a2s_input_ids = to_var(val_a2s_input_ids)
            validate_a2s_masks = to_var(val_a2s_masks)
    #         validate_ar = to_var(val_ar)

            with torch.no_grad():
                validate_output, _ = model(validate_c, validate_qs_input_ids, validate_qs_masks, \
                                        validate_a2s_input_ids, validate_a2s_masks)
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


    # In[ ]:


    print('testing model')
    test_score = []
    test_pred = []
    test_true = []
#     weights = []
    for i, (test_c, test_qs_input_ids, test_qs_masks, test_a2s_input_ids, test_a2s_masks, test_labels) in enumerate(test_loader):
        test_labels = to_var(test_labels)

        test_c = to_var(test_c)
        test_qs_input_ids = to_var(test_qs_input_ids)
        test_qs_masks = to_var(test_qs_masks)

        test_a2s_input_ids = to_var(test_a2s_input_ids)
        test_a2s_masks = to_var(test_a2s_masks)
    #     test_ar = to_var(test_ar)

        with torch.no_grad():
            test_output, attn = model(test_c, test_qs_input_ids, test_qs_masks, test_a2s_input_ids, test_a2s_masks)
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
    
#         weights+=list(to_np(attn))

#     artifact = wandb.Artifact('table', 'attention weights')
#     table = wandb.Table(columns=["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"], data=weights)
#     artifact.add(table, "table")
#     wandb.log_artifact(artifact)
    
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
