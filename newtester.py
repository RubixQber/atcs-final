import datasets
import json
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.utils.data as data
import util
import transformers
import tensorflow

from args import get_train_args
from collections import OrderedDict
from datasets import load_metric
from datasets import load_dataset
from datasets import Dataset
from json import dumps
from models import BiDAF
from tensorboardX import SummaryWriter
from transformers import AdamW
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
from transformers import TrainingArguments
from transformers import Trainer
from tqdm import tqdm
from torch.utils.data import DataLoader

from ujson import load as json_load
from util import collate_fn, SQuAD

class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)

def add_token_positions(encodings, answers):
    # initialize lists to contain the token indices of answer start/end
    start_positions = []
    end_positions = []
    for i in range(len(answers)):
        # append start/end token position using char_to_token method
        start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
        end_positions.append(encodings.char_to_token(i, answers[i]['answer_end']))

        # if start position is None, the answer passage has been truncated
        if start_positions[-1] is None:
            start_positions[-1] = tokenizer.model_max_length
        # end position cannot be found, char_to_token found space, so shift position until found
        shift = 1
        while end_positions[-1] is None:
            end_positions[-1] = encodings.char_to_token(i, answers[i]['answer_end'] - shift)
            shift += 1
    # update our encodings object with the new token-based start/end positions
    encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

def read_squad(path):
    with open(path, 'rb') as f:
            squad_dict = json.load(f)

    # initialize lists for contexts, questions, and answers
    contexts = []
    questions = []
    answers = []
    # iterate through all data in squad data
    for group in squad_dict['data']:
        for passage in group['paragraphs']:
            context = passage['context']
            for qa in passage['qas']:
                question = qa['question']
                # check if we need to be extracting from 'answers' or 'plausible_answers'
                if 'plausible_answers' in qa.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in qa[access]:
                    # append data to lists
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
#    return formatted data lists
    return contexts, questions, answers

def add_end_idx(answers, contexts):
    # loop through each answer-context pair
    for answer, context in zip(answers, contexts):
        # gold_text refers to the answer we are expecting to find in context
        gold_text = answer['text']
        # we already know the start index
        start_idx = answer['answer_start']
        # and ideally this would be the end index...
        end_idx = start_idx + len(gold_text)

        # ...however, sometimes squad answers are off by a character or two
        if context[start_idx:end_idx] == gold_text:
            # if the answer is not off :)
            answer['answer_end'] = end_idx
        else:
            # this means the answer is off by 1-2 tokens
            for n in [1, 2]:
                if context[start_idx-n:end_idx-n] == gold_text:
                    answer['answer_start'] = start_idx - n
                    answer['answer_end'] = end_idx - n


model_path = './custom'
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# move model over to detected device
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
val_contexts, val_questions, val_answers = read_squad('./data/dev-v2.0.json')
add_end_idx(val_answers, val_contexts)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
add_token_positions(val_encodings, val_answers)
val_dataset = SquadDataset(val_encodings)
# switch model out of training mode
model.eval()
# initialize validation set data loader
val_loader = DataLoader(val_dataset, batch_size=16)
# initialize list to store accuracies
acc = []
# loop through batches
print("starting batches")
loop = tqdm(val_loader, leave=True)
for batch in loop:
    # we don't need to calculate gradients as we're not training
    with torch.no_grad():
        # pull batched items from loader
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # we will use true positions for accuracy calc
        start_true = batch['start_positions'].to(device)
        end_true = batch['end_positions'].to(device)
        # make predictions
        outputs = model(input_ids, attention_mask=attention_mask)
        # pull prediction tensors out and argmax to get predicted tokens
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        # calculate accuracy for both and append to accuracy list
        acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
        acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
# calculate average accuracy in total
acc = sum(acc)/len(acc)
print("fine-tuned acc = " + str(acc))

model2 = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
device2 = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# move model over to detected device
model2.to(device2)
tokenizer2 = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
val_contexts, val_questions, val_answers = read_squad('./data/dev-v2.0.json')
add_end_idx(val_answers, val_contexts)
val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)
add_token_positions(val_encodings, val_answers)
val_dataset = SquadDataset(val_encodings)
# switch model out of training mode
model2.eval()
# initialize validation set data loader
val_loader = DataLoader(val_dataset, batch_size=16)
# initialize list to store accuracies
acc = []
# loop through batches
print("starting batches")
loop = tqdm(val_loader, leave=True)
for batch in loop:
    # we don't need to calculate gradients as we're not training
    with torch.no_grad():
        # pull batched items from loader
        input_ids = batch['input_ids'].to(device2)
        attention_mask = batch['attention_mask'].to(device2)
        # we will use true positions for accuracy calc
        start_true = batch['start_positions'].to(device2)
        end_true = batch['end_positions'].to(device2)
        # make predictions
        outputs = model2(input_ids, attention_mask=attention_mask)
        # pull prediction tensors out and argmax to get predicted tokens
        start_pred = torch.argmax(outputs['start_logits'], dim=1)
        end_pred = torch.argmax(outputs['end_logits'], dim=1)
        # calculate accuracy for both and append to accuracy list
        acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
        acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
# calculate average accuracy in total
acc = sum(acc)/len(acc)
print("original acc = " + str(acc))
