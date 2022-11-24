import nltk
import numpy as np
#import tdqm
import argparse
import torch
# import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from datasets import load_metric

from tqdm import tqdm
import evaluate
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import sys
sys.path.append("data")
from ..data.dataset import TitleDataset
import os
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="/home/mjkim/nas/Projects/paper_title_generation/checkpoint2")

args = parser.parse_args()

device = "cuda:0"
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
finetuned_model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_path)
finetuned_model.eval()
finetuned_model.to(device)
test_dataset = TitleDataset("test")
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
rouge = evaluate.load('rouge')
collection = []


with torch.no_grad():
    for b in tqdm(test_dataloader):


        device = "cuda:0"
        cur_example = b
        # print(cur_example)
        input_ids = cur_example["input_ids"].to(device)
        attention_mask = cur_example["attention_mask"].to(device)
        response = finetuned_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=100,
            num_beams=5
        ).cpu()
        
        
        history =  tokenizer.batch_decode(cur_example["input_ids"], skip_special_tokens=True)
        label = tokenizer.batch_decode(cur_example["labels"], skip_special_tokens=True)
        pred = tokenizer.batch_decode(response, skip_special_tokens=True)
        score = rouge.compute(references=label, predictions=pred, use_aggregator=False)["rouge1"]
        for i in range(len(history)):
            s_buffer=""
            s_buffer += str("="*100+"\n")
            s_buffer += history[i]+"\n"
            s_buffer += str("Label >>> "+"\n"+label[i]+"\n")
            s_buffer += str("Model >>> "+"\n"+pred[i]+"\n")
            
            
            
            collection.append([s_buffer, score[i][1]])
    
collection = sorted(collection, key=lambda x: x[1])

with open(os.path.join(args.checkpoint_path, "result.txt"), "w") as f:
    for c in collection:
        f.write(c[0])
                