from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelWithLMHead
import os
import json
tokenizer = AutoTokenizer.from_pretrained("t5-base")

DATA_DIR = "/home/mjkim/nas/Projects/paper_title_generation/data"

class TitleDataset(Dataset):
    def __init__(self, split, enc_max_len=512, dec_max_len=50):
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        self.split = split
        self.enc_max_len = enc_max_len
        self.dec_max_len = dec_max_len
        
        # load dataset
        
        with open(os.path.join(DATA_DIR, self.split+".json"), "r" ) as f:
            self.data = json.load(f)     
        
           
    def __len__(self):
        return len(self.data)
        

    def __getitem__(self, index):
        title = self.data[index]["title"]
        abstract = self.data[index]["abstract"]
        
        tokenized_abstract = self.tokenizer(abstract, max_length=self.enc_max_len, padding="max_length", return_tensors="pt")
        tokenized_title = self.tokenizer(title, max_length=self.dec_max_len, padding="max_length", return_tensors="pt")
        model_inputs = {}
        model_inputs['input_ids'] = tokenized_abstract['input_ids'].squeeze(0)
        model_inputs['attention_mask'] = tokenized_abstract['attention_mask'].squeeze(
            0)
        model_inputs['labels'] = tokenized_title['input_ids'].squeeze(0)
        
        return model_inputs
    
    
