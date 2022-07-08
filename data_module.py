from argparse import ArgumentParser
from sklearn import preprocessing
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from datasets import Dataset
import datasets
import pickle
from transformers import AutoModel, AutoTokenizer
from vncorenlp import VnCoreNLP

import warnings
warnings.filterwarnings("ignore")

class DataModule(pl.LightningDataModule):

    loader_columns = [
        "input_ids",
        "labels",
    ]

    num_labels = 15

    max_seq_length = 50

    def __init__(
        self,
        path_to_vncore: str,
        max_seq_length: int = 50,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.root_path_other = './dataset'
        preprocessor = Preprocessor(train_path= self.root_path_other + '/train_update_10t01.pkl',\
                            mode= 'init',\
                            val_path= self.root_path_other + '/dev_update_10t01.pkl',\
                            test_path= self.root_path_other + '/test_update_10t01.pkl',)
        self.embed = preprocessor.batch_to_matrix

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.text_fields = 'sentences'
        self.num_labels = 15
        
        # self.dataset = datasets.load_dataset(*self.dataset_names[self.task_name])
        # self.max_seq_length = self.tokenizer.model_max_length

    def setup(self, stage):

        self.dataset = datasets.load_dataset("uit-nlp/vietnamese_students_feedback")

        for split in self.dataset.keys():

            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=['topic', 'sentiment','sentence'],
            )

            self.columns = [
                c
                for c in self.dataset[split].column_names
                if c in self.loader_columns
            ]
            
        self.dataset.set_format(type="torch", columns=self.columns)

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.train_batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["validation"],
            batch_size=self.eval_batch_size,
        )


    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.eval_batch_size,
        )

    @property
    def train_dataset(self):
        return self.dataset["train"]

    @property
    def val_dataset(self):
        return self.dataset["validation"]

    @property
    def metric(self):
        return f1_score

    def convert_to_features(self, example_batch, indices=None):
        
        phobert = AutoModel.from_pretrained("vinai/phobert-base")
        tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        # Tokenize the text/text pairs
        features = {}
        # features['input_ids'] = np.array(self.embed(data= texts_or_text_pairs, max_seq_length= self.max_seq_length))
        
        encode_lines = []
        for line in example_batch['sentence']:
            # encode_line = torch.tensor([tokenizer.encode(line)])
            encode_lines.append(tokenizer.encode(line))
        with torch.no_grad():
            encode_lines = phobert(torch.tensor(encode_lines))

        features["input_ids"] = encode_lines

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = torch.tensor(example_batch['labels'])

        return features

    @staticmethod
    def add_cache_arguments(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--root_path_other", default='', type=str)
        return parser

if __name__ == '__main__':
    rdrsegmenter = VnCoreNLP(r"C:/Users/ThinkPro/OneDrive/Máy tính/khaiphaweb/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m') 

    dt = datasets.load_dataset("uit-nlp/vietnamese_students_feedback")
    line = dt['train']['sentence'][0]
    line = rdrsegmenter.tokenize(line) 
    print(line)
    print(dt['train']['sentiment'][0])
    print(dt['train']['topic'][0])
    phobert = AutoModel.from_pretrained("vinai/phobert-base")
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    input_ids = torch.tensor([tokenizer.encode(line)])
    print(input_ids)
    encode_line = torch.tensor(phobert(input_ids))
    print(encode_line.size())