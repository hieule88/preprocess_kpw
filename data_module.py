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

    def __init__(
        self,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

    def setup(self, stage):
        
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

        self.dataset = datasets.load_dataset("uit-nlp/vietnamese_students_feedback")

        for split in self.dataset.keys():

            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                remove_columns=['topic', 'sentiment','sentence'],
            )

            self.columns = [
                c
                for c in self.dataset[split].column_names
                if c in self.loader_columns
            ]
            
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

        # Tokenize the text/text pairs
        features = {}
        # features['input_ids'] = np.array(self.embed(data= texts_or_text_pairs, max_seq_length= self.max_seq_length))
        
        line = example_batch['sentence']
        encode_line = self.tokenizer.encode(line)
        with torch.no_grad():
            encode_line = self.phobert(torch.tensor([encode_line]))
        encode_line = encode_line['pooler_output']

        features["input_ids"] = encode_line # shape = [1, 768]

        # Rename label to labels to make it easier to pass to model forward
        features["labels"] = torch.tensor(example_batch['sentiment'])

        return features

    @staticmethod
    def add_cache_arguments(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument("--root_path_other", default='', type=str)
        return parser


if __name__ == '__main__':
    datamodule = DataModule()
    datamodule.setup('fit')
    print(datamodule.train_dataloader())