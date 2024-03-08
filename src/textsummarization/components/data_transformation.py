import os
from textsummarization.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textsummarization.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config= DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_example_to_feature(self,example_batch):
        input_encoding = self.tokenizer(example_batch['dialogue'], max_length = 1024, truncation = True)

        with self.tokenizer.as_target_tokenizer():
            target_encoding = self.tokenizer(example_batch['summary'], max_length = 1024, truncation = True)

        return {
            'input_ids' : input_encoding['input_ids'],
            'attention_mask' : input_encoding['attention_mask'],
            'labels' : target_encoding['input_ids']
        }

    def convert(self):
        data_sumsum = load_from_disk(self.config.data_path) 
        data_sumsum_pt = data_sumsum.map(self.convert_example_to_feature, batched = True)
        data_sumsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset"))  

