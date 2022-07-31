from transformers import AutoTokenizer

from transforchess.data import load_dataset
from . import BASE_CHECKPOINT
from .paths import TOKENIZED_DATASET


def tokenize_dataset():
    dataset = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)

    def tokenize(element):
        if element['label'] == '1-0':
            element['label'] = 0
        elif element['label'] == '0-1':
            element['label'] = 1
        else:
            element['label'] = 2
        element = tokenizer(element['text'], truncation=True)
        return element

    tokenized_dataset = dataset.map(
        tokenize,
        batched=False,
        num_proc=24,
        remove_columns=['text'],
    )

    tokenized_dataset.save_to_disk(TOKENIZED_DATASET)
