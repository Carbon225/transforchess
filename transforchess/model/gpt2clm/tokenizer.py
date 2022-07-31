from transformers import AutoTokenizer

from transforchess.data import load_dataset
from . import BASE_CHECKPOINT
from .paths import TOKENIZED_DATASET


def tokenize_dataset():
    dataset = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)

    def tokenize(element):
        return tokenizer(element['text'], truncation=True)

    tokenized_dataset = dataset.map(
        tokenize,
        batched=False,
        num_proc=24,
        remove_columns=dataset.column_names,
    )

    tokenized_dataset.save_to_disk(TOKENIZED_DATASET)
