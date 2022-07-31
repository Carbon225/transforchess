from random import choice
from transformers import AutoTokenizer

from transforchess.data import load_dataset
from . import BASE_CHECKPOINT
from .paths import TOKENIZED_DATASET


def tokenize_dataset():
    dataset = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)

    def filter_won(element) -> bool:
        return element['label'] in ('1-0', '0-1')

    def filter_long(element) -> bool:
        return len(element['text'].split('.')) > 10

    def make_qa(element):
        game = element['text']
        winner = element['label']

        moves = [move.strip() for move in game.split('.')[:-1]]

        if winner == '1-0':
            answer_index = choice(range(0, len(moves) - 1, 2))    
        else:
            answer_index = choice(range(1, len(moves) - 1, 2))
        
        element['question'] = ' ' + '. '.join(moves[:answer_index]) + '.'
        element['answer'] = ' ' + moves[answer_index] + '.'

        return element

    def tokenize(element):
        inputs = tokenizer(element['question'], truncation=False)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(element['answer'], truncation=False)
        inputs['labels'] = labels['input_ids']
        return inputs

    def filter_short(element) -> bool:
        return len(element['input_ids']) <= tokenizer.model_max_length and \
               len(element['labels']) <= tokenizer.model_max_length

    dataset = dataset.filter(
        filter_won,
        batched=False,
        num_proc=24,
    )

    dataset = dataset.filter(
        filter_long,
        batched=False,
        num_proc=24,
    )

    dataset = dataset.map(
        make_qa,
        batched=False,
        num_proc=24,
        remove_columns=['text', 'label'],
    )

    dataset = dataset.map(
        tokenize,
        batched=True,
        num_proc=24,
        remove_columns=['question', 'answer'],
    )

    dataset = dataset.filter(
        filter_short,
        batched=False,
        num_proc=24,
    )

    dataset.save_to_disk(TOKENIZED_DATASET)
