from random import choice
from transformers import AutoTokenizer

from transforchess.data import load_dataset
from . import config


def train_tokenizer():
    dataset = load_dataset()

    def prepend_space(element):
        element['text'] = ' ' + element['text']
        return element

    dataset = dataset.map(
        prepend_space,
        batched=False,
        num_proc=24,
    )

    def get_corpus():
        for start_idx in range(0, len(dataset), 1024):
            samples = dataset[start_idx : start_idx + 1024]
            yield samples['text']

    tokenizer = AutoTokenizer.from_pretrained(config.BASE_CHECKPOINT)
    
    tokenizer = tokenizer.train_new_from_iterator(get_corpus(), tokenizer.vocab_size)

    print(f'Vocab size: {tokenizer.vocab_size}')

    tokenizer.save_pretrained(config.TOKENIZER)


def tokenize_dataset():
    dataset = load_dataset()
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)

    # common pipeline

    def filter_long(element) -> bool:
        return len(element['text'].split('.')) > 10

    # QA pipeline

    def filter_won(element) -> bool:
        return element['label'] in ('1-0', '0-1')

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

        if element['question'] == '.':
            element['question'] = ''

        return element

    def tokenize_qa(element):
        inputs = tokenizer(element['question'], truncation=False)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(element['answer'], truncation=False)
        inputs['labels'] = labels['input_ids']
        return inputs

    def filter_short(element) -> bool:
        return len(element['input_ids']) <= tokenizer.model_max_length and \
               len(element['labels']) <= tokenizer.model_max_length

    # pretrain pipeline

    def prepend_space(element):
        element['text'] = ' ' + element['text']
        return element

    def tokenize_pretrain(element):
        return tokenizer(element['text'], truncation=True)

    # common pipeline

    dataset = dataset.filter(
        filter_long,
        batched=False,
        num_proc=24,
    )

    # QA pipeline

    dataset_qa = dataset.filter(
        filter_won,
        batched=False,
        num_proc=24,
    )

    dataset_qa = dataset_qa.map(
        make_qa,
        batched=False,
        num_proc=24,
        remove_columns=['text', 'label'],
    )

    dataset_qa = dataset_qa.map(
        tokenize_qa,
        batched=True,
        num_proc=24,
        remove_columns=['question', 'answer'],
    )

    dataset_qa = dataset_qa.filter(
        filter_short,
        batched=False,
        num_proc=24,
    )

    # pretrain pipeline

    dataset_pretrain = dataset.map(
        prepend_space,
        batched=False,
        num_proc=24,
    )

    dataset_pretrain = dataset_pretrain.map(
        tokenize_pretrain,
        batched=True,
        num_proc=24,
        remove_columns=['text', 'label'],
    )

    dataset_qa.save_to_disk(config.DATASET_QA)
    dataset_pretrain.save_to_disk(config.DATASET_PRETRAIN)
