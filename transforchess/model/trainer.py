from time import time as get_time
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoConfig,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

from . import config
from .metrics import GameLength


def pretrain(resume=False):
    dataset = load_from_disk(config.DATASET_PRETRAIN)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT, revision='dev')

    model_config = AutoConfig.from_pretrained(
        config.BASE_CHECKPOINT,
        vocab_size=tokenizer.vocab_size,
    )
    model = AutoModelForMaskedLM.from_config(model_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    args = TrainingArguments(
        output_dir=config.RESULTS_PRETRAIN,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=10,
        learning_rate=1e-4,
        weight_decay=0.01,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    try:
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        trainer.save_model(config.MODEL_PRETRAIN_PATH)
        print('Saved model')


class Seq2SeqTrainerWithGameLengthMetric(Seq2SeqTrainer):
    def __init__(self, *args, game_length_metric='median', game_length_metric_n=30, **kwargs):
        super().__init__(*args, **kwargs)
        self.game_length_metric = GameLength(game_length_metric, game_length_metric_n)

    def evaluate(self, *args, **kwargs):
        print('Running evaluation...')
        start = get_time()
        metrics =  {
            'eval_game_length': self.game_length_metric(self.model, self.tokenizer, self.args.device),
        }
        runtime = get_time() - start
        metrics['eval_runtime'] = runtime
        print(metrics)
        return metrics


def train(resume=False):
    dataset = load_from_disk(config.DATASET_MAIN)

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_CHECKPOINT, revision='dev')

    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_CHECKPOINT, revision='dev')

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=config.RESULTS,
        optim='adamw_torch',
        fp16=True,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        
        save_strategy='steps',
        save_steps=100,
        save_total_limit=5,
        logging_strategy='steps',
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=100,
        
        gradient_accumulation_steps=16,
        learning_rate=9.117e-5,
        weight_decay=0.00037,
    )

    trainer = Seq2SeqTrainerWithGameLengthMetric(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    try:
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        print('Stopping...')
    finally:
        trainer.save_model(config.MODEL_PATH)
        print('Saved model')
