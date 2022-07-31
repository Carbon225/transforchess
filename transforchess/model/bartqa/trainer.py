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


def pretrain(resume=False):
    dataset = load_from_disk(config.DATASET_PRETRAIN)

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)

    model_config = AutoConfig.from_pretrained(
        config.BASE_CHECKPOINT,
        vocab_size=tokenizer.vocab_size,
    )
    model = AutoModelForMaskedLM.from_config(model_config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    args = TrainingArguments(
        output_dir=config.RESULTS_PRETRAIN,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
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
        trainer.save_model(config.MODEL_PRETRAIN)
        print('Saved model')


def train(resume=False):
    dataset = load_from_disk(config.DATASET_QA)

    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)

    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_PRETRAIN)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=config.RESULTS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
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
        trainer.save_model(config.MODEL)
        print('Saved model')
