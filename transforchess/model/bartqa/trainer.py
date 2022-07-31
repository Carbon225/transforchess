from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)

from . import BASE_CHECKPOINT
from . import paths


def train(resume=False):
    dataset = load_from_disk(paths.TOKENIZED_DATASET)

    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)

    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_CHECKPOINT)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        output_dir=paths.RESULTS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        report_to='tensorboard',
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
    finally:
        trainer.save_model(paths.MODEL)
        print('Saved model')
