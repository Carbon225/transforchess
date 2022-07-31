from datasets import load_from_disk
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from . import BASE_CHECKPOINT
from . import paths


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    good = 0
    for i in range(len(predictions)):
        if predictions[i] == labels[i]:
            good += 1
    return {
        'acc': good / len(predictions),
    }


def train(resume=False):
    dataset = load_from_disk(paths.TOKENIZED_DATASET)

    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_CHECKPOINT)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=paths.RESULTS,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=1,
        save_steps=100,
        logging_steps=10,
        learning_rate=1e-5,
        fp16=True,
        report_to='tensorboard',
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
        trainer.save_model(paths.MODEL)
        print('Saved model')
