import argparse
import os
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, logging, set_seed
import torch
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

import warnings
warnings.filterwarnings("ignore")




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--train_file", type=str, default="train_set.csv")
    parser.add_argument("--valid_file", type=str, default="validation_set.csv")
    parser.add_argument("--test_file", type=str, default="test_set.csv")
    
    parser.add_argument("--seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eos_token_id", type=int, default=49152)
    
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.05)

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_freq", default=1, type=int)
    parser.add_argument("--eval_freq", default=1000, type=int)
    parser.add_argument("--save_freq", default=1000, type=int)

    return parser.parse_args()


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    text = f"Question: {example['context']}\n\nAnswer: {example['response']}"
    return text


def create_datasets(tokenizer, args):
    train_files = {"train": [args.train_file]}
    valid_files = {"validation": [args.valid_file]}
    test_files = {"test": [args.test_file]}

    train_data = load_dataset("csv", data_files=train_files, split="train")  
    valid_data = load_dataset("csv", data_files=valid_files, split="validation")
    test_data = load_dataset("csv", data_files=test_files, split="test")

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}. Size of the test set: {len(test_data)}")
    chars_per_token = chars_token_ratio(train_data, tokenizer, len(train_data))
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data, tokenizer):
    print("Loading the model")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    train_data.start_iteration = 0

    print("Starting main loop")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        eval_strategy="epoch",
        num_train_epochs=3,
        do_train=True,
        do_eval=True,
        save_steps=args.save_freq,
        logging_steps=args.log_freq,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        run_name="model-finetuned",
        ddp_find_unused_parameters=False,
        report_to="none",
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    os.environ["WANDB_DISABLED"] = "True"
    accelerator = Accelerator()

    access_token = ""


    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, load_in_4bit=True, device_map="auto", token=access_token
    )

    model.resize_token_embeddings(len(tokenizer))
    # Configure the pad token in the model
    model.config.pad_token_id = tokenizer.pad_token_id

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=lora_config,
        max_seq_length=1024,
        packing=True,
    )

    print_trainable_parameters(trainer.model)

    print("Training...")
    trainer.train()

    print("Saving last checkpoint of the model")
    trainer.model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
    trainer.save_model(os.path.join(args.output_dir, "last_checkpoint/"))
    trainer.save_model(os.path.join(args.output_dir, "best_model/"))
    best_ckpt_path = trainer.state.best_model_checkpoint
    print('Best Model Checkpoint Path: ', best_ckpt_path)


def main(args):
    access_token = ""
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, token=access_token)
    special_tokens_dict = {'additional_special_tokens': ['[EOC]', '[SOC]', '[Traveler]', '[Agent]', '[SOR]', '[EOR]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset, tokenizer)


if __name__ == "__main__":
    args = get_args()
    assert args.model_path != "", "Please provide the model path"

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()

    main(args)
