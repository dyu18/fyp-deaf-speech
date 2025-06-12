"""
This code implements the fine-tuning and evaluation of Whisper on the Corpus of Deaf Speech. 
Code partially adapted from https://huggingface.co/blog/fine-tune-whisper.
"""

import time
start_time = time.perf_counter()        # timing the process

import wandb
import pandas as pd
import torch
import os
import gc
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import evaluate
import jiwer
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from transformers import WhisperForConditionalGeneration
from datasets import load_from_disk
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl, EarlyStoppingCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
# from transformers.utils import logging
# logging.set_verbosity_info()

import tracemalloc
tracemalloc.start()         # measuring memory usage

# --------------------------------------------------------------------------------

# os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_SILENT"] = "true"

DATA_PATH = "CorpusOfDeafSpeech" 
## TODO: choose intelligibility group
intelligibility = "medium" # "low" # "high"
## TODO: specify dataset path
dataset_dir = f"HI_dataset_trainvaltest_{intelligibility}_LPO_chunked"
## TODO: define trainer output path
trainer_output_path = f"trainer_output_{intelligibility}_with_callbacks_LPO_steps_chunked"
## TODO: define WandB project name
# wandb_proj_name = f"{intelligibility}_LPO_steps_chunked_val"

if torch.cuda.is_available():
    device = "cuda"
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
    allocated_mem = torch.cuda.memory_allocated(0) / 1024**3  # Convert to GB
    reserved_mem = torch.cuda.memory_reserved(0) / (1024**3)  # Convert to GB
    free_mem = total_mem - allocated_mem
    print(f"GPU 0: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {total_mem:.1f}GB")
    print(f"Allocated Memory: {allocated_mem:.1f}GB")
    print(f"Reserved Memory: {reserved_mem:.1f}GB")
    print(f"Free Memory: {free_mem:.1f}GB")
else:
    device = "cpu"
print(device)

model_name = "openai/whisper-medium.en"
language = "en"
task = "transcribe"
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)

# --------------------------------------------------------------------------------


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# ---------------------------------- initialisations --------------------------------------------

## load dataset from saved location
dataset_path = os.path.join(DATA_PATH, dataset_dir)
HI_dataset_trainvaltest = load_from_disk(dataset_path)

## init model
model = WhisperForConditionalGeneration.from_pretrained(model_name,
                                                        load_in_8bit=True,      # True if on gpu
                                                        device_map="auto"       # "auto" if on gpu
                                                        )

## init data collator
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)


## ----------------- evaluate pre-trained model to get baseline performance ------------------------
eval_dataloader = DataLoader(HI_dataset_trainvaltest["test"], batch_size=32, collate_fn=data_collator)
forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
normalizer = BasicTextNormalizer()

def get_wer_breakdown(predictions, references):
    ## get detailed error breakdown
    error_details = jiwer.compute_measures(references, predictions)

    ## extract individual error types
    num_substitutions = error_details['substitutions']
    num_insertions = error_details['insertions']
    num_deletions = error_details['deletions']
    total_errors = num_substitutions + num_insertions + num_deletions

    ## compute percentage contribution of each error type
    substitution_pct = (num_substitutions / total_errors) if total_errors > 0 else 0
    insertion_pct = (num_insertions / total_errors) if total_errors > 0 else 0
    deletion_pct = (num_deletions / total_errors) if total_errors > 0 else 0

    ## store results in a dict
    wer_breakdown = {"wer": error_details["wer"],        # word error rate
                     "mer": error_details["mer"],        # match error rate
                     "wil": error_details["wil"],        # word information lost
                     "wip": error_details["wip"],        # word information preserved
                     "insertions": insertion_pct,
                     "substitutions": substitution_pct,
                     "deletions": deletion_pct}
        
    return wer_breakdown

def evaluate_model(model, 
                   eval_dataloader=eval_dataloader, 
                   forced_decoder_ids=forced_decoder_ids, 
                   processor=processor, 
                   normalizer=normalizer
                   ):
    predictions = []
    references = []
    normalized_predictions = []
    normalized_references = []

    model.config.use_cache = True
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
        with torch.amp.autocast('cuda'):
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        forced_decoder_ids=forced_decoder_ids,
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)
                decoded_preds = processor.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
                predictions.extend(decoded_preds)
                references.extend(decoded_labels)
                for pred, ref in zip(decoded_preds, decoded_labels):
                    norm_pred = normalizer(pred).strip()
                    norm_ref = normalizer(ref).strip()
                    # only evaluate the sample if neither of its prediction nor its reference is empty string
                    if len(norm_pred)>0 and len(norm_ref)>0:
                        normalized_predictions.append(norm_pred)
                        normalized_references.append(norm_ref)
            del generated_tokens, labels, batch
        gc.collect()
    
    wer_breakdown = get_wer_breakdown(predictions=normalized_predictions, references=normalized_references)
    return wer_breakdown


print("Baseline performance (in percentage):")
wer_baseline = evaluate_model(model)
for key, value in wer_baseline.items():
    print(f"{key}: {value:.3%}")


## -------------------------- fine-tune the model ---------------------------

## apply post-processing steps on the 8-bit model to enable training
model = prepare_model_for_kbit_training(model)
model.train()

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

## make conv layers in the encoder trainable
model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

## config lora
config = LoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none")
model = get_peft_model(model, config)
model.print_trainable_parameters()

## init WandB project
# wandb.init(project=wandb_proj_name, mode="offline")

def batch_decode_in_chunks(predictions, tokenizer, chunk_size=100):
    """Decodes predictions in smaller chunks to reduce memory usage."""
    decoded = []
    for i in range(0, len(predictions), chunk_size):
        decoded.extend(tokenizer.batch_decode(predictions[i:i + chunk_size], skip_special_tokens=True))
    return decoded

## define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=trainer_output_path,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-4,
    warmup_steps=50,
    num_train_epochs=20,             # TODO 
    weight_decay=0.01,
    push_to_hub=False,
    fp16=True,
    # no_cuda=False,                # not use GPU
    per_device_eval_batch_size=32,
    generation_max_length=128,
    logging_strategy="steps",    # "epoch"
    logging_steps=20,            # print training logs averaged over every 20 steps   # TODO
    eval_strategy="steps",       # "epoch"
    eval_steps=20,               # evaluation and save happens every 20 steps
    save_strategy="steps",       # "epoch"
    save_steps=20,
    save_total_limit = 7,        # only last 7 models are saved, older ones are deleted
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,     # lower is better
    # report_to="none",     # disable wandb logging
    # max_steps=100, # only for testing purposes, remove this from the full run
    remove_unused_columns=False,  # required as the PeftModel forward doesn't have the signature of the wrapped model's forward
    label_names=["labels"]  # same reason as above
)

## this callback helps to save only the adapter weights and remove the base model weights
class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        ## by step
        # checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        ## by epoch
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-epoch{state.epoch}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        return control

## define trainer
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=HI_dataset_trainvaltest["train"],
    eval_dataset=HI_dataset_trainvaltest["val"],
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[SavePeftModelCallback,   # save adapter weights
               EarlyStoppingCallback(early_stopping_patience=5)  # stop if no improvement after 5 evals
               ],
)
model.config.use_cache = False  # silence the warnings

## start training
trainer.train()                                             # TODO

## finish WandB run
# wandb.finish()


## ----------------- evaluate best model after training -------------------

## Load base model again
base_model = WhisperForConditionalGeneration.from_pretrained(model_name,
                                                        load_in_8bit=True,      # True if on gpu
                                                        device_map="auto"       # "auto" if on gpu
                                                        )

## Load best checkpoint
best_checkpoint = trainer.state.best_model_checkpoint
print("loading best model from checkpoint: ", best_checkpoint)
best_model = PeftModel.from_pretrained(base_model, best_checkpoint, local_files_only=True)

## save the fine-tuned model and tokenizer
best_model.save_pretrained(os.path.join(trainer_output_path,"best_model"))  # save model

best_model.eval()  # set to evaluation mode


## ----------------- evaluate fine-tuned model ------------------------

print("Performance after fine-tuning (in percentage): ")
wer_finetuned = evaluate_model(best_model)
for key, value in wer_finetuned.items():
    print(f"{key}: {value:.3%}")

werr = (wer_finetuned["wer"] - wer_baseline["wer"]) / wer_baseline["wer"]
print(f"WERR: {werr:.3%}")


# -------------------------------------------------------------------
# get memory and time statistics
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / (1024 * 1024):.2f} MB")
print(f"Peak memory usage: {peak / (1024 * 1024):.2f} MB")
# GPU memory summarys
allocated_after = torch.cuda.memory_allocated(0) / (1024**3)
reserved_after = torch.cuda.memory_reserved(0) / (1024**3)
print(f"Allocated GPU Memory After Inference: {allocated_after:.2f} GB")
print(f"Reserved GPU Memory After Inference: {reserved_after:.2f} GB")
torch.cuda.memory_summary(device=0, abbreviated=False)

tracemalloc.stop()