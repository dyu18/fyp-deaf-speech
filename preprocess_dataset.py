"""
This code implements the train-val-test split of "leave-passages/prompts-out (LPO)" method 
for the Corpus of Deaf Speech.
"""

import time
start_time = time.perf_counter()        # timing the process

import numpy as np
import pandas as pd
import torch
import os
from datasets import Dataset, DatasetDict, Audio
from transformers import WhisperProcessor, WhisperFeatureExtractor, WhisperTokenizer
from datasets import load_from_disk
import tracemalloc
tracemalloc.start()         # measuring memory usage

# --------------------------------------------------------------------------------

DATA_PATH = "CorpusOfDeafSpeech"
METADATA = "metadata_chunked"      # TODO
DATASET = "LPO_chunked"      # TODO: define the name of the dataset
SEED = 42
BATCH_SIZE = 32

if torch.cuda.is_available():
    device = "cuda"
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

## assign passages to each dataset split
def select_val_test_passages(val_ratio=0.1,test_ratio=0.2, group="HI"):
    metadata_path = os.path.join(DATA_PATH,"data_info",f"{group}_{METADATA}.csv")

    ## load metadata
    df_metadata = pd.read_csv(metadata_path)
    ## remove rows with nan in groundtruth_30sec (mfa not able to align)
    df_metadata.dropna(subset=['groundtruth_30sec'], inplace=True)

    train_ratio = 1-val_ratio-test_ratio
    total_count = len(df_metadata)
    counts_per_passage = df_metadata['passage_id'].value_counts().rename('num_samples')
    
    subjects_per_passage = df_metadata.groupby('passage_id')['subject_id'].nunique().rename('num_subjects')
    ## reindex sample_counts using the sorted order of unique_subjects
    ## combine into one DataFrame
    combined = pd.concat([counts_per_passage, subjects_per_passage], axis=1)
    ## sort by num_subjects
    combined_sorted = combined.sort_values(by='num_subjects', ascending=False)
    print(combined_sorted)

    target_train = total_count * train_ratio
    target_val = total_count * val_ratio
    target_test = total_count * test_ratio

    train, val, test = [], [], []       # to store passage_id
    counts = {'train': 0, 'val': 0, 'test': 0}

    for pid, row in combined_sorted.iterrows():
        sample_count = row['num_samples']
        if counts['test'] + sample_count <= target_test:
            test.append(pid)
            counts['test'] += sample_count
        elif counts['val'] + sample_count <= target_val:
            val.append(pid)
            counts['val'] += sample_count
        else:
            train.append(pid)
            counts['train'] += sample_count
    print(counts)
    print(counts['val']/total_count)
    print(counts['test']/total_count)
    
    return train, val, test


HI_train_passages, HI_val_passages, HI_test_passages = select_val_test_passages()

print("Selected train passages:")
print(HI_train_passages)
print("Selected val passages:")
print(HI_val_passages)
print("Selected test passages:")
print(HI_test_passages)

# load metadata
metadata_path = os.path.join(DATA_PATH,"data_info",f"HI_{METADATA}.csv")
df_metadata = pd.read_csv(metadata_path)
# remove rows with nan in groundtruth_30sec (mfa not able to align)
df_metadata.dropna(subset=['groundtruth_30sec'], inplace=True)

for intelli in ["high", "medium", "low"]:
    print()
    print(intelli)
    df = df_metadata.loc[df_metadata['intelligibility'] == intelli]

    df_train = df[df['passage_id'].isin(HI_train_passages)]
    print("train set:")
    print(len(df_train))
    print(len(df_train)/len(df))

    df_val = df[df['passage_id'].isin(HI_val_passages)]
    print("val set:")
    print(len(df_val))
    print(len(df_val)/len(df))

    df_test = df[df['passage_id'].isin(HI_test_passages)]
    print("test set:")
    print(len(df_test))
    print(len(df_test)/len(df))

# --------------------------------------------------------------------------------

def load_dataset(group, intelligibility=None, minibatch=False, batchsize=10, val_passages=None, test_passages=None, excluded_subject_ids=[]):
    if group == "HI":
        group_directory = "Deaf"
    elif group == "NH":
        group_directory = "Normal Hearing"
    else:
        raise Exception("Invalid group, must be either \"HI\" (deaf) or \"NH\" (normal hearing)")
    dataset_path = os.path.join(DATA_PATH,group_directory)
    metadata_path = os.path.join(DATA_PATH,"data_info",f"{group}_{METADATA}.csv")

    ## load metadata
    df = pd.read_csv(metadata_path)

    ## remove rows with nan in groundtruth_30sec (mfa not able to align)
    df.dropna(subset=['groundtruth_30sec'], inplace=True)

    ## select intelligibility if deaf group
    if group == "HI":
        if intelligibility != None and intelligibility in ["high", "medium", "low"]:
            df = df.loc[df['intelligibility'] == intelligibility]      # select according to intelligibility

    ## take a minibatch if prototyping
    if minibatch:
        df = df.head(batchsize)
    
    ## exclude subjects in the given list
    df = df[~df['subject_id'].isin(excluded_subject_ids)]

    ## split the dataset
    ## put samples with passage_id in the HI_test_passages list into the test set
    if test_passages != None and val_passages != None:  # train-val-test split
        ## split into train, val, and test sets
        df_test = df[df['passage_id'].isin(test_passages)]
        df_val = df[df['passage_id'].isin(val_passages)]
        df_trainval = df[~df['passage_id'].isin(test_passages)]
        df_train = df_trainval[~df_trainval['passage_id'].isin(val_passages)]
        ## convert to Hugging Face dataset format
        ## and shuffle them separately
        dataset_test = Dataset.from_pandas(df_test).shuffle(seed=SEED)
        dataset_val = Dataset.from_pandas(df_val).shuffle(seed=SEED)
        dataset_train = Dataset.from_pandas(df_train).shuffle(seed=SEED)
        dataset = DatasetDict({
                               'train':dataset_train,
                               'val':dataset_val,
                               'test': dataset_test
                              })
    elif test_passages != None:                         # train-test split
        ## split into train and test sets
        df_test = df[df['passage_id'].isin(test_passages)]
        df_train = df[~df['passage_id'].isin(test_passages)]
        ## convert to Hugging Face dataset format
        ## and shuffle them separately
        dataset_test = Dataset.from_pandas(df_test).shuffle(seed=SEED)
        dataset_train = Dataset.from_pandas(df_train).shuffle(seed=SEED)
        dataset = DatasetDict({'train':dataset_train,
                               'test': dataset_test
                              })
    else:                                               # no split
        ## convert to Hugging Face dataset format
        dataset = Dataset.from_pandas(df)
        ## split and shuffle
        dataset = dataset.train_test_split(test_size=0.2)

    ## helper function to load audio files
    def load_audio(sample):
        passage_id = sample["passage_id"]
        if passage_id.startswith("DC"):
            passage_folder = "Davy Crockett"
        elif passage_id.startswith("C") and passage_id[1:].isdigit():
            passage_folder = "Short Story"
        else:
            passage_folder = "Passages"

        filepath = os.path.join(sample["subject_id"], passage_folder, sample["filename"])
        sample["audio"] = {"path": os.path.join(dataset_path, filepath)}
        return sample

    ## load audio for the entire dataset
    dataset = dataset.map(load_audio)
    ## cast to Audio format
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    return dataset


def preprocess(batch, feature_extractor=feature_extractor, tokenizer=tokenizer):
    ## load audio data
    audio_arrays = [audio_sample["array"] for audio_sample in batch["audio"]]
    sr = batch["audio"][0]["sampling_rate"] # assume all samples have the same sampling rate
    start_times = batch["start"]
    end_times = batch["end"]

    ## get 30-second chunks
    truncated_audio_arrays = []
    for audio_array, start, end in zip(audio_arrays,start_times,end_times):
        start_idx = int(start*sr)
        end_idx = int(end*sr)
        truncated_audio_arrays.append(audio_array[start_idx:end_idx])

    ## compute log-Mel input features from input audio array
    input_features = feature_extractor(truncated_audio_arrays, 
                                       sampling_rate=sr, 
                                       return_tensors="pt", 
                                       padding='max_length',
                                       truncation=True,
                                       ).input_features
    batch["input_features"] = input_features

    ## load groundtruth transcription
    groundtruth = batch["groundtruth_30sec"]

    ## encode target text to label ids
    labels = tokenizer(groundtruth).input_ids
    batch["labels"] = labels

    return batch

# --------------------------------------------------------------------------------

def preprocess_dataset(intelligibility, dataset_dir, num_proc=1, minibatch=False, excluded_subject_ids=[]):
    ## load dataset
    HI_dataset_intelli = load_dataset("HI", 
                                      intelligibility=intelligibility, 
                                      val_passages=HI_val_passages,
                                      test_passages=HI_test_passages, 
                                      minibatch=minibatch,
                                      excluded_subject_ids=excluded_subject_ids)

    ## preprocess the dataset
    ## and remove not needed columns, only leave "input_features" and "labels"
    batched = True # False
    HI_dataset_trainvaltest = HI_dataset_intelli.map(preprocess, 
                                                  remove_columns=["audio", "groundtruth_30sec"], 
                                                #   remove_columns=HIdataset_intelli.column_names["train"],
                                                  num_proc=num_proc, 
                                                  batched=batched, 
                                                  batch_size=BATCH_SIZE)

    ## save processed dataset
    HI_dataset_trainvaltest.save_to_disk(dataset_dir)

    print(f"Preprocessing completed for {intelligibility}")
    return

## perform preprocessing
for intelli in ["high", "medium", "low"]:
    dataset_path = os.path.join(DATA_PATH, f"HI_dataset_trainvaltest_{intelli}_{DATASET}")
    preprocess_dataset(intelli, dataset_dir=dataset_path)
    ## load dataset from saved location
    HI_dataset_trainvaltest = load_from_disk(dataset_path)
    print("Train samples:", len(HI_dataset_trainvaltest['train']))
    print("Val samples:", len(HI_dataset_trainvaltest['val']))
    print("Test samples:", len(HI_dataset_trainvaltest['test']))
    total_samples = sum(len(split) for split in HI_dataset_trainvaltest.values())
    print("Total samples:", total_samples)


##------- sweetspot set -------
intelli = "medium"
excluded_subject_no = [29, 26, 30, 15, 12, 6]
excluded_subject_ids = ['S'+str(i) for i in excluded_subject_no]
dataset_path = os.path.join(DATA_PATH, f"HI_dataset_trainvaltest_{intelli}_{DATASET}_sweetspot")
preprocess_dataset(intelli, dataset_dir=dataset_path, excluded_subject_ids = excluded_subject_ids)

## load dataset from saved location
HI_dataset_trainvaltest = load_from_disk(dataset_path)
print("Train samples:", len(HI_dataset_trainvaltest['train']))
print("Val samples:", len(HI_dataset_trainvaltest['val']))
print("Test samples:", len(HI_dataset_trainvaltest['test']))
total_samples = sum(len(split) for split in HI_dataset_trainvaltest.values())
print("Total samples:", total_samples)

### ------- complement set ------
excluded_subject_no = [27, 25, 7, 31, 23]
excluded_subject_ids = ['S'+str(i) for i in excluded_subject_no]
dataset_path = os.path.join(DATA_PATH, f"HI_dataset_trainvaltest_{intelli}_{DATASET}_comp")
preprocess_dataset(intelli, dataset_dir=dataset_path, excluded_subject_ids = excluded_subject_ids)

## load dataset from saved location
HI_dataset_trainvaltest = load_from_disk(dataset_path)
# print("batch size: 10")
print("Train samples:", len(HI_dataset_trainvaltest['train']))
print("Val samples:", len(HI_dataset_trainvaltest['val']))
print("Test samples:", len(HI_dataset_trainvaltest['test']))
total_samples = sum(len(split) for split in HI_dataset_trainvaltest.values())
print("Total samples:", total_samples)

# -------------------------------------------------------------------
# get memory and time statistics
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / (1024 * 1024):.2f} MB")
print(f"Peak memory usage: {peak / (1024 * 1024):.2f} MB")

tracemalloc.stop()