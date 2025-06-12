"""
This code implements the deployment of fine-tuned Whisper model on the Corpus of Deaf Speech. 
"""

import time
start_time = time.perf_counter()        # timing the process

import os
import whisper
import pandas as pd
import jiwer
import torch
from transformers import pipeline
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel, PeftConfig

DATA_PATH = "CorpusOfDeafSpeech"

## TODO: choose one set between NH and HI
group = "NH"
group_dir = "Normal Hearing"
subject_list = ["NH01","NH02","NH03","NH04","NH05","NH06","NH07","NH08","NH09","NH10"]    ## TODO

group = "HI"
group_dir = "Deaf"
intelligibility =  "low" # "high" # "medium"
HI_intell_high = [1,4,5,8,9,10]
HI_intell_med = [6,7,12,15,23,25,26,27,29,30,31]  # 
HI_intell_low = [2,3,13,14,16,17,18,19,20,21,22,24,28]
subject_list = ["S"+str(i) for i in range(1,32) if i!=11]   ## total: 31 subjects

## select model
if intelligibility == "high":
    subject_list = ["S"+str(i) for i in HI_intell_high]
    model_checkpoint = "trainer_output_high_with_callbacks_LPO_steps_chunked_7_1_2/checkpoint-100"     ## TODO
elif intelligibility == "medium":
    subject_list = ["S"+str(i) for i in HI_intell_med]
    model_checkpoint = "trainer_output_medium_with_callbacks_LPO_epochs_chunked/checkpoint-180"  ## TODO
elif intelligibility == "low":
    subject_list = ["S"+str(i) for i in HI_intell_low]
    model_checkpoint = "trainer_output_low_with_callbacks_LPO_steps_chunked_7_1_2/checkpoint-120"      ## TODO
else:
    raise Exception("Invalid intelligibility group.")


## Load Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

config = PeftConfig.from_pretrained(model_checkpoint)
## load base model
base_model = WhisperForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
## load finetuned model
model = PeftModel.from_pretrained(model=base_model, model_id=model_checkpoint)
model = model.merge_and_unload()
## load processor
processor = WhisperProcessor.from_pretrained(config.base_model_name_or_path)
## load into a pipeline
asr_pipeline = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30       ## TODO
)


audio_dir = os.path.join(DATA_PATH, group_dir)
wer_path = os.path.join(DATA_PATH,"data_info",f"{group}_wer_HIfinetuned_{intelligibility}.csv")   ## TODO

## load metadata
metadata_path = os.path.join(DATA_PATH,"data_info",f"{group}_metadata.csv")
metadata_df = pd.read_csv(metadata_path)

## define text normalisations
transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ExpandCommonEnglishContractions()
])

def get_wer_breakdown(predictions, references):
    ## normalise texts
    predictions = transformation(predictions)
    references = transformation(references)

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


## iterate over audio files
for (i,subject) in enumerate(subject_list):  
    wer_list = []    # list of dicts
    subject_dir = os.path.join(audio_dir, subject)  ## TODO

    for root, dirs, files in os.walk(subject_dir):
        for file in files:
            if (file.endswith(".wav")):
                file_path = os.path.join(root, file)
                if (metadata_df["filename"]==file).any():   ## check if a filename exists in metadata
                    ## get passage_id from metadata
                    passage_id = metadata_df[metadata_df["filename"]==file]["passage_id"].iloc[0]
                    print(f"Transcribing {file}...")
                    
                    ## get groundtruth from metadata
                    groundtruth = metadata_df[metadata_df["filename"]==file]["groundtruth"].iloc[0]
                    
                    ## transcribe the audio file with pipeline
                    result = asr_pipeline(file_path, return_timestamps=True)
                    # each chunk is a dictionary with 'text' and 'timestamp'
                    chunks = result["chunks"]
                    # extract and join all the texts
                    transcription = " ".join(chunk["text"].strip() for chunk in chunks)

                    ## calculate WER
                    wer_info = get_wer_breakdown(transcription,groundtruth)
                    wer_info["filename"]=file
                    wer_info["transcription"]=transcription
                    wer_list.append(wer_info)
    
    ## convert into dataframe
    wer_df = pd.DataFrame(wer_list)
    ## move the filename column (8th col, index 7) to the front
    cols = wer_df.columns.tolist()
    n = 7
    filename_col = cols[n]
    new_order = [filename_col] + cols[:n] + cols[(n+1):]
    wer_df = wer_df[new_order]

    ## load previously saved dataframe
    wer_df_prev = pd.read_csv(wer_path)
    ## concat two dataframes
    wer_df_new = pd.concat([wer_df_prev, wer_df], ignore_index=True)

    ## save dataframe to csv file
    wer_df_new.to_csv(wer_path, index=False)


## ----------------------------------------------------------
## get time statistics
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")

