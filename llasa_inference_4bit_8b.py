"""
This code performs data augmentation using Llasa-8B 
with normalised transcriptions obtained from the LJ Speech Dataset
for selected speakers in the Corpus of Deaf Speech. 
Code adapted from https://github.com/nivibilla/local-llasa-tts/blob/main/colab_notebook_4bit.ipynb,
created and published by the authors of Llasa TTS.
"""

import time
start_time = time.perf_counter()        # timing the process

import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import soundfile as sf
from xcodec2.modeling_xcodec2 import XCodec2Model
import torchaudio
from IPython.display import Audio
import soundfile as sf
import numpy as np
import pandas as pd


quantization_config = BitsAndBytesConfig(load_in_4bit=True)

## TODO: choose 3b or 8b model
model_size = '8b' # '3b'
llasa =f'tts/llasa-{model_size}'

# be patient this takes a couple mins...
tokenizer = AutoTokenizer.from_pretrained(llasa)

model = AutoModelForCausalLM.from_pretrained(
    llasa,
    trust_remote_code=True,
    device_map='auto',
    quantization_config=quantization_config,
    low_cpu_mem_usage=True
)

model_path = "tts/xcodec2"

Codec_model = XCodec2Model.from_pretrained(model_path, low_cpu_mem_usage=True)
Codec_model.eval().cuda()


def ids_to_speech_tokens(speech_ids):

    speech_tokens_str = []
    for speech_id in speech_ids:
        speech_tokens_str.append(f"<|s_{speech_id}|>")
    return speech_tokens_str

def extract_speech_ids(speech_tokens_str):

    speech_ids = []
    for token_str in speech_tokens_str:
        if token_str.startswith('<|s_') and token_str.endswith('|>'):
            num_str = token_str[4:-2]

            num = int(num_str)
            speech_ids.append(num)
        else:
            print(f"Unexpected token: {token_str}")
    return speech_ids

def infer(sample_audio_path, 
          target_text, 
          prompt_text,   #=None
          start=0,
          ):
    waveform, sample_rate = torchaudio.load(sample_audio_path)
    if len(waveform[0])/sample_rate > 15:
        end = start+15
        print(f"Trimming audio to first voiced 15secs: from {start} to {end}")
        waveform = waveform[:, int(sample_rate*start) : int(sample_rate*end)]

    # Check if the audio is stereo (i.e., has more than one channel)
    if waveform.size(0) > 1:
        # Convert stereo to mono by averaging the channels
        waveform_mono = torch.mean(waveform, dim=0, keepdim=True)
    else:
        # If already mono, just use the original waveform
        waveform_mono = waveform

    prompt_wav = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform_mono)

    # if prompt_text is None:
    #     prompt_text = whisper_turbo_pipe(prompt_wav[0].numpy())['text'].strip()

    print(f"Prompt: {prompt_text}")
    print(f"Target: {target_text}")

    if len(target_text) == 0:
        return None
    elif len(target_text) > 300:
        print("Text is too long. Please keep it under 300 characters.")
        target_text = target_text[:300]

    input_text = prompt_text + ' ' + target_text

    #TTS start!
    with torch.no_grad():
        # Encode the prompt wav
        vq_code_prompt = Codec_model.encode_code(input_waveform=prompt_wav)

        vq_code_prompt = vq_code_prompt[0,0,:]
        # Convert int 12345 to token <|s_12345|>
        speech_ids_prefix = ids_to_speech_tokens(vq_code_prompt)

        formatted_text = f"<|TEXT_UNDERSTANDING_START|>{input_text}<|TEXT_UNDERSTANDING_END|>"

        # Tokenize the text and the speech prefix
        chat = [
            {"role": "user", "content": "Convert the text to speech:" + formatted_text},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>" + ''.join(speech_ids_prefix)}
        ]

        input_ids = tokenizer.apply_chat_template(
            chat,
            tokenize=True,
            return_tensors='pt',
            continue_final_message=True
        )
        input_ids = input_ids.to('cuda')
        speech_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')

        # Generate the speech autoregressively
        outputs = model.generate(
            input_ids,
            max_length=2048,  # We trained our model with a max length of 2048
            eos_token_id= speech_end_id ,
            do_sample=True,
            top_p=0.95, # 1
            temperature=0.9 #0.8
        )
        # Extract the speech tokens
        generated_ids = outputs[0][input_ids.shape[1]-len(speech_ids_prefix):-1]

        speech_tokens = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Convert  token <|s_23456|> to int 23456
        speech_tokens = extract_speech_ids(speech_tokens)

        speech_tokens = torch.tensor(speech_tokens).cuda().unsqueeze(0).unsqueeze(0)

        # Decode the speech tokens to speech waveform
        gen_wav = Codec_model.decode_code(speech_tokens)

        # if only need the generated part
        gen_wav = gen_wav[:,:,prompt_wav.shape[1]:]

    return gen_wav[0, 0, :].cpu().numpy()



## -----------------------------------------------------
DATA_PATH = "CorpusOfDeafSpeech"
group_dir = "Deaf"

## load LJSpeech dataset
dataset = torchaudio.datasets.LJSPEECH(root="LJSpeech")
transcript_id_start = 0
transcript_id_end = 186

metadata_path = os.path.join(DATA_PATH,"data_info","HI_metadata_15sec_voiced.csv")
df = pd.read_csv(metadata_path)
HI_intell_high = [1,4,5,8,9,10]
HI_intell_med = [6,7,12,15,23,25,26,27,29,30,31]
HI_intell_low = [2,3,13,14,16,17,18,19,20,21,22,24,28]
HI_intell_med_tts = [6,7,12,23,25,27,31]         ## TODO: specify subjects to augment

passage_dir = 'Davy Crockett'
passage_id = "DC4"
# subject_list = ['S'+str(i) for i in range(1,32) if i!=11]         # all subjects
subject_list = ['S'+str(i) for i in HI_intell_med_tts]

for subject_id in subject_list:
    print(subject_id)   # debug
    sample = df[(df['subject_id'] == subject_id) & (df['passage_id'] == passage_id)]
    audio_filename = sample['filename'].values[0]

    output_dir = os.path.join('tts','output','ljspeech',subject_id)
    os.makedirs(output_dir, exist_ok=True)    # TODO
    
    if pd.isna(sample['groundtruth_15sec']).values[0]==False:     ## not empty
        prompt_text = "d c four" + sample['groundtruth_15sec'].values[0]
        start = sample['start'].values[0]

        audio_path = os.path.join(DATA_PATH,group_dir,subject_id,passage_dir,audio_filename)
        samplerate = 16000  # Hz
        
        ##
        for transcript_i in range(transcript_id_start,transcript_id_end):
            _waveform, _sample_rate, _transcript, normalized_transcript = dataset[transcript_i]
            target_text = normalized_transcript

            ## generate output audio
            output_audio = infer(audio_path, target_text, prompt_text=prompt_text, start=start)

            ## save output audio locally
            sf.write(os.path.join(output_dir, f'{subject_id}_LJ{transcript_i}.wav'), output_audio, samplerate)


# -------------------------------------------------------------------
# Get time statistics
end_time = time.perf_counter()
print(f"Elapsed time: {end_time - start_time:.6f} seconds")

