"""
This code implements the extraction of openSMILE (GeMAPS) acoustic parameters 
and the correlation analysis for each parameter with WER
for samples from the Corpus of Deaf Speech.
"""

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import opensmile
from tqdm import tqdm
from scipy import stats
from sklearn.manifold import TSNE

DATA_PATH = "CorpusOfDeafSpeech"
METADATA = "metadata" # "metadata_chunked"
group = "NH" #"HI"

if group == "HI":
    group_dir = "Deaf"
elif group == "NH":
    group_dir = "Normal Hearing"
metadata_path = os.path.join(DATA_PATH,"data_info",f"{group}_{METADATA}.csv")

## load metadata
df_metadata = pd.read_csv(metadata_path)
## remove rows with nan in groundtruth_30sec (mfa not able to align)
if group == "HI":
    df_metadata.dropna(subset=['groundtruth_30sec'], inplace=True)
    df_metadata = df_metadata.reset_index(drop=True)        # important!!

## set up opensmile feature extractor
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def get_acoustic_params(sample):
    passage_id = sample.passage_id
    if passage_id.startswith("DC"):
        passage_folder = "Davy Crockett"
    elif passage_id.startswith("C") and passage_id[1:].isdigit():
        passage_folder = "Short Story"
    else:
        passage_folder = "Passages"

    subject_id = sample.subject_id
    if (group == "NH") and (not subject_id.endswith("0")):    # special processing for NH group
        subject_id = subject_id[:-1] + "0" + subject_id[-1]

    filepath = os.path.join(subject_id, passage_folder, sample.filename)
    audio_path = os.path.join(DATA_PATH, group_dir, filepath)

    # get opensmile params
    y, sr = librosa.load(audio_path)
    params = smile.process_signal(y,sr)

    return params

## add tqdm support to pandas
tqdm.pandas()

df_params = None
## extract acoustic params for the entire dataset
for sample in tqdm(df_metadata.itertuples(index=True)):
    params = get_acoustic_params(sample)
    if df_params is None:
        df_params = params
    else:
        df_params = pd.concat([df_params, params], axis=0, ignore_index=True)

## merge dataframes
df_metadata = df_metadata.drop("groundtruth", axis=1)
if group == "HI":
    df_metadata = df_metadata.drop("groundtruth_30sec", axis=1)
## beware of index mismatch!!
df_metadata = df_metadata.reset_index(drop=True)
df_params = df_params.reset_index(drop=True)
df_params = pd.concat([df_metadata, df_params], axis=1)

## save as csv file
csv_path = os.path.join(DATA_PATH,"data_info","params",f"{group}_params_all.csv")
df_params.to_csv(csv_path, encoding = 'utf-8-sig', index=False)


## ---------------------------------------------

## correlation analysis for HI group
group = "HI"
wer_path = os.path.join(DATA_PATH,"data_info",f"{group}_wer.csv")
df_wer = pd.read_csv(wer_path)
merged_df = pd.merge(df_params, df_wer, on='filename', how='left')

## find spearman correlation for each 
corr_dict = []
param_y = 'wer'
for param_x in df_params.columns[7:-1]:    
    x = merged_df[param_x]
    y = merged_df[param_y]
    rho, p_val = stats.spearmanr(x, y)
    corr_dict.append({'param': param_x,
                      'rho': rho,
                      'rho_abs': abs(rho),
                      'p_value': p_val
                      })

corr_df = pd.DataFrame(corr_dict)
corr_df = corr_df.sort_values(['rho_abs'], ascending=[False]).reset_index(drop=True)
corr_df = corr_df.drop(columns=['rho_abs'])
corr_path = os.path.join(DATA_PATH, 'data_info', 'correlations', 'spearman.csv')
corr_df.to_csv(corr_path)


## ---------------------------------------------
## t-SNE visualisation

def load_data(group, num_params=3):
    wer_path = os.path.join(DATA_PATH,"data_info",f"{group}_wer.csv")
    df_wer = pd.read_csv(wer_path)  # , index_col=0


    p = f"{DATA_PATH}\data_info\params\{group}_params_all.csv"
    df_params = pd.read_csv(p)
    merged_df = pd.merge(df_params, df_wer, on='filename', how='left')
    merged_df = merged_df.reset_index(drop=True)        # important!!
    

    param_list = corr_df['param'].to_list()[:num_params]
    print(param_list)

    sel_params_df = merged_df[param_list]
    x = np.array(sel_params_df)

    ## process target
    if group == "HI":
        target_names = merged_df['intelligibility'].to_list()
    else:
        target_names = ['normal'] * len(x)

    return x, target_names


## select some parameters to keep
num_params = 3 # 88      ## TODO

x_NH, target_names_NH = load_data("NH", num_params=num_params)
x_HI, target_names_HI = load_data("HI", num_params=num_params)
x = np.vstack((x_NH, x_HI))
target_names = target_names_NH
target_names.extend(target_names_HI)
target_names = np.asarray(target_names)

## make sure the datapoints are unique (required for Sammon mapping)
(x,index) = np.unique(x,axis=0,return_index=True)
target_names = target_names[index]
print(x.shape)
print(target_names.shape)


## run t-SNE to reduce to 2 dimensions
perp = 30
tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
x_2d = tsne.fit_transform(x)

## plot
plt.figure()
plt.scatter(x_2d[target_names == 'high', 0], x_2d[target_names == 'high', 1], s=20, alpha=0.5, label='Deaf (high)')
plt.scatter(x_2d[target_names == 'medium', 0], x_2d[target_names == 'medium', 1], s=20, alpha=0.5, label='Deaf (medium)')
plt.scatter(x_2d[target_names == 'low', 0], x_2d[target_names == 'low', 1], s=20, alpha=0.5, label='Deaf (low)')
plt.scatter(x_2d[target_names == 'normal', 0], x_2d[target_names == 'normal', 1], s=20, alpha=0.5, label='Normal')
plt.title(f't-SNE with top {num_params} parameters and perplexity {perp}')
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(loc=2)
plt.show()

