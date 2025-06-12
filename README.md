# Final Year Project: Development of a Speaker-Independent Automatic Deaf Speech Recognition System

This repo contains the main code developed for the experiments and analyses in the project. 

The dataset used in this project is not contained in this repo. However, the file structure is shown below:
```
CorpusOfDeafSpeech
├─data_info
│  ├─correlations
│  ├─params
│  └─triphone_analysis
├─Deaf
│  ├─S1
│  ├─S1_aligned
│  ├─S2
│  ├─S2_aligned
|  └─...(to S31)
├─Normal Hearing
│  ├─NH01
│  ├─NH01_aligned
│  ├─NH02
│  ├─NH02_aligned
|  └─...(to NH10)
├─ReadingsText
│  └─txt
└─triphone_results
```

In each folder named by a subject ID, there are three folders, each containing audio files (`.wav`) and their corresponding groundtruth text files (`.txt`) of that speaker. 

For example:
```
NH01
├─Davy Crockett
│      NH1DC1DR2.txt
│      NH1DC1DR2.wav
│      NH1DC2DR2.txt
│      NH1DC2DR2.wav
|      ...
│
├─Passages
│      NH1CGCDR2.txt
│      NH1CGCDR2.wav
│      NH1GPDR2.txt
│      NH1GPDR2.wav
|      ...
│
└─Short Story
        NH1C1DR2.txt
        NH1C1DR2.wav
        NH1C2DR2.txt
        NH1C2DR2.wav
        ...
```

In each folder named by a subject ID plus "aligned", there are three folders, each containing the word and phone level alignments generated using Montreal Forced Aligner (`.TextGrid`) and their corresponding CSV versions (`.csv`). 

For example:
```
NH01_aligned
├─Davy Crockett
│      NH1DC1DR2.csv
│      NH1DC1DR2.TextGrid
│      NH1DC2DR2.csv
│      NH1DC2DR2.TextGrid
|      ...
│
├─Passages
│      NH1CGCDR2.csv
│      NH1CGCDR2.TextGrid
│      NH1GPDR2.csv
│      NH1GPDR2.TextGrid
|      ...
│
└─Short Story
        NH1C1DR2.csv
        NH1C1DR2.TextGrid
        NH1C2DR2.csv
        NH1C2DR2.TextGrid
        ...
```

