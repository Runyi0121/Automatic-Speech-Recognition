# Speech-to-IPA
## English Audio → International Phonetic Alphabet Transcription

A fine-tuned speech recognition model that converts English audio into International Phonetic Alphabet (IPA) transcriptions, built on top of Facebook's `wav2vec2-large-xlsr-53` model.

---

## Overview

This project fine-tunes a Wav2Vec2 model to predict IPA transcriptions from English speech audio. Rather than producing standard word transcriptions, the model outputs phonetic representations — useful for language learners, linguistics research, and pronunciation analysis.

Training data comes from the [George Mason University Speech Accent Archive](https://accent.gmu.edu), which pairs MP3 audio recordings with corresponding IPA transcriptions.

---

## Results

| Split | Files | Character Error Rate (CER) |
|-------|-------|---------------------------|
| Train | 114   | —                         |
| Dev   | 37    | ~0.1923 (19.23%)          |
| Test  | 38    | ~0.2034 (20.34%)          |

CER measures the percentage of characters in the predicted transcription that differ from the reference IPA. Remaining errors are partly attributable to alternate phoneme representations for the same sounds.

---

## Setup

Install required dependencies:

```bash
pip install -r requirements.txt
```

Key libraries: `transformers`, `datasets`, `pandas`, `torch`, `tqdm`

---

## Usage

### 1. Preprocess Data

Run the following steps in order:

```bash
# Convert IPA transcription files from .gif to .txt
python3 rename_transcriptions.py

# Normalize all filenames to lowercase
bash renameE.sh
```

Then manually create a `metadata.csv` with two columns: the MP3 audio filename and its corresponding transcription filename.

```bash
# Replace transcription filenames in metadata.csv with actual IPA content
python3 txt_entry.py
```

Finally, clean each transcription file so it contains only the IPA string (remove headers like "English Speaker 8").

---

### 2. Train

```bash
python3 main.py -e 22 -vo vocab.json
```

This will:
- Load and preprocess the audio dataset
- Build a character-level IPA vocabulary (`vocab.json`)
- Fine-tune `facebook/wav2vec2-large-xlsr-53` with CTC loss
- Use an AdamW optimizer and cosine learning rate scheduler
- Save checkpoints throughout training (each ~1.2 GB)

---

### 3. Evaluate

Update the vocabulary file for the checkpoint you want to evaluate:

```bash
python3 upload.py \
  -m ./wav2vec2-large-xlsr-53-english/checkpoint-900 \
  -v ./vocab.json \
  -d ./wav2vec2-large-xlsr-53-english/checkpoint-900/
```

Run evaluation on the dev or test set:

```bash
python3 eval.py \
  -m ./wav2vec2-large-xlsr-53-english/checkpoint-900/ \
  -o dev_result.txt
```

---

## Code & Data

Full code and results are available here:  
[Google Drive](https://drive.google.com/drive/folders/1hT3sXC-st9imOlroo-yrpYH5rhAfJuG0?usp=share_link)

---

## Model Architecture

- **Base model:** `facebook/wav2vec2-large-xlsr-53`
- **Task:** CTC-based sequence-to-sequence transcription
- **Vocabulary:** Character-level IPA symbols (generated from training data)
- **Loss:** Connectionist Temporal Classification (CTC)

---

## Author

Runyi Shi
