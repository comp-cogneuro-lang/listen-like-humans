import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'analysis')))
import time
import unicodedata
import soundfile as sf
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, AutoModelForCTC, AutoProcessor
from phonological_competition import Category_Dict_Generate, dict_to_dataframe
import glob
import random

# Configuration
datapath = './dataset/'
model_name = os.environ.get("MODEL_NAME", "facebook/wav2vec2-base-960h")
basepath = os.environ.get("OUTPUT_DIR", "experiments/wav2vec2")
os.makedirs(basepath, exist_ok=True)
os.makedirs(os.path.join(basepath, "imgs"), exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = time.time()
dealy20 = True

print(f"Loading Wav2Vec2 model: {model_name}")
# Use Auto* classes so the same script handles both wav2vec2 and hubert CTC checkpoints.
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCTC.from_pretrained(model_name).to(device)
model.eval()

# Load Pronunciation Dictionary
print("Loading pronunciation dictionary...")
# Assuming dataset/en/vocab.txt exists as in original code
pronunciation_path = "dataset/en/vocab.txt"
if not os.path.exists(pronunciation_path):
    # Fallback to dataset/phonemes_en.csv if vocabulary format changed
    print(f"Warning: {pronunciation_path} not found.")

pronunciation_dict = pd.read_csv(pronunciation_path, sep="\t", names=["word","pronunciation"])
pronunciation_dict = pronunciation_dict[["word","pronunciation"]]
pronunciation_dict["pronunciation"] = pronunciation_dict["pronunciation"].apply(lambda x: x.split("."))
pronunciation_dict = pronunciation_dict.set_index("word").to_dict(orient="index")
words_list = list(pronunciation_dict.keys())

# Generate Categories
print("Generating/Loading category dictionary...")
word_competition_dict = Category_Dict_Generate(words_list, pronunciation_dict)

def load_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    return waveform.squeeze()

def get_ctc_probability_trace(logits, target_word, processor):
    """
    Calculates the probability trace of a target word over time using CTC logits.
    """
    # Wav2Vec2-base-960h uses uppercase 
    target_text = target_word.upper()
    
    # Tokenize
    with torch.no_grad():
        target_ids = processor.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids[0].to(device)
    
    # CTC Logits: (T, V)
    log_probs = F.log_softmax(logits, dim=-1) # (T, V)
    T, V = log_probs.shape
    
    # Blank ID
    blank_id = 0 # Default for Wav2Vec2
    if hasattr(processor.tokenizer, 'pad_token_id'):
         blank_id = processor.tokenizer.pad_token_id

    # Create extended targets: [blank, t0, blank, t1, ..., blank, tN-1, blank]
    L = len(target_ids)
    S = 2 * L + 1
    extended_ids = torch.full((S,), blank_id, dtype=torch.long, device=device)
    extended_ids[1::2] = target_ids
    
    # Alpha initialization (in log space)
    alpha = torch.full((S,), -float('inf'), device=device)
    alpha[0] = log_probs[0, blank_id]
    if S > 1:
        alpha[1] = log_probs[0, extended_ids[1]]
        
    trace = []
    
    # Calculate t=0 probability
    prob_t0_log = torch.logaddexp(alpha[-1], alpha[-2]) if S > 1 else alpha[-1]
    trace.append(torch.exp(prob_t0_log).item())
    
    # Forward Pass
    for t in range(1, T):
        prev_alpha = alpha.clone()
        alpha = torch.full((S,), -float('inf'), device=device)
        
        current_log_probs = log_probs[t, extended_ids]
        
        # 1. Stay
        alpha = prev_alpha + current_log_probs
        
        # 2. Advance from s-1
        ds1 = torch.cat([torch.tensor([-float('inf')], device=device), prev_alpha[:-1]])
        alpha = torch.logaddexp(alpha, ds1 + current_log_probs)
        
        # 3. Skip from s-2
        if S > 2:
            ds2 = torch.cat([torch.tensor([-float('inf'), -float('inf')], device=device), prev_alpha[:-2]])
            
            mask = torch.ones(S, dtype=torch.bool, device=device)
            mask[0::2] = False # Can't skip to blank (even indices)
            
            # Prevent skip if target characters are same: A -> blank -> A
            # indices in extended_ids: s and s-2.
            # s=3 (index 3 is token 1), s-2=1 (index 1 is token 0).
            # extended_ids[s] == extended_ids[s-2] corresponds to tokens being same.
            
            # condition: extended_ids[s] == extended_ids[s-2]
            # Valid indices for s in skip are 2, 3, ... S-1.
            # Blanks are at even indices. s%2!=0 guarantees not blank.
            # But wait, skip logic says:
            # transitions allowed s-2 -> s IF label(s) != blank AND label(s) != label(s-2).
            
            # Implementation:
            # We want to forbid transition if label(s) == label(s-2).
            # Note: label(s) is guaranteed != blank because s is odd.
            # So just check extended_ids[s] == extended_ids[s-2].
            
            # Vectorized check:
            # Create a boolean tensor of size S
            skip_allowed = torch.zeros(S, dtype=torch.bool, device=device)
            # Only odd indices >= 3 can skip from s-2 (which is odd >= 1)
            # And specific check 
            if S >= 3:
                odd_indices = torch.arange(3, S, 2, device=device)
                # Check labels
                # We need extended_ids[odd_indices] != extended_ids[odd_indices-2]
                can_skip = extended_ids[odd_indices] != extended_ids[odd_indices-2]
                skip_allowed[odd_indices] = can_skip
            
            # Apply mask
            ds2_masked = ds2.clone()
            ds2_masked[~skip_allowed] = -float('inf')
            
            alpha = torch.logaddexp(alpha, ds2_masked + current_log_probs)
        
        # Total probability of word at time t
        final_log_prob = torch.logaddexp(alpha[-1], alpha[-2]) if S > 1 else alpha[-1]
        trace.append(torch.exp(final_log_prob).item())
        
    return np.array(trace)

# Data Collection
categorized_data_dict = {}

# Find audio files
# Assuming English dataset for now based on 'vocab.txt' path
# Adjust pattern as needed. 
wav_files = list(open("dataset/en_train.txt", "r").readlines()) + list(open("dataset/en_test.txt", "r").readlines()) 
wav_files = [os.path.join(datapath, line.strip() + ".wav") for line in wav_files if line.strip()]
print(f"Found {len(wav_files)} audio files.")

cor, tot = 0, len(wav_files)
print("Starting processing...")
for wav_path in tqdm(wav_files):
    filename = os.path.basename(wav_path)
    word_raw = filename.lower().replace('.wav', '').strip()
    # Try direct match or splitting
    # If filename is "fire_tom.wav" or similar?
    # Original: word = audio_path.split('/')[-1].replace('.wav','')
    # Unicodedata normalize
    word = unicodedata.normalize("NFC", word_raw)
    
    # Speaker extraction
    # dataset/en/speaker/word.wav ?
    parts = wav_path.split(os.sep)
    if len(parts) >= 2:
        speaker = parts[-2]
    else:
        speaker = "unknown"
        
    # Check if word in competition dict
    if word not in words_list:
        continue

    # Load Audio
    try:
        waveform = load_audio(wav_path).to(device)
    except Exception as e:
        print(f"Error loading {wav_path}: {e}")
        continue
        
    # Get Logits
    with torch.no_grad():
        if dealy20:
            # add 200ms of silence at the beginning to avoid the unusual high activation at the start
            silence = torch.zeros(int(0.2 * 16000), device=device) # 200ms of silence
            waveform = torch.cat([silence, waveform], dim=0)
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(device)
        logits = model(input_values).logits[0] # (T, V)

    # Get Prediction
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids)
    if transcription.strip().lower() == word:
        cor += 1 
    
        # Define Categories
        competition_set = {"Target": [word]}
        competition_set["Cohort"] = word_competition_dict.get((word, "Cohort"), [])
        competition_set["Rhyme"] = word_competition_dict.get((word, "Rhyme"), []) 

        # Compute Traces
        for category in ["Target", "Cohort", "Rhyme"]:
            cand_words = competition_set.get(category, [])
            if not cand_words:
                categorized_data_dict[0, word, speaker, category] = [np.nan]
                continue
                
            traces = []
            limit = 20 # Limit to avoid performance bottleneck
            for candid in cand_words[:limit]:
                trace = get_ctc_probability_trace(logits, candid, processor)
                traces.append(trace)
            
            if traces:
                # Need strict alignment? All traces should be length T (same as logits).
                traces_arr = np.array(traces)
                mean_trace = np.mean(traces_arr, axis=0)
                categorized_data_dict[0, word, speaker, category] = mean_trace
            else:
                categorized_data_dict[0, word, speaker, category] = [np.nan]
        print(f"Transcription for {word}: {transcription} P(target):{max(categorized_data_dict[0, word, speaker, 'Target']):.4f}")
        
        # plot some examples for target, cohort, rhyme
        if random.random() < 0.1: # Plot 10% of examples
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10,6))
            for category in ["Target", "Cohort", "Rhyme"]:
                trace = categorized_data_dict[0, word, speaker, category]
                if trace is not None and not np.isnan(trace).all():
                    plt.plot(trace, label=category)
            plt.title(f"CTC Probability Traces for '{word}' (Speaker: {speaker})")
            plt.xlabel("Time Steps")
            plt.ylabel("Mean CTC Probability")
            plt.legend()
            plt.savefig(os.path.join(basepath, f"imgs/{word}_{speaker}_trace.png"))
            plt.close()
            
print("\n" + "="*80)
print(f"Processing completed. Accuracy: {cor}/{tot} = {cor/tot:.2%}")
print("Converting results and saving...")
if len(categorized_data_dict) == 0:
    print("No data collected. Exiting.")
    exit()
df = dict_to_dataframe(categorized_data_dict)
print(f"DataFrame shape: {df.shape}")
output_path = os.path.join(basepath, 'competition.csv')
df.to_csv(output_path, index=False)
print(f" Data saved to: {output_path}")

total_time = time.time() - start_time
print(f"⏱️  Total Processing Time: {total_time:.2f} seconds")
print("="*80)


# Processing facebook/wav2vec2-base-960h completed. Accuracy: 6324/10731 = 58.93%
