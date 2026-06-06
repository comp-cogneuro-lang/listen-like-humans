import torch
import torch.nn.functional as F
import pandas as pd
import joblib
import numpy as np
from tqdm import tqdm
import os,sys
# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_paths = [os.path.join(project_root, 'analysis'), os.path.join(project_root, 'src')]
for src_path in src_paths:
    if src_path not in sys.path:
        print(f"Adding {src_path} to sys.path")
        sys.path.append(src_path)

import time
import unicodedata
from data import read_config, CollateItems
from main import setup_training
from phonological_competition import Category_Dict_Generate, RT_Dict_Generate_item, dict_to_dataframe
from params import num_workers, is_cpu, lang_order
 
datapath = './dataset/'
train_set = False
n_epoch = 5690
name = "noncausal-2LSTM"
activation_fn = F.sigmoid if "SRV" in name else F.tanh
start_time = time.time()
# baseline
config_path = f'experiments/en_words_ku_2lstmbi.cfg'

basepath = f'{config_path.replace(".cfg","")}/{name}/pretraining'
pronunciation_dict = pd.read_csv("dataset/en/vocab.txt", sep="\t", names=["word","pronunciation"])
pronunciation_dict = pronunciation_dict[["word","pronunciation"]]
pronunciation_dict["pronunciation"] = pronunciation_dict["pronunciation"].apply(lambda x: x.split("."))
pronunciation_dict = pronunciation_dict.set_index("word").to_dict(orient="index")
words = list(pronunciation_dict.keys())
word_competition_dict = Category_Dict_Generate(
    words, 
    pronunciation_dict
)

# Setup network and dataset, load checkpoint
config = read_config(config_path, name)
trainer, train_dataset, valid_dataset, test_dataset = setup_training(
    config, datapath, num_workers, is_cpu
)
words = train_dataset.words
phonemes = train_dataset.phonemes
word_embedding = train_dataset.word_embedding_tensor
word_index_dict = {}
assert len(train_dataset.Sy_word) == len(lang_order), "Mismatch between number of words groups and languages"
idx = 0 
for lang_words, lang in zip(train_dataset.Sy_word, lang_order):
    for word in lang_words:
        word_index_dict[word] = idx
        idx += 1
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
if trainer.load_checkpoint(i_checkpoint=n_epoch):
    print(f"Checkpoint loaded successfully for epoch {n_epoch}.")
else:
    raise FileNotFoundError(f"Checkpoint for epoch {n_epoch} not found. Please check the checkpoint path and epoch number.")

trainer.model.to(device)
trainer.model.eval()
cor, tot = 0, 0
rt_cor, rt_tot = [],0
categorized_data_dict = {}
rt_dict = {}

batch_size = 256
if train_set:
    dataset = train_dataset
else:
    dataset = test_dataset
num_samples = len(dataset)
collect_fn = CollateItems()
idx = 0 
while idx < num_samples:
    st = time.time()
    batch_indices = list(range(idx, min(idx + batch_size, num_samples)))
    batch = [dataset[i] for i in batch_indices]
    batch = collect_fn(batch)
    inputs, targets, lengths, mask, loss_mask, lang_mask = trainer.generate_inputs_and_targets(
        batch, 0, 0, 1
    )
    # Move inputs to GPU
    inputs = [inp.to(device) if isinstance(inp, torch.Tensor) else inp for inp in inputs]
    langs = [dataset.lang_ind[i] for i in batch_indices]
    audio_paths = [dataset.wav_paths[i] for i in batch_indices]
    words = [audio_path.split('/')[-1].replace('.wav','') for audio_path in audio_paths]
    words = [unicodedata.normalize("NFC", word) for word in words]
    speakers = [audio_path.split('/')[-2] for audio_path in audio_paths]
 
    out = trainer.model(inputs, lengths, training=False)
    word_inters = activation_fn(out["word"])

    # Move word_embedding to GPU for similarity computation
    word_embedding_gpu = word_embedding.to(device)

    for word_inter, word, length, speaker, lang in zip(word_inters, words, lengths, speakers, langs):
        word_inter = word_inter[:length]  # [seq_len, semantic_size]
        tiled_result = word_inter.expand(word_embedding_gpu.size(0), -1, -1)  # [n_words, seq_len, semantic_size]
        tiled_target = word_embedding_gpu.unsqueeze(1).expand(-1, word_inter.size(0), -1)  # [n_words, seq_len, semantic_size]
        cs = torch.cosine_similarity(tiled_target, tiled_result, dim=2)  # [n_words, seq_len]
        cs_array = cs.detach().cpu().numpy()
        max_cycle = cs_array.shape[1]
        cs_array = np.concatenate(
            [cs_array, np.zeros((cs_array.shape[0],10))], axis=1
        )
        cs_array[:, -10:] = cs_array[:, [-11]] # TODO: check its effect
        rt = RT_Dict_Generate_item(
            0,
            word,
            cs_array,
            speaker,
            word_index_dict,
            None,
            None,
            max_cycle,
            absolute_Criterion=0.7,
            relative_Criterion=0.02,
            time_Dependency_Criterion=(10, 0.05)
        )
        rt_tot += 1
        if not np.isnan(rt[('Time_Dependent', 0, word, speaker)]):
            rt_cor.append(lang_order[lang])
        else:
            continue # skip if no RT detected
        # Skip if no RT detected
        related_indices = []
        for category in ["Target", "Cohort", "Rhyme"]:
            if len(word_competition_dict[word, category]) > 0:
                category_words = word_competition_dict[word, category]
                category_indices = [word_index_dict[w] for w in category_words] 
                related_indices.extend(category_indices)
                categorized_data_dict[0, word, speaker, category] = np.mean(cs_array[category_indices,:], axis=0)
            else:
                categorized_data_dict[0, word, speaker, category] = np.zeros((cs_array.shape[1])) * np.nan
        related_indices = list(set(related_indices))
        
        categorized_data_dict[0, word, speaker, "Unrelated"] = np.mean(np.delete(cs_array, related_indices, 0), axis=0)
        
        # case study activations (relative to repo root)
        case_dir = "experiments/en_words_ku_2lstmbi/noncausal-2LSTM/case_studies"
        os.makedirs(case_dir, exist_ok=True)
        np.savez(f"{case_dir}/{word}_{speaker}_activations.npz", cs_array=cs_array, related_indices=related_indices)
      
    rt_accuracy_pct = (len(rt_cor)/rt_tot*100) if rt_tot > 0 else 0
    batch_time = time.time() - st
    print(f'  ├─ RT Accuracy: {len(rt_cor):5d}/{rt_tot:5d} ({rt_accuracy_pct:6.2f}%) | Batch Time: {batch_time:6.2f}s')
    idx += batch_size


print("\n" + "="*80)
print("                         FINAL RESULTS SUMMARY")
print("="*80)
word_accuracy_pct = (cor/tot*100) if tot > 0 else 0
rt_accuracy_pct = (len(rt_cor)/rt_tot*100) if rt_tot > 0 else 0
rt_accuracy_en = (len([rt for rt in rt_cor if rt == 'en'])/(rt_tot//2)*100) if rt_tot > 0 else 0
rt_accuracy_es = (len([rt for rt in rt_cor if rt == 'es'])/(rt_tot//2)*100) if rt_tot > 0 else 0

print(f"✓ Word Accuracy:        {cor:6d}/{tot:6d} ({word_accuracy_pct:6.2f}%)")
print(f"✓ RT Accuracy:          {len(rt_cor):6d}/{rt_tot:6d} ({rt_accuracy_pct:6.2f}%)")
print(f"  ├─ EN RT Accuracy:          {len([rt for rt in rt_cor if rt == 'en']):6d}/{rt_tot//2:6d} ({rt_accuracy_en:6.2f}%)")
print(f"  ├─ ES RT Accuracy:          {len([rt for rt in rt_cor if rt == 'es']):6d}/{rt_tot//2:6d} ({rt_accuracy_es:6.2f}%)")
print(f"✓ Total Samples (RT):   {rt_tot:6d}")
print("="*80)

print("\n📊 Converting data to DataFrame and saving...")
df = dict_to_dataframe(categorized_data_dict)
print(f"   └─ DataFrame shape: {df.shape}")
print(f"   └─ Columns: {list(df.columns)[:5]}..." if len(df.columns) > 5 else f"   └─ Columns: {list(df.columns)}")
print(f"\n📄 Sample output (first 5 rows):")
print(df.head().to_string())
df.to_csv(os.path.join(basepath, 'competition.csv'), index=False)
print(f"\n Data saved to: competition.csv")

total_time = time.time() - start_time
print("\n" + "="*80)
print(f"⏱️  Total Processing Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
print("="*80)


# 22LSTM-512-ENES-WORD-KUW2V-SpecF35-LangLSTM-SharedHead
# ✓ RT Accuracy:            2088/  3066 ( 68.10%)
#   ├─ EN RT Accuracy:            1129/  1533 ( 73.65%)
#   ├─ ES RT Accuracy:             959/  1533 ( 62.56%)
# ✓ Total Samples (RT):     3066

# ✓ Word Accuracy:             0/     0 (  0.00%)
# ✓ RT Accuracy:            2096/  3066 ( 68.36%)
#   ├─ EN RT Accuracy:            1163/  1533 ( 75.86%)
#   ├─ ES RT Accuracy:             933/  1533 ( 60.86%)
# ✓ Total Samples (RT):     3066