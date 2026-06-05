import sys
import os
import argparse
import time
import unicodedata
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Ensure src is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    print(f"Adding {src_path} to sys.path")
    sys.path.append(src_path)

try:
    from params import (
        num_workers, is_cpu, lang_order
    )
    from data import read_config, CollateItems
    from main import setup_training
    from phonological_competition import Category_Dict_Generate, RT_Dict_Generate_item, dict_to_dataframe
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"sys.path: {sys.path}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Run competition analysis")
    parser.add_argument("--config_path", type=str, required=True, help="Path to config file")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--epochs", type=int, nargs='+', required=True, help="List of epochs to process")
    parser.add_argument("--datapath", type=str, default="./dataset/", help="Path to dataset")
    parser.add_argument("--train_set", action="store_true", help="Use training set instead of test set")
    parser.add_argument("--causal_transformer", action="store_true", default=False, help="Use causal transformer inference")
    parser.add_argument("--output_base_dir", type=str, default=None, help="Base directory to save results")
    return parser.parse_args()

def process_epoch(args, n_epoch, trainer, dataset, word_competition_dict, word_index_dict, device, activation_fn, train_dataset_ref):
    print(f"\nProcessing Epoch {n_epoch}...")
    try:
        trainer.load_checkpoint(i_checkpoint=n_epoch)
    except Exception as e:
        print(f"Failed to load checkpoint {n_epoch}: {e}")
        return {}

    trainer.model.to(device)
    trainer.model.eval()
    
    rt_cor, rt_tot = [], 0
    categorized_data_dict = {}
    
    batch_size = 1
    num_samples = len(dataset)
    collect_fn = CollateItems()
    idx = 0 
 
    # Progress bar
    pbar = tqdm(total=num_samples, desc=f"Epoch {n_epoch}", unit="sample")
    
    # Use word embeddings from training set reference as in original script
    word_embedding_tensor = train_dataset_ref.word_embedding_tensor
    
    while idx < num_samples:
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
        words_batch = [audio_path.split('/')[-1].replace('.wav','') for audio_path in audio_paths]
        words_batch = [unicodedata.normalize("NFC", word) for word in words_batch]
        speakers = [audio_path.split('/')[-2] for audio_path in audio_paths]

        if args.causal_transformer:
            out = trainer.model(inputs, lengths, training=False, causal_inference=True)
        else:
            out = trainer.model(inputs, lengths, training=False)
            
        word_inters = activation_fn(out["word"])

        # Move word_embedding to GPU for similarity computation
        word_embedding_gpu = word_embedding_tensor.to(device)

        for word_inter, word, length, speaker, lang in zip(word_inters, words_batch, lengths, speakers, langs):
            word_inter = word_inter[:length]  # [seq_len, semantic_size]
            tiled_result = word_inter.expand(word_embedding_gpu.size(0), -1, -1)  # [n_words, seq_len, semantic_size]
            tiled_target = word_embedding_gpu.unsqueeze(1).expand(-1, word_inter.size(0), -1)  # [n_words, seq_len, semantic_size]
            cs = torch.cosine_similarity(tiled_target, tiled_result, dim=2)  # [n_words, seq_len]
            cs_array = cs.detach().cpu().numpy()
            max_cycle = cs_array.shape[1]
            cs_array = np.concatenate(
                [cs_array, np.zeros((cs_array.shape[0],10))], axis=1
            )
            cs_array[:, -10:] = cs_array[:, [-11]] 
            
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
            
            # Only consider RTs that are not NaN for accuracy calculation and category assignment
            if not np.isnan(rt[('Time_Dependent', 0, word, speaker)]):
                rt_cor.append(lang_order[lang])
            
                # Categories
                related_indices = []
                for category in ["Target", "Cohort", "Rhyme", "Anadrome"]:
                    if len(word_competition_dict[word, category]) > 0:
                        category_words = word_competition_dict[word, category]
                        # Filter words that are in word_index_dict
                        valid_category_words = [w for w in category_words if w in word_index_dict]
                        if valid_category_words:
                            category_indices = [word_index_dict[w] for w in valid_category_words] 
                            related_indices.extend(category_indices)
                            categorized_data_dict[n_epoch, word, speaker, category] = np.mean(cs_array[category_indices,:], axis=0)
                        else:
                            categorized_data_dict[n_epoch, word, speaker, category] = np.zeros((cs_array.shape[1])) * np.nan
                    else:
                        categorized_data_dict[n_epoch, word, speaker, category] = np.zeros((cs_array.shape[1])) * np.nan
                related_indices = list(set(related_indices))
                
                if related_indices:
                    categorized_data_dict[n_epoch, word, speaker, "Unrelated"] = np.mean(np.delete(cs_array, related_indices, 0), axis=0)
                else:
                    categorized_data_dict[n_epoch, word, speaker, "Unrelated"] = np.mean(cs_array, axis=0)

        idx += batch_size
        pbar.update(batch_size)
    pbar.close()
    
    # Calculate stats
    rt_accuracy_pct = (len(rt_cor)/rt_tot*100) if rt_tot > 0 else 0 
    print("\n" + "="*80)
    print(f"RESULTS SUMMARY - Epoch {n_epoch}")
    print("="*80)
    print(f"✓ RT Accuracy:          {len(rt_cor):6d}/{rt_tot:6d} ({rt_accuracy_pct:6.2f}%)")
    print("="*80)
    
    return categorized_data_dict


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    
    # activation_fn
    activation_fn = F.sigmoid if "SRV" in args.name else F.tanh
 
    vocab_path = os.path.join(args.datapath, "en/vocab.txt")
    if not os.path.exists(vocab_path):
        vocab_path = "dataset/en/vocab.txt" # Fallback

    print(f"Loading vocab from {vocab_path}")
    if os.path.exists(vocab_path):
        pronunciation_dict = pd.read_csv(vocab_path, sep="\t", names=["word","pronunciation"])
        pronunciation_dict = pronunciation_dict[["word","pronunciation"]]
        pronunciation_dict["pronunciation"] = pronunciation_dict["pronunciation"].apply(lambda x: x.split("."))
        pronunciation_dict = pronunciation_dict.set_index("word").to_dict(orient="index")
        vocab_words = list(pronunciation_dict.keys())
    else:
        print(f"Warning: Vocab file {vocab_path} not found.")
        sys.exit(1)

    word_competition_dict = Category_Dict_Generate(
        vocab_words, 
        pronunciation_dict
    )

    # Setup network and dataset
    config = read_config(args.config_path, args.name)
    trainer, train_dataset, valid_dataset, test_dataset = setup_training(
        config, args.datapath, num_workers, is_cpu
    )
    
    dataset = train_dataset if args.train_set else test_dataset
    
    # Build word index dict
    word_index_dict = {}
    assert len(train_dataset.Sy_word) == len(lang_order), "Mismatch between number of words groups and languages"
    idx = 0 
    for lang_words, lang in zip(train_dataset.Sy_word, lang_order):
        for word in lang_words:
            word_index_dict[word] = idx
            idx += 1
            
    start_total_time = time.time()
    
    all_results = {}
    
    for n_epoch in args.epochs:
        epoch_results = process_epoch(args, n_epoch, trainer, dataset, word_competition_dict, word_index_dict, device, activation_fn, train_dataset)
        all_results.update(epoch_results)
        
    total_time = time.time() - start_total_time
    print("\n" + "="*80)
    print(f"⏱️  Total Processing Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*80)

    # Save aggregated results
    if args.output_base_dir:
        basepath = args.output_base_dir
    else:
        # Default: experiments/config_name/run_name/training
        config_dir_name = os.path.splitext(args.config_path)[0]
        basepath = os.path.join(config_dir_name, args.name, 'training')

    os.makedirs(basepath, exist_ok=True)
    
    if all_results:
        print(f"\n📊 Converting aggregated data to DataFrame and saving...")
        df = dict_to_dataframe(all_results)
        print(f"   └─ DataFrame shape: {df.shape}")
        
        output_path = os.path.join(basepath, 'competition.csv')
        df.to_csv(output_path, index=False)
        print(f" All data saved to: {output_path}")
    else:
        print("Warning: No results to save.")

if __name__ == "__main__":
    main()
