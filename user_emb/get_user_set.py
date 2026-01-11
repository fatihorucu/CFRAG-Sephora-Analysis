import sys
sys.path.append(".")
import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from prompts.pre_process import load_get_corpus_fn

parser = argparse.ArgumentParser()
parser.add_argument("--task", default='LaMP_2_time')
parser.add_argument("--source", default='recency')
parser.add_argument("--target_split", default='dev', help="dev/test") 

if __name__ == "__main__":
    opts = parser.parse_args()
    print(f"Task: {opts.task} | Target Split: {opts.target_split}")
    
    get_corpus_fn = load_get_corpus_fn(opts.task)
    use_date = opts.source.endswith('date')
    opts.input_path = os.path.join("data", opts.task)
    
    # Hedef klasör
    split_file = os.path.join(opts.input_path, f'{opts.target_split}/{opts.target_split}_questions.json')
    if not os.path.exists(split_file):
        print(f"❌ HATA: {split_file} yok!")
        exit(1)
        
    questions = json.load(open(split_file, 'r'))
    
    # Kullanıcı Listesi ve Corpus
    user_id_list = []
    corpus_list = []
    print("⏳ Taranıyor...")
    for item in tqdm(questions):
        user_id_list.append(item['user_id'])
        profile = sorted(item['profile'], key=lambda x: str(x.get('date', '2000-01-01')))
        corpus = get_corpus_fn(profile, use_date=use_date)
        corpus_list.extend(corpus)

    user_df = pd.DataFrame({"user_id": list(set(user_id_list))})
    user_df['id'] = np.arange(len(user_df))
    corpus_df = pd.DataFrame({"corpus": ['<pad>'] + list(set(corpus_list)) + ['<mask>', '']})
    corpus_df['id'] = np.arange(len(corpus_df))

    user_vocab = user_df.set_index('id', drop=False).to_dict('index')
    user2id = {u['user_id']: i for i, u in user_vocab.items()}
    corpus_vocab = corpus_df.set_index('id', drop=False).to_dict('index')
    corpus2id = {c['corpus']: i for i, c in corpus_vocab.items()}

    for idx in user_vocab:
        user_vocab[idx]['profile'] = []
        user_vocab[idx]['corpus_ids'] = []

    print("⏳ Profiller Eşleniyor...")
    for item in tqdm(questions):
        user_id = item['user_id']
        profile = sorted(item['profile'], key=lambda x: str(x.get('date', '2000-01-01')))
        corpus = get_corpus_fn(profile, use_date=use_date)
        corpus_ids = [corpus2id.get(x, 0) for x in corpus]
        uid_idx = user2id[user_id]
        if not user_vocab[uid_idx]['corpus_ids']:
            user_vocab[uid_idx]['profile'] = profile
            user_vocab[uid_idx]['corpus_ids'] = corpus_ids

    output_dir = os.path.join(opts.input_path, f"{opts.target_split}/{opts.source}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "user_vocab.pkl"), "wb") as f: pickle.dump(user_vocab, f)
    with open(os.path.join(output_dir, "user2id.pkl"), "wb") as f: pickle.dump(user2id, f)
    with open(os.path.join(output_dir, "corpus_vocab.pkl"), "wb") as f: pickle.dump(corpus_vocab, f)
    with open(os.path.join(output_dir, "corpus2id.pkl"), "wb") as f: pickle.dump(corpus2id, f)
    print(f"✅ Hazır: {output_dir}")
