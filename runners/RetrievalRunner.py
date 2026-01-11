import copy
import json
import os
import pickle
import random
import re
import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from models.retriever import RetrieverModel
from prompts.pre_process import load_get_corpus_fn, load_get_query_fn

class Retriever:
    @staticmethod
    def parse_args(parser):
        parser.add_argument("--ret_type", default="dense", choices=['zero_shot', 'random', 'recency', 'bm25', 'dense', 'dense_tune'])
        parser.add_argument("--base_retriever_path", default="BAAI/bge-base-en-v1.5")
        parser.add_argument("--retriever_checkpoint", default="")
        parser.add_argument("--retriever_pooling", default="average")
        parser.add_argument("--retriever_normalize", type=int, default=1)
        parser.add_argument("--retrieve_user", type=int, default=1)
        parser.add_argument("--user_emb_path", required=True)
        parser.add_argument("--user_vocab_path", default="")
        parser.add_argument("--user_topk", type=int, default=4) # k=4
        return parser

    def __init__(self, opts) -> None:
        self.task = opts.task
        self.get_query = load_get_query_fn(self.task)
        self.get_corpus = load_get_corpus_fn(self.task)
        self.use_date = opts.source.endswith('date')
        self.llm_name = opts.llm_name
        self.data_addr = opts.data_addr
        self.output_addr = opts.output_addr
        self.data_split = opts.data_split # Dinamik Split (train/dev)
        self.source = opts.source
        self.ret_type = opts.ret_type
        self.topk = opts.topk # Document sayısı (n=3)
        self.retrieve_user = opts.retrieve_user
        self.device = opts.device
        
        # Parçalı kayıt için indisleri sakla
        self.begin_idx = opts.begin_idx
        self.end_idx = opts.end_idx
        
        self.load_user(opts)

        if self.ret_type in ['dense', 'dense_tune']:
            self.batch_size = opts.batch_size
            if self.ret_type == 'dense':
                self.retriever_checkpoint = opts.base_retriever_path
            else:
                self.retriever_checkpoint = os.path.join(opts.output_addr, f"train/{opts.source}", opts.retriever_checkpoint)
            
            self.retriever = RetrieverModel(
                ret_type=self.ret_type, model_path=self.retriever_checkpoint, base_model_path=opts.base_retriever_path,
                user2id=self.user2id, user_emb_path=self.user_emb_path, batch_size=self.batch_size, device=self.device,
                max_length=opts.max_length, pooling=opts.retriever_pooling, normalize=opts.retriever_normalize
            ).eval().to(self.device)

        input_path = os.path.join(self.data_addr, opts.data_split, self.source, 'rank_merge.json')
        self.dataset = json.load(open(input_path, 'r'))
        
        # end_idx None ise tüm veri setini al
        actual_end = self.end_idx if self.end_idx is not None else len(self.dataset)
        print(f"📄 Klasör: {self.data_split} | İşlenecek Aralık: {self.begin_idx}-{actual_end} | Toplam Veri: {len(self.dataset)}")
        
        self.dataset = self.dataset[self.begin_idx:actual_end]

    def load_user(self, opts):
        # Dinamik Klasör Yolu
        opts.user_vocab_path = os.path.join(opts.data_addr, f"{opts.data_split}/{opts.source}")
        
        with open(os.path.join(opts.user_vocab_path, 'user_vocab.pkl'), 'rb') as file:
            self.user_vocab = pickle.load(file)
        with open(os.path.join(opts.user_vocab_path, 'user2id.pkl'), 'rb') as file:
            self.user2id = pickle.load(file)
        
        # Dinamik Embedding Yolu
        opts.user_emb_path = os.path.join(opts.user_vocab_path, "user_emb", opts.user_emb_path)
        self.user_emb_path = opts.user_emb_path
        self.user_embedding = torch.load(self.user_emb_path).to(self.device)
        self.user_emb_name = '.'.join(os.path.basename(self.user_emb_path).split('.')[:-1])
        self.user_topk = opts.user_topk

    def run(self):
        file_name = "base"
        sub_dir = f"{self.ret_type}_{self.topk}"
        if self.ret_type == 'dense_tune':
            retriever_name = self.retriever_checkpoint.split('/')[-2]
            train_time = self.retriever_checkpoint.split('/')[-1]
            sub_dir = f"{retriever_name}_{self.topk}"
            file_name = f"{train_time}"
        
        if self.retrieve_user:
            file_name += f'_user-{self.user_topk}_{self.user_emb_name}'

        results = []
        for data in tqdm(self.dataset):
            # Input temizliği
            raw_input = str(data.get('input', ''))
            clean_input = raw_input
            match = re.search(r"review:\s*(.*)", raw_input, re.IGNORECASE)
            if match:
                clean_input = match.group(1).strip()
            
            query, selected_profs = self.retrieve_topk(clean_input, data['user_id'])
            results.append({
                "input": data['input'], "query": query, "output": data['output'],
                "user_id": data['user_id'], "retrieval": selected_profs
            })

        output_addr = os.path.join(self.output_addr, self.data_split, self.source, sub_dir, 'retrieval')
        if not os.path.exists(output_addr): os.makedirs(output_addr)
        
        # Dosya ismine parçalı indisleri ekle
        actual_end = self.end_idx if self.end_idx is not None else (self.begin_idx + len(self.dataset))
        result_path = os.path.join(output_addr, f"{file_name}_{self.begin_idx}-{actual_end}.json")
        
        with open(result_path, 'w') as file:
            json.dump(results, file, indent=4, ensure_ascii=False)
        print(f"✅ Kaydedildi: {result_path}")

    def retrieve_topk(self, inp, user):
        # 1, 2, 3. Adımların uygulandığı kontrol
        all_profiles, valid_user_key = self.retrieve_user_topk(user)
        
        if not all_profiles:
            return inp, []

        query = inp
        all_retrieved = []
        for i in range(len(all_profiles)):
            cur_corpus = self.get_corpus(all_profiles[i], self.use_date)
            # Alt modellere "doğrulanmış" anahtarı geçiyoruz
            cur_retrieved, cur_scores = self.retrieve_topk_one_user(cur_corpus, query, all_profiles[i], valid_user_key, self.topk)
            for data_idx, data in enumerate(cur_retrieved):
                cur_data = copy.deepcopy(data)
                if self.task.startswith('LaMP_3'): cur_data['rate'] = cur_data['score']
                cur_data['score'] = cur_scores[data_idx]
                all_retrieved.append(cur_data)
        return query, all_retrieved

    def retrieve_topk_one_user(self, corpus, query, profile, user, topk):
        if self.ret_type == "bm25":
            bm25 = BM25Okapi([x.split() for x in corpus])
            scores = bm25.get_scores(query.split())
            top_n = np.argsort(scores)[::-1][:topk]
            return [profile[i] for i in top_n], [scores[i] for i in top_n]
        elif self.ret_type in ["dense", "dense_tune"]:
            return self.retriever.retrieve_topk_dense(corpus, profile, str(query), user, topk)
        elif self.ret_type == "recency":
            profile = sorted(profile, key=lambda x: tuple(map(int, str(x['date']).split("-"))))
            return profile[::-1][:topk], [1.0] * topk
        return [], []

    def retrieve_user_topk(self, user):
        # --- GÜVENLİ VE SIKI ID KONTROLÜ (1, 2, 3. Adımlar) ---
        valid_key = None
        u_idx = self.user2id.get(user)
        if u_idx is not None: valid_key = user
        
        if valid_key is None:
            u_idx = self.user2id.get(str(user))
            if u_idx is not None: valid_key = str(user)
            
        if valid_key is None:
            try:
                u_idx = self.user2id.get(int(user))
                if u_idx is not None: valid_key = int(user)
            except: pass

        if valid_key is None:
            return [], None # Adım 2: Güvenli Çıkış
        
        if self.retrieve_user:
            cur_user_emb = self.user_embedding[[u_idx]]
            sims = F.cosine_similarity(cur_user_emb, self.user_embedding)
            _, topk_indices = torch.topk(sims, min(self.user_topk, len(sims)))
            top_k_ids = topk_indices.tolist()
            topk_scores = [sims[i].item() for i in top_k_ids]
        else:
            top_k_ids, topk_scores = [u_idx], [1.0]

        topk_profile = []
        for idx, user_idx in enumerate(top_k_ids):
            cur_prof = copy.deepcopy(self.user_vocab[user_idx]['profile'])
            for p in cur_prof: p['user_sim'] = topk_scores[idx]
            topk_profile.append(cur_prof)
        return topk_profile, valid_key