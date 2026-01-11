# CFRAG: Personalized Text Generation for Sephora (LaMP-3)

This repository implements the **Retrieval Augmented Generation with Collaborative Filtering (CFRAG)** framework for the Sephora product rating task (LaMP-3). The pipeline optimizes document selection by integrating user history with collaborative information from similar users and fine-tuning models based on Large Language Model (LLM) feedback.

## 🏗 Project Setup



---

## Stage 1: Data Preprocessing and User Dictionaries

Prepare chronological user profiles and establish user ID mappings.

```bash
# 1. Preprocess Training and Development Sets
python data/preprocess_profile.py --task LaMP_3_Sephora --data_phase train --ranker recency
python data/preprocess_profile.py --task LaMP_3_Sephora --data_phase dev --ranker recency

# 2. Generate User Vocabularies
python user_emb/get_user_set.py --task LaMP_3_Sephora --source recency --target_split train

```

---

## Stage 2: User Embedding Training

Train user embeddings using contrastive learning to capture semantic similarities between users based on their historical interactions.

```bash
# Generate Corpus Embeddings
python user_emb/get_corpus_emb.py --task LaMP_3_Sephora --stage dev --emb_model_path BAAI/bge-base-en-v1.5

# Train User Embeddings
cd user_emb/train_user_emb
python run.py --task LaMP_3_Sephora --num_train_epochs 5 --per_device_train_batch_size 64

```

---

## Stage 3: Retriever Tuning (Bi-Encoder / ROPG)

Optimize the BGE model to align its document retrieval preferences with the feedback provided by a teacher LLM (Qwen2).

1. **Point Scoring (Collect LLM Feedback):**
```bash
python generation/generate_point.py \
    --task LaMP_3_Sephora --model_name Qwen2-7B-Instruct-AWQ \
    --file_name base_0-11545 --batch_size 2048

```


2. **Fine-tune Retriever:**
```bash
python rank_tune/retriever/run.py \
    --learning_rate 1e-5 --num_train_epochs 2 --persona_weight 0.2 \
    --output_dir "checkpoints/LaMP_3_Sephora/retriever"

```



---

## Stage 4: Reranker Training (Cross-Encoder)

Train a Cross-Encoder for high-precision document ranking, focusing on the interaction between user queries and retrieved documents.

```bash
python rank_tune/reranker/run.py \
    --model_name_or_path BAAI/bge-reranker-base \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4

```

---

## Stage 5: Inference and Evaluation

Retrieve documents using the tuned models and calculate final performance metrics (MAE/MSE).

```bash
# 1. Execute Tuned Retrieval on Dev Set
python ranking.py --rank_stage retrieval --data_split dev --ret_type dense_tune --retriever_checkpoint [TIMESTAMP]

# 2. Final Generation and MAE Calculation
python generation/generate.py \
    --model_name Qwen2-7B-Instruct-AWQ --task LaMP_3_Sephora \
    --input_path "dev/recency/recency_3/" --source "retrieval"

```

## 🛠 Requirements

* **Core Libraries:** `transformers==4.42.3`, `torch==2.3.0`, `vllm==0.5.1`, `evaluate==0.4.2`.
* **Models:** Qwen2-7B-Instruct-AWQ (Generation), BAAI/bge-base-en-v1.5 (Retrieval), BM25.
