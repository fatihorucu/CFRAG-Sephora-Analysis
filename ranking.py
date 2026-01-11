import argparse
import os
import sys

# Yolları ekle
sys.path.append(os.getcwd())

from runners.ReRankRunner import ReRanker
from runners.RetrievalRunner import Retriever

def parse_global_args(parser: argparse.ArgumentParser):
    parser.add_argument("--CUDA_VISIBLE_DEVICES", default='0')
    parser.add_argument("--device", default='cuda:0')
    parser.add_argument("--llm_name", default="Meta-Llama-3-8B-Instruct")
    parser.add_argument("--data_addr", default='data/')
    parser.add_argument("--output_addr", default='')
    parser.add_argument("--data_split", default='test') # Varsayılan test
    parser.add_argument("--source", default='recency')
    parser.add_argument("--task", default="LaMP_3_Sephora")
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=512)
    return parser

if __name__ == "__main__":
    init_parser = argparse.ArgumentParser()
    init_parser.add_argument("--rank_stage", default='retrieval', choices=['retrieval', 'rerank'])
    init_args, _ = init_parser.parse_known_args()
    
    parser = argparse.ArgumentParser()
    parser = parse_global_args(parser)
    
    # Argümanları yükle
    if init_args.rank_stage == 'retrieval':
        parser = Retriever.parse_args(parser)
    elif init_args.rank_stage == 'rerank':
        parser = ReRanker.parse_args(parser)

    opts, _ = parser.parse_known_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.CUDA_VISIBLE_DEVICES
    
    # --- DEBUG: YOL AYARLAMALARI ---
    # Kodun veriyi nerede aradığını tam kontrol edelim
    
    # 1. Output Yolu
    opts.output_addr = f"{opts.llm_name}_outputs/"
    opts.output_addr = os.path.join(opts.output_addr, opts.task)
    
    # 2. Input Yolu (Kritik Nokta)
    # Eğer data_addr zaten task ismini içeriyorsa tekrar ekleme
    if opts.task not in opts.data_addr:
        opts.data_addr = os.path.join(opts.data_addr, opts.task)
    
    print("\n" + "="*60)
    print("🐞 DEBUG MODU BAŞLATILDI")
    print(f"📂 Veri Klasörü (data_addr): {opts.data_addr}")
    print(f"📂 Hedef Klasör (output_addr): {opts.output_addr}")
    print(f"📂 Split: {opts.data_split}")
    print("="*60 + "\n")
    
    # Klasörlerin varlığını kontrol et
    expected_input = os.path.join(opts.data_addr, opts.data_split, opts.source, 'rank_merge.json')
    if os.path.exists(expected_input):
        print(f"✅ GİRDİ DOSYASI BULUNDU: {expected_input}")
    else:
        print(f"❌ HATA: Girdi dosyası bulunamadı!")
        print(f"   Aranan Yol: {expected_input}")
        print("   Lütfen 'preprocess' adımını kontrol edin veya 'test' klasörüne veri taşıyın.")

    # Çıktı klasörünü zorla oluştur
    os.makedirs(opts.output_addr, exist_ok=True)
    print(f"✅ Çıktı ana klasörü oluşturuldu/doğrulandı: {opts.output_addr}\n")

    if init_args.rank_stage == 'retrieval':
        retriever = Retriever(opts)
        retriever.run()
    elif init_args.rank_stage == 'rerank':
        reranker = ReRanker(opts)
        reranker.run()
