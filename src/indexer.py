import json
from pathlib import Path
import os
import gc
import time
from datetime import datetime, timedelta
import torch
from typing import List
import multiprocessing
from tqdm import tqdm
import shutil

# LangChainと関連ライブラリをインポート
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
from langchain_chroma import Chroma

# --- 設定項目 ---
CLEANED_DATA_PATH = Path("outputs/cleaned_toshin_data.json")
VECTOR_STORE_DIR = Path("vector_store")

# モデル選択（コマンドライン引数で上書きされるグローバル変数）
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# チャンク設定
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- ONNX高速化のためのヘルパークラス ---
class ONNXEmbeddings:
    """ONNXモデルをLangChainで使えるようにするためのラッパークラス"""
    def __init__(self, model, tokenizer, device="cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.cpu().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# --- 並列処理のためのワーカー関数（進捗可視化対応） ---
def embed_batch_worker_with_progress(args):
    """
    サブプロセスで実行されるワーカー関数。
    個別の進捗バーを表示する機能を追加。
    """
    worker_id, texts = args  # 引数をアンパック
    
    try:
        model = ORTModelForFeatureExtraction.from_pretrained(
            EMBEDDING_MODEL_NAME,
            file_name="model_qint8_avx512_vnni.onnx"
        )
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        embeddings_model = ONNXEmbeddings(model=model, tokenizer=tokenizer)
        
        pbar = tqdm(total=len(texts),
                    desc=f"  Worker-{worker_id:02d}",
                    position=worker_id + 1,
                    leave=False)
        
        results = []
        inner_batch_size = 64
        for i in range(0, len(texts), inner_batch_size):
            batch = texts[i:i + inner_batch_size]
            results.extend(embeddings_model.embed_documents(batch))
            pbar.update(len(batch))
            
        pbar.close()
        return results
    except Exception as e:
        if 'pbar' in locals() and pbar:
            pbar.close()
        # エラーが発生したプロセスIDを明確に表示
        print(f"\n❌ Error in Worker-{worker_id}: {e}")
        # 他のプロセスに影響を与えないよう、空のリストを返す
        return []


# --- メインの処理関数 ---
def build_vector_store():
    """並列処理で高速化したベクトルストア構築（進捗可視化・最終確定版）"""
    # 1. データ読み込み
    if not CLEANED_DATA_PATH.exists():
        print(f"❌ エラー: クリーニング済みファイルが見つかりません: {CLEANED_DATA_PATH}")
        return
    print(f"📂 '{CLEANED_DATA_PATH}' からデータを読み込んでいます...")
    with open(CLEANED_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # テスト用
    # data = data[:3]
    
    total_docs = len(data)
    print(f"  ➡️ {total_docs:,}件の文書を検出")

    # 2. ONNXモデルの事前準備
    print("\n  🔍 ONNXモデルのキャッシュを確認・準備しています...")
    try:
        model = ORTModelForFeatureExtraction.from_pretrained(
            EMBEDDING_MODEL_NAME,
            file_name="model_qint8_avx512_vnni.onnx"
        )
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    except Exception:
        print("    キャッシュが見つからないため、ONNXモデルの変換を一度だけ行います...")
        model = ORTModelForFeatureExtraction.from_pretrained(EMBEDDING_MODEL_NAME, export=True)
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        try:
            model_id = EMBEDDING_MODEL_NAME.replace('/', '--')
            cache_dir = Path.home() / ".cache/huggingface/hub" / f"models--{model_id}"
            snapshot_path = next(cache_dir.glob("snapshots/*"))
            onnx_dir = snapshot_path / "onnx"
            if (onnx_dir / "model.onnx").exists(): (onnx_dir / "model.onnx").unlink()
            if (onnx_dir / "model_O4.onnx").exists(): (onnx_dir / "model_O4.onnx").unlink()
            print("    ✅ 不要なONNXファイルを自動でクリーンアップしました。")
        except Exception as e:
            print(f"    ⚠️ 自動クリーンアップ中に軽微なエラーが発生しました: {e}")
    
    embeddings_function = ONNXEmbeddings(model=model, tokenizer=tokenizer)
    print("  ✅ モデルの準備が完了しました。")

    # 3. ベクトルストアの初期化
    print(f"\n💾 ベクトルストアを初期化しています @ '{VECTOR_STORE_DIR}'...")
    if VECTOR_STORE_DIR.exists():
        response = input("  既存のベクトルストアが見つかりました。上書きしますか？ (y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(VECTOR_STORE_DIR)
            print("  ✅ 既存のストアを削除しました。")
    vector_store = Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=embeddings_function
    )

    # 4. チャンク分割の準備
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "、", " ", ""]
    )

    # 5. 並列プロセス数の決定
    print(f"\n🔄 並列処理を開始します...")
    if "base" in EMBEDDING_MODEL_NAME or "large" in EMBEDDING_MODEL_NAME:
        num_processes = 4
        print(f"  🧠 base/largeモデルのため、メモリを考慮してプロセス数を {num_processes} に制限します。")
    else:
        num_processes = max(1, os.cpu_count() - 2)
        print(f"  ⚡ smallモデルのため、CPU性能を最大限に活用するプロセス数 {num_processes} を設定します。")
    print(f"  ⚙️ 設定: モデル={EMBEDDING_MODEL_NAME.split('/')[-1]}, 並列プロセス数={num_processes}")
    
    start_time = time.time()
    
    # 6. 全ドキュメントからチャンクを一括作成
    print("\n  📦 全文書のチャンクを作成中...")
    all_docs_to_process = []
    for record in tqdm(data, desc="    Creating all chunks"):
        base_metadata = {
            "source_url": record.get("URL", ""), "case_name": record.get("事件名", ""),
            "agency": record.get("諮問庁", ""), "report_date": record.get("答申日_iso", "")
        }
        all_texts = []
        summary = record.get("summary_text", "")
        if summary and summary.strip(): all_texts.append(("summary", summary))
        for section, text in record.get("detail_texts", {}).items():
            if text and text.strip(): all_texts.append((section, text))
        
        for text_type, text in all_texts:
            chunks = text_splitter.split_text(text)
            for i, chunk_content in enumerate(chunks):
                metadata = base_metadata.copy()
                metadata["type"] = text_type
                metadata["chunk_index"] = i
                all_docs_to_process.append(Document(page_content=chunk_content, metadata=metadata))

    print(f"  ➡️ 作成された総チャンク数: {len(all_docs_to_process):,}")

    # 7. 並列処理でベクトル化
    print(f"\n  ⚙️  {num_processes}個のプロセスで並列ベクトル化を実行中...")
    texts_to_embed = [doc.page_content for doc in all_docs_to_process]
    
    num_texts = len(texts_to_embed)
    # 実際に起動するプロセス数を、タスクがある数に限定する
    actual_num_processes = min(num_processes, num_texts)
    
    tasks = []
    if actual_num_processes > 0:
        chunk_size_per_process = (num_texts + actual_num_processes - 1) // actual_num_processes
        for i in range(actual_num_processes):
            start = i * chunk_size_per_process
            end = min((i + 1) * chunk_size_per_process, num_texts)
            tasks.append((i, texts_to_embed[start:end]))

    embeddings = []
    if tasks: # タスクがある場合のみ並列処理を実行
        with tqdm(total=num_texts, desc="Overall Progress     ", position=0) as overall_pbar:
            with multiprocessing.Pool(processes=actual_num_processes) as pool:
                for result_batch in pool.imap_unordered(embed_batch_worker_with_progress, tasks):
                    embeddings.extend(result_batch)
                    overall_pbar.update(len(result_batch))
    
    print("\n") # プログレスバーの表示をクリアにするための改行

# 8. ベクトルストアにバッチで分割して追加
    print("\n\n  💾 計算済みのベクトルをデータベースに分割して追加中...")
    if texts_to_embed:
        db_batch_size = 2048  # 一度にDBへ書き込むチャンク数
        
        for i in tqdm(range(0, len(texts_to_embed), db_batch_size), desc="    Adding to DB"):
            start_index = i
            end_index = min(i + db_batch_size, len(texts_to_embed))
            
            vector_store.add_texts(
                texts=texts_to_embed[start_index:end_index],
                metadatas=[doc.metadata for doc in all_docs_to_process[start_index:end_index]],
                embeddings=[e for e in embeddings[start_index:end_index]]
            )
        print("  ✅ 完了！（データは自動的に永続化されます）")
    else:
        print("  ℹ️ 追加するデータがありませんでした。")
        
    # 9. 完了報告
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"🎉 ベクトルストアの構築が完了しました！")
    print(f"{'='*60}")
    print(f"📊 最終統計:")
    print(f"  - 処理した文書数: {total_docs:,}")
    print(f"  - 作成したチャンク数: {len(all_docs_to_process):,}")
    print(f"  - 総処理時間: {str(timedelta(seconds=int(total_time)))}")
    if total_time > 0 and len(all_docs_to_process) > 0:
        avg_speed = len(all_docs_to_process) / total_time
        print(f"  - 平均速度: {avg_speed:.1f} chunks/sec")
    print(f"  - 保存先: {VECTOR_STORE_DIR}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    import sys
    
    model_map = {
        "small": "intfloat/multilingual-e5-small",
        "base": "intfloat/multilingual-e5-base",
        "large": "intfloat/multilingual-e5-large"
    }
    
    selected_model = EMBEDDING_MODEL_NAME
    
    if len(sys.argv) > 1:
        model_arg = sys.argv[1]
        if model_arg in model_map:
            selected_model = model_map[model_arg]
            print(f"🎯 {model_arg}モデルを引数として選択しました。")
        else:
            print(f"⚠️ 未知の引数 '{model_arg}' です。デフォルトのモデルを使用します。")
            print(f"  使用可能な引数: {', '.join(model_map.keys())}")
    
    EMBEDDING_MODEL_NAME = selected_model
    print(f"    使用モデル: {EMBEDDING_MODEL_NAME}")
    
    build_vector_store()