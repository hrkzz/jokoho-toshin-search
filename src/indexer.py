import json
import re
import random
from pathlib import Path
import os
from tqdm import tqdm

# LangChainと関連ライブラリをインポート
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- 設定項目 ---
CLEANED_DATA_PATH = Path("outputs/cleaned_toshin_data.json")
VECTOR_STORE_DIR = Path("vector_store")
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# ME5-largeのトークン上限(512)を考慮した閾値
TOKEN_THRESHOLD = 500 
# 長いテキストを分割する際のチャンクサイズ
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def build_vector_store():
    """
    クリーニング済みJSONからチャンクを作成し、ベクトルストアを構築・保存する
    """
    if not CLEANED_DATA_PATH.exists():
        print(f"エラー: クリーニング済みファイルが見つかりません: {CLEANED_DATA_PATH}")
        return

    # 1. データの読み込み
    print(f"'{CLEANED_DATA_PATH}' からデータを読み込んでいます...")
    with open(CLEANED_DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 2. チャンク分割の準備
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_docs = []
    print(f"{len(data)}件の文書を処理します...")

    for record in tqdm(data, desc="Processing documents"):
        base_metadata = {
            "source_url": record.get("URL", ""),
            "case_name": record.get("事件名", ""),
            "agency": record.get("諮問庁", ""),
            "report_date": record.get("答申日_iso", "")
        }

        # --- ハイブリッド戦略の実装 ---
        texts_to_process = {
            "summary": record.get("summary_text", "")
        }
        # detail_textsの内容も個別に追加
        for section, text in record.get("detail_texts", {}).items():
            texts_to_process[section] = text

        for text_type, text in texts_to_process.items():
            if not text:
                continue

            # ME5-largeのモデルはトークン数を直接数えるのが少し複雑なため、
            # 文字数で簡易的に判定します。500文字 ≒ 500トークン以下と仮定。
            if len(text) <= CHUNK_SIZE:
                # 閾値以下なら、そのまま1チャンクとして追加
                metadata = base_metadata.copy()
                metadata["type"] = text_type
                all_docs.append(Document(page_content=text, metadata=metadata))
            else:
                # 閾値を超えたら、分割して追加
                chunks = text_splitter.split_text(text)
                for chunk_content in chunks:
                    metadata = base_metadata.copy()
                    metadata["type"] = text_type
                    all_docs.append(Document(page_content=chunk_content, metadata=metadata))
    
    print(f"合計 {len(all_docs)} 個のチャンクが作成されました。")

    # 3. 埋め込みモデルの準備
    print(f"埋め込みモデル '{EMBEDDING_MODEL_NAME}' をロードしています...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    # 4. ベクトルストアの構築と保存
    print(f"ベクトルストアを初期化しています @ '{VECTOR_STORE_DIR}'...")
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(VECTOR_STORE_DIR)
    )

    # バッチ処理で少しずつドキュメントを追加し、進捗を表示
    batch_size = 32 # 一度に処理するチャンク数
    for i in tqdm(range(0, len(all_docs), batch_size), desc="Embedding and Storing Chunks"):
        batch = all_docs[i:i + batch_size]
        vector_store.add_documents(documents=batch)

if __name__ == '__main__':
    build_vector_store()