import streamlit as st
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import LlamaCpp  # LlamaCppをインポート
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 定数設定 ---
VECTOR_STORE_DIR = Path("vector_store")
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

# --- ローカルのGGUFモデルへのパスを指定 ---
# ユーザーの環境に合わせてパスを修正してください
# WSL内のパスを指定します
LOCAL_MODEL_PATH = "/home/hrkzz/local_models/models/llm-jp-3.1-1.8b-instruct4-Q4_K_M.gguf"

# --- Streamlitのキャッシュ機能を使って、モデルとDBを効率的にロード ---

@st.cache_resource
def load_embedding_model():
    """埋め込みモデルをロードする"""
    print("Loading embedding model...")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def load_vector_store(_embeddings):
    """既存のベクトルストアをロードする"""
    if not VECTOR_STORE_DIR.exists():
        st.error(f"ベクトルストアが見つかりません: {VECTOR_STORE_DIR}")
        st.stop()
    print("Loading vector store...")
    return Chroma(
        persist_directory=str(VECTOR_STORE_DIR),
        embedding_function=_embeddings
    )

@st.cache_resource
def load_llm():
    """ローカルのGGUFモデルをLlamaCppでロードする"""
    print(f"Loading local LLM from: {LOCAL_MODEL_PATH}")
    
    if not Path(LOCAL_MODEL_PATH).exists():
        st.error(f"LLMモデルファイルが見つかりません: {LOCAL_MODEL_PATH}")
        st.error("app.py内の LOCAL_MODEL_PATH を正しいパスに修正してください。")
        st.stop()

    return LlamaCpp(
        model_path=LOCAL_MODEL_PATH,
        n_gpu_layers=0,  # GPUを使わない場合は0
        n_batch=512,     # 一度に処理するトークン数
        n_ctx=2048,      # モデルが扱えるコンテキストの最大長
        verbose=False,   # ログ出力を抑制
    )

# --- メインのアプリケーション部分 ---

# 1. モデルとDBのロード
embeddings = load_embedding_model()
vector_store = load_vector_store(embeddings)
llm = load_llm()

# Retriever（検索機）の作成
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 2. RAGプロンプトテンプレートの定義
template = """
### 指示:
提供されたコンテキスト情報のみを使用して、質問に日本語で回答してください。
コンテキストに答えがない場合は、「情報が見つかりませんでした。」と回答してください。

### コンテキスト:
{context}

### 質問:
{question}

### 回答:
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. RAGチェーンの構築
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Streamlit UIの構築
st.title("答申情報検索 RAGシステム")
st.write("総務省の答申データベースに関する質問を入力してください。")

question = st.text_input("質問を入力してください:", placeholder="パワハラによる退職に関する事例はありますか？")

if st.button("質問する"):
    if question:
        with st.spinner("回答を生成中です..."):
            answer = rag_chain.invoke(question)
            st.markdown("### 回答")
            st.write(answer)
            
            st.markdown("---")
            st.markdown("### 回答の根拠となった情報")
            retrieved_docs = retriever.invoke(question)
            for i, doc in enumerate(retrieved_docs):
                with st.expander(f"参考情報 {i+1}: {doc.metadata.get('case_name', 'N/A')} ({doc.metadata.get('report_date', 'N/A')})"):
                    st.text(doc.page_content)
                    st.caption(f"Source: {doc.metadata.get('source_url', 'N/A')}")
    else:
        st.warning("質問を入力してください。")
