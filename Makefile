# --- 変数定義 ---
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
PIP = $(VENV_DIR)/bin/pip
STREAMLIT = $(VENV_DIR)/bin/streamlit

# --- ターゲット定義 ---
# .PHONY: ターゲットがファイル名と被ってもコマンドとして実行されるようにするおまじない
.PHONY: setup activate pipeline scrape clean index run clean-all

# --- 注意: 以下のコマンド行は必ずスペースではなくタブ(Tab)でインデントしてください ---

# 環境構築
setup: $(VENV_DIR)/bin/activate

$(VENV_DIR)/bin/activate: requirements.txt
	@echo "--- 仮想環境を作成し、ライブラリをインストールします ---"
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "\nセットアップが完了しました。以下のコマンドで仮想環境を有効化してください:"
	@echo "source $(VENV_DIR)/bin/activate"

# 仮想環境を有効化するためのコマンドを表示
activate:
	@echo "仮想環境を有効化するには、お使いのシェルで以下のコマンドを実行してください:"
	@echo "source $(VENV_DIR)/bin/activate"

# データパイプライン
pipeline: scrape clean index
	@echo "--- 全てのデータ処理パイプラインが完了しました ---"

scrape:
	@echo ">>> Step 1: scraper.py を実行中..."
	$(PYTHON) src/scraper.py

clean:
	@echo ">>> Step 2: cleaner.py を実行中..."
	$(PYTHON) src/cleaner.py

index:
	@echo ">>> Step 3: indexer.py を実行中..."
	$(PYTHON) src/indexer.py

# アプリケーションの実行
run:
	@echo "--- Streamlitアプリケーションを起動します (URL: http://localhost:8501) ---"
	$(STREAMLIT) run app.py --server.headless true

# 生成物のクリーンアップ
clean-all:
	@read -p "生成されたデータとベクトルストアを全て削除しますか？ (y/N) " yn; \
	if [ "$$yn" = "y" ]; then \
		echo "削除を実行します..."; \
		rm -rf outputs/* outputs_test/* vector_store; \
		echo "削除が完了しました。"; \
	else \
		echo "キャンセルされました。"; \
	fi
