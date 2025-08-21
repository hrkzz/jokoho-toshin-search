import json
import re
import random
from pathlib import Path
import os
import unicodedata
from tqdm import tqdm

def parse_japanese_date(text):
    if not isinstance(text, str): return None
    text = text.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    eras = {
        '明治': 1868 - 1, '大正': 1912 - 1, '昭和': 1926 - 1,
        '平成': 1989 - 1, '令和': 2019 - 1,
    }
    match = re.search(r'(明治|大正|昭和|平成|令和)\s*(\d+|元)\s*年\s*(\d+)\s*月\s*(\d+)\s*日', text)
    if not match: return None
    era, year_str, month, day = match.groups()
    year = 1 if year_str == '元' else int(year_str)
    seireki_year = eras[era] + year
    return f"{seireki_year:04d}-{int(month):02d}-{int(day):02d}"

def structure_committee_members(committee_data):
    """
    委員リストを整形する改善版関数（最終版）。
    入力データの形式を判断し、最適な方法で処理するハイブリッドアプローチを採用。
    """
    if not committee_data:
        return []

    potential_names = []
    # --- データ形式に応じた処理の分岐 ---
    if isinstance(committee_data, str):
        # ケースA: 単一の文字列で、改行を含む場合 -> 最初の1行目のみを対象とする
        if '\n' in committee_data:
            first_line = committee_data.split('\n', 1)[0]
            potential_names = re.split(r'[、，]+', first_line)
        # ケースB: 単一の文字列だが改行を含まない場合 -> 全体を対象とする
        else:
            potential_names = re.split(r'[、，]+', committee_data)
    # ケースC: 文字列のリストの場合 -> 各要素を処理
    elif isinstance(committee_data, list):
        for item in committee_data:
            if isinstance(item, str):
                # 各要素をさらに読点やカンマで分割する
                potential_names.extend(re.split(r'[、，]+', item))
    else:
        # 未知のデータ形式の場合は空リストを返す
        return []

    # --- 共通のクリーンアップ処理 ---
    cleaned_members = []
    for name in potential_names:
        # "委員"という接頭辞や前後の空白を除去
        processed_name = name.replace('委員', '').strip()
        # 姓と名の間に存在する可能性のある空白（全角/半角）も除去
        processed_name = re.sub(r'\s+', '', processed_name)

        # 空文字列でなければリストに追加
        if processed_name:
            cleaned_members.append(processed_name)

    # 重複を除去して返す
    return list(set(cleaned_members))

def clean_text_content(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'（以下「法」という。）', '', text)
    return text

def clean_data(input_path, output_path, sample_size=None):
    input_file = Path(input_path)
    output_file = Path(output_path)
    if not input_file.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_file}")
        return
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if sample_size and len(data) > sample_size:
        print(f"全{len(data)}件から{sample_size}件をランダムにサンプリングします...")
        data = random.sample(data, sample_size)
    
    cleaned_data = []
    for record in tqdm(data, desc="データクリーニング中"):
        new_record = {}
        
        new_record['URL'] = record.get('URL', '')
        agency = record.get('諮問庁', '')
        case_name = record.get('事件名', '')
        new_record['諮問庁'] = agency
        new_record['事件名'] = case_name

        new_record['諮問日_iso'] = parse_japanese_date(record.get('諮問日', ''))
        new_record['答申日_iso'] = parse_japanese_date(record.get('答申日', ''))
        new_record['委員'] = structure_committee_members(record.get('委員', []))
        
        conclusion = clean_text_content(record.get("第１_審査会の結論", ""))
        reason = clean_text_content(record.get("第５_審査会の判断の理由", ""))
        
        new_record['summary_text'] = (
            f"事件名：{case_name}\n"
            f"諮問庁：{agency}\n\n"
            f"結論：{conclusion}\n\n"
            f"判断の理由：{reason}"
        )
        
        detail_texts = {}
        detail_keys = [
            "第２_審査請求人の主張の要旨",
            "第３_諮問庁の説明の要旨",
            "参加人の主張の要旨"
        ]
        for key in detail_keys:
            content = clean_text_content(record.get(key, ""))
            if content:
                detail_texts[key] = content
        new_record['detail_texts'] = detail_texts
        
        cleaned_data.append(new_record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
    print(f"クリーニングが正常に完了しました。'{output_file}' を確認してください。")

if __name__ == '__main__':
    INPUT_JSON_PATH = 'outputs/toshin_data.json'  
    OUTPUT_JSON_PATH = 'outputs/cleaned_toshin_data.json'
    SAMPLE_COUNT = None
    
    clean_data(INPUT_JSON_PATH, OUTPUT_JSON_PATH, sample_size=SAMPLE_COUNT)