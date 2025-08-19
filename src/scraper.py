import requests
from bs4 import BeautifulSoup
import re
import json
import time
import argparse
import os

def ensure_output_dir():
    """出力ディレクトリ 'outputs' が存在しない場合に作成します。"""
    os.makedirs("outputs", exist_ok=True)

def save_json(data, filename):
    """抽出したデータをJSONファイルとして保存します。"""
    temp_filename = filename + ".tmp"
    with open(os.path.join("outputs", temp_filename), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(os.path.join("outputs", temp_filename), os.path.join("outputs", filename))

def extract_report_info_and_urls(html_content):
    """
    検索結果ページのHTMLから「本文表示」のURLと詳細情報を抽出します。
    この関数はURLの重複を保持します。
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    results = []
    urls = []
    
    for table in soup.find_all('table', class_='search-result'):
        honbun_link_tag = table.find('a', text=re.compile(r'^\s*本文表示\s*$'))
        
        if honbun_link_tag and honbun_link_tag.get('href') and '/reportBody/' in honbun_link_tag.get('href'):
            record = {}
            href = honbun_link_tag.get('href')
            full_url = f"https://koukai-hogo-db.soumu.go.jp{href}" if href.startswith('/') else href
            
            record['本文URL'] = full_url
            urls.append(full_url)

            for row in table.find_all('tr', class_='search-result-body'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if '答申日／答申番号' in label:
                        record['答申日'] = value
                    elif '諮問日／諮問番号' in label:
                        record['諮問日'] = value
                    elif '諮問庁' in label:
                        record['諮問庁'] = value
                    elif '事件名' in label:
                        record['事件名'] = value
            
            results.append(record)
            
    return results, urls

def get_total_pages(html_content):
    """検索結果の最初のページから総ページ数を取得します。"""
    soup = BeautifulSoup(html_content, 'html.parser')
    paging_text = soup.find('div', class_='paging-item')
    if paging_text:
        match = re.search(r'／\s*([\d,]+)件', paging_text.get_text())
        if match:
            try:
                total_items = int(match.group(1).replace(',', ''))
                return (total_items + 19) // 20
            except (ValueError, IndexError):
                pass
    return 1

def get_search_results(session, response, headers, num_pages):
    """検索結果を巡回し、全答申のURLと基本情報を取得します。"""
    all_results = []
    all_urls = []
    
    print("1ページ目を取得中...")
    page_results, page_urls = extract_report_info_and_urls(response.text)
    all_results.extend(page_results)
    all_urls.extend(page_urls)
    print(f"  {len(page_urls)}件の本文URLを取得")

    for page_num in range(2, num_pages + 1):
        print(f"{page_num}ページ目を取得中...")
        try:
            time.sleep(0.5)
            page_response = session.get(
                f"https://koukai-hogo-db.soumu.go.jp/report/search/page/{page_num}",
                headers=headers, timeout=20
            )
            if page_response.status_code == 200:
                page_results, page_urls = extract_report_info_and_urls(page_response.text)
                all_results.extend(page_results)
                all_urls.extend(page_urls)
                print(f"  {len(page_urls)}件の本文URLを取得")
            else:
                print(f"  エラー: ステータスコード {page_response.status_code} (ページ {page_num})")
        except requests.RequestException as e:
            print(f"  ページ取得エラー: {e} (ページ {page_num})")
            
    return all_urls, all_results

def clean_urls_file(input_path, output_path):
    """
    URLファイルから不要なリンクを除外し、正しい形式のURLのみを抽出します。
    正しい形式: https://koukai-hogo-db.soumu.go.jp/reportBody/数字
    """
    print(f"\nURLファイルのクリーニングを開始: {input_path}")
    if not os.path.exists(input_path):
        print(f"エラー: 入力ファイルが見つかりません。")
        return []

    with open(input_path, 'r', encoding='utf-8') as f:
        urls = [line.strip() for line in f if line.strip()]

    original_count = len(urls)
    
    pattern = re.compile(r'^https://koukai-hogo-db\.soumu\.go\.jp/reportBody/\d+$')
    # 重複を除去し、ソートして順序を安定させる
    cleaned_urls = sorted(list(set(url for url in urls if pattern.match(url))))
    
    cleaned_count = len(cleaned_urls)
    excluded_count = original_count - cleaned_count

    with open(output_path, 'w', encoding='utf-8') as f:
        for url in cleaned_urls:
            f.write(url + '\n')
            
    print(f"クリーニング完了:")
    print(f"  元のURL数: {original_count}件")
    print(f"  不正/重複URLを除外: {excluded_count}件")
    print(f"  最終的なURL数: {cleaned_count}件")
    print(f"  保存先: {output_path}")
    
    return cleaned_urls

def extract_toshin_info(html_content, url):
    """答申詳細ページのHTMLから情報を抽出します（多様なフォーマット対応版）。"""
    soup = BeautifulSoup(html_content, 'html.parser')
    result = {
        'URL': url, '諮問庁': '', '諮問日': '', '答申日': '', '事件名': '',
        '第１_審査会の結論': '', '第２_審査請求人の主張の要旨': '', '第３_諮問庁の説明の要旨': '',
        '第４_参加人の主張の要旨': '',
        '第４_調査審議の経過': '', '第５_審査会の判断の理由': '', '委員': '', '別紙': ''
    }
    
    # === ヘッダー情報の抽出 (テーブル形式とPタグ形式に対応) ===
    header_table = soup.find('table')
    if header_table:
        rows = header_table.find_all('tr')
        if len(rows) < 10:
            for row in rows:
                cells = [cell.get_text(strip=True) for cell in row.find_all('td')]
                if len(cells) < 2: continue
                
                label = cells[0].replace('：', '').strip()
                value_raw = cells[1]
                if len(cells) > 2 and cells[1] == '：':
                    value_raw = cells[2]
                value = value_raw.lstrip('：').strip()

                if '諮問庁' in label: result['諮問庁'] = value
                elif '諮問日' in label: result['諮問日'] = value
                elif '答申日' in label: result['答申日'] = value
                elif '事件名' in label: result['事件名'] = value

    if not soup.body: return result
    for tag in soup(["script", "style"]): tag.decompose()
    body_text = soup.body.get_text(separator='\n', strip=True)

    header_fields_map = {
        '諮問庁': r'^\s*諮問庁\s*[:：]\s*(.*)',
        '諮問日': r'^\s*諮問日\s*[:：]\s*(.*)',
        '答申日': r'^\s*答申日\s*[:：]\s*(.*)',
        '事件名': r'^\s*事件名\s*[:：]\s*(.*)',
    }
    for field, pattern in header_fields_map.items():
        if not result[field]:
            match = re.search(pattern, body_text, re.MULTILINE)
            if match:
                result[field] = match.group(1).strip()

    # === 本文セクションの抽出 (セクション番号のズレに対応) ===
    next_section_pattern = r'^\s*第\s*[\d０-９一二三四五六七八九十]+'
    sections = {
        '第１_審査会の結論': fr'^\s*第[１1]\s*[　\s]*(?:審査会の結論)?(.*?)(?={next_section_pattern}|\Z)',
        '第２_審査請求人の主張の要旨': fr'^\s*第\s*[\d０-９一二三四五六七八九十]+\s*[　\s]*(?:審査請求人|異議申立人)[の\s]+主張[の\s]+(?:要旨|概要)(.*?)(?={next_section_pattern}|\Z)',
        '第３_諮問庁の説明の要旨': fr'^\s*第\s*[\d０-９一二三四五六七八九十]+\s*[　\s]*諮問庁[の\s]+説明[の\s]+(?:要旨|概要)(.*?)(?={next_section_pattern}|\Z)',
        '第４_参加人の主張の要旨': fr'^\s*第\s*[\d０-９一二三四五六七八九十]+\s*[　\s]*参加人[の\s]+主張[の\s]+要旨(.*?)(?={next_section_pattern}|\Z)',
        '第４_調査審議の経過': fr'^\s*第\s*[\d０-９一二三四五六七八九十]+\s*[　\s]*調査審議[の\s]+経過(.*?)(?={next_section_pattern}|\Z)',
        '第５_審査会の判断の理由': fr'^\s*第\s*[\d０-９一二三四五六七八九十]+\s*[　\s]*審査会[の\s]+判断(?:の理由)?(.*?)(?=[（(]第\d+部会[）)]|{next_section_pattern}|^\s*第\s*[\d０-９一二三四五六七八九十]+\s*.*答申に関与した委員|\Z)'
    }
    
    for key, pattern in sections.items():
        match = re.search(pattern, body_text, re.DOTALL | re.MULTILINE)
        if match:
            content = match.group(1).strip()
            if content:
                result[key] = re.sub(r'\s*\n\s*', '\n', content)

    # === 委員・別紙の抽出 (複数パターンに対応する改善版) ===
    search_text = body_text
    dai5_match = re.search(sections['第５_審査会の判断の理由'], body_text, re.DOTALL | re.MULTILINE)
    if dai5_match:
        search_text = body_text[dai5_match.end():]

    committee_match = re.search(r'（第\d+部会）\s*(委員.+?)(?=\n\n|\n\s*別紙|\n\s*別表|\Z)', search_text, re.DOTALL)
    if committee_match:
        result['委員'] = committee_match.group(1).strip()
    else:
        committee_match = re.search(r'第\s*[\d０-９一二三四五六七八九十]+\s*.*答申に関与した委員\s*\n\s*(.+?)(?=\n\n|\n\s*別紙|\n\s*別表|\Z)', search_text, re.DOTALL)
        if committee_match:
            result['委員'] = committee_match.group(1).strip()
    
    if not result['委員']:
        committee_match = re.search(r'^\s*委員\s*([^委員\n]+(?:、委員\s*[^委員\n]+)*)', body_text, re.MULTILINE)
        if committee_match:
            result['委員'] = committee_match.group(0).strip()

    besshi_match = re.search(r'(\n\s*(?:別紙|別表).*)', body_text, re.DOTALL)
    if besshi_match:
        result['別紙'] = besshi_match.group(1).strip()
        
    return result
    
def extract_all_toshin_data(urls, session, headers, initial_data, scraped_urls_set, sleep_time=0.8):
    """URLリストを元に全答申の詳細データを抽出します。中断・再開に対応。"""
    extracted_data = initial_data
    failed_urls = []
    total_urls = len(urls)
    
    urls_to_process_count = total_urls - len(scraped_urls_set)
    
    print(f"\n答申本文の取得を開始... 全{total_urls}件")
    print(f"（うち{len(scraped_urls_set)}件は取得済み、残り{urls_to_process_count}件を処理します）")
    
    processed_count = 0
    for i, url in enumerate(urls, 1):
        if url in scraped_urls_set:
            if i % 200 == 0:
                print(f"  進捗: {i}/{total_urls}件を確認済 (スキップ)")
            continue
        
        processed_count += 1
        
        # ▼▼▼ 修正点 ▼▼▼
        # 処理開始を知らせるメッセージを一度だけ表示
        if processed_count == 1:
            print("  詳細データの処理を開始しました...")

        try:
            time.sleep(sleep_time)
            response = session.get(url, headers=headers, timeout=20)
            if response.status_code == 200:
                response.encoding = response.apparent_encoding
                extracted_data.append(extract_toshin_info(response.text, url))
            else:
                print(f"  取得エラー: {url} (ステータスコード: {response.status_code})")
                failed_urls.append(url)
        except requests.RequestException as e:
            print(f"  取得エラー: {url} ({e})")
            failed_urls.append(url)
        
        # ▼▼▼ 修正点 ▼▼▼
        # 10件処理するごとに、進捗を含めた保存メッセージを表示
        if processed_count > 0 and processed_count % 10 == 0:
            save_json(extracted_data, "toshin_data.json")
            print(f"  進捗 {processed_count}/{urls_to_process_count}件: データを保存しました。(全体で {i}/{total_urls} 件目)")
    
    return extracted_data, failed_urls

def main():
    parser = argparse.ArgumentParser(description="総務省答申データベース取得ツール")
    parser.add_argument("--pages", type=int, help="取得するページ数（指定がない場合は全件）")
    args = parser.parse_args()
    ensure_output_dir()

    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
        'Referer': 'https://koukai-hogo-db.soumu.go.jp/report/'
    }
    
    cleaned_urls_path = os.path.join("outputs", 'report_urls_cleaned.txt')

    # --- ステップ1 & 2: URLリストファイルがなければ作成、あれば読み込む ---
    if os.path.exists(cleaned_urls_path):
        print("【ステップ1 & 2】はスキップします。（クリーンなURLリストが既に存在します）")
        with open(cleaned_urls_path, 'r', encoding='utf-8') as f:
            final_urls = [line.strip() for line in f if line.strip()]
        print(f"{len(final_urls)}件のURLを読み込みました: {cleaned_urls_path}")
    else:
        print("【ステップ1: 全検索結果からURLを取得】")
        try:
            response = session.post(
                'https://koukai-hogo-db.soumu.go.jp/report/search',
                data={'gyouseiKoukai': 'on', 'gyouseiHogo': 'on', 'dokuhouKoukai': 'on', 'dokuhouHogo': 'on',
                      'searchType': '1', 'searchQuery': '', 'order1': '12', 'limitOption': '1', 'openFlag': '1', 'status': ''},
                headers=headers
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"検索の実行に失敗しました: {e}")
            return

        total_pages_from_site = get_total_pages(response.text)
        total_pages = min(args.pages, total_pages_from_site) if args.pages else total_pages_from_site
        print(f"取得対象ページ数: {total_pages} ページ")
        
        all_urls, _ = get_search_results(session, response, headers, num_pages=total_pages)
        
        raw_urls_path = os.path.join("outputs", 'report_urls.txt')
        with open(raw_urls_path, 'w', encoding='utf-8') as f:
            for url in all_urls:
                f.write(url + '\n')
        print(f"\n{len(all_urls)}件のURLを {raw_urls_path} に保存しました。")

        print("\n【ステップ2: URLリストのクリーニング】")
        final_urls = clean_urls_file(input_path=raw_urls_path, output_path=cleaned_urls_path)
    
    # --- ステップ3: 詳細データの取得（中断・再開ロジック） ---
    if not final_urls:
        print("\n処理対象のURLがありません。処理を終了します。")
        return
        
    print("\n【ステップ3: 詳細情報の取得】")
    toshin_data_path = os.path.join("outputs", "toshin_data.json")
    toshin_data = []
    scraped_urls_set = set()

    if os.path.exists(toshin_data_path):
        print(f"再開処理を開始します: {toshin_data_path} を読み込み中...")
        try:
            with open(toshin_data_path, 'r', encoding='utf-8') as f:
                toshin_data = json.load(f)
            if not isinstance(toshin_data, list):
                print(f"警告: {toshin_data_path} が不正な形式です。新規に開始します。")
                toshin_data = []
            else:
                scraped_urls_set = {item['URL'] for item in toshin_data if 'URL' in item}
        except (json.JSONDecodeError, IOError) as e:
            print(f"エラー: 既存ファイルの読み込みに失敗 ({e})。新規に開始します。")
            toshin_data = []
    
    toshin_data, failed = extract_all_toshin_data(final_urls, session, headers, toshin_data, scraped_urls_set)

    # --- 最終結果の保存 ---
    print("\n【最終結果の保存】")
    save_json(toshin_data, "toshin_data.json")
    if failed: save_json(failed, "failed_urls.json")

    print("\n処理完了。")
    print(f"成功: {len(toshin_data)}件, 失敗: {len(failed)}件")
    print("保存先: outputs/toshin_data.json")
    if failed: print("失敗したURLは outputs/failed_urls.json を確認してください。")

if __name__ == "__main__":
    main()