import os
import re
import ssl
import sys
import json
import time
from google import genai
from google.genai import types
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Union

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.configs import GEMINI_API_KEY, CONFIG, RAG_PROMPT, AUGMENT_LARGE_PROMPT
api_key=GEMINI_API_KEY

ssl._create_default_https_context = ssl._create_unverified_context

def augment_large_dataset(folder_path, merged_filename, BATCH_SIZE=40):
    client = genai.Client(api_key=api_key)
    clean_folder_path = os.path.normpath(folder_path)
    clean_filename = os.path.normpath(merged_filename)
    input_file = os.path.abspath(os.path.join(clean_folder_path, clean_filename))
    
    # \,/문제 해결용
    output_file = os.path.abspath(os.path.join(clean_folder_path, f"final_augmented_{clean_filename}"))
    status_file = os.path.abspath(os.path.join(clean_folder_path, "augmentation_status.json"))

    if not os.path.exists(input_file):
        print(f"[오류] 원본 파일이 없습니다: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        merged_data = json.load(f)

    # 기존 진행 상황 로드 (이어하기용)
    if os.path.exists(output_file) and os.path.exists(status_file):
        with open(output_file, "r", encoding="utf-8") as f:
            augmented_dataset = json.load(f)
        with open(status_file, "r", encoding="utf-8") as f:
            last_processed_idx = json.load(f).get("last_idx", 0)
        print(f"-> 이어서 시작합니다. (지난 진행 위치: {last_processed_idx}번 데이터까지 완료)")
    else:
        augmented_dataset = []
        last_processed_idx = 0
        print(f"-> 처음부터 증강을 시작합니다. 총 데이터: {len(merged_data)}개")

    # BATCH_SIZE(예: 40개)씩 묶어서 루프 돌기
    for i in range(last_processed_idx, len(merged_data), BATCH_SIZE):
        batch_items = merged_data[i:i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        total_batches = (len(merged_data) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n[{current_batch_num}/{total_batches} 묶음] {i}번 ~ {i + len(batch_items)-1}번 처리 중...")

        # 모델 전송용 데이터 포맷팅
        prompt_data = []
        for local_id, item in enumerate(batch_items):
            prompt_data.append({
                "id": local_id,
                "question": item.get("question"),
                "answer": item.get("answer")
            })
            # 원본 데이터 먼저 결과셋에 포함
            augmented_dataset.append(item)

        batch_json_text = json.dumps(prompt_data, ensure_ascii=False, indent=2)
        prompt = AUGMENT_LARGE_PROMPT.format(batch_json_text=batch_json_text)

        # API 호출 및 재시도 로직
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=CONFIG["MODEL_NAME"],
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        # 명확한 API 표준 스키마 정의
                        response_schema=types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    "id": types.Schema(type=types.Type.INTEGER),
                                    "similar_questions": types.Schema(
                                        type=types.Type.ARRAY,
                                        items=types.Schema(type=types.Type.STRING)
                                    )
                                },
                                required=["id", "similar_questions"]
                            )
                        )
                    ),
                )
                
                generated_results = json.loads(response.text)
                '''generated_results =[ {
                        "question": "선익시스템은 현재 어떤 디스플레이 장비의 국산화에 역점을 두고 있나요?",
                        "answer": "차세대 IT용 OLED 증착 양산 설비 개발 및 양산화와 OLED 디스플레이 핵심 장비의 국산화에 역점을 두고 있습니다."
                    }]'''
                # 결과 매칭 및 누적
                for res in generated_results:
                    local_id = res.get("id")
                    sim_questions = res.get("similar_questions", [])
                    
                    if local_id < len(batch_items):
                        orig_a = batch_items[local_id].get("answer")
                        for sim_q in sim_questions:
                            augmented_dataset.append({
                                "question": sim_q,
                                "answer": orig_a
                            })
                
                success = True
                break
                
            except Exception as e:
                error_msg = str(e)
                print(f" -> [오류] 시도 {attempt+1}/{max_retries}: {error_msg}")
                if "503" in error_msg or "429" in error_msg:
                    time.sleep((attempt + 1) * 30)
                else:
                    break

        if not success:
            print(f" -> [경고] {current_batch_num}번 묶음은 오류로 인해 건너뜁니다. 나중에 수동 확인 필요.")

        # 매 배치가 끝날 때마다 '안전하게 중간 저장' (크래시 대비)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(augmented_dataset, f, ensure_ascii=False, indent=4)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"last_idx": i + BATCH_SIZE}, f)

        # 대량 호출 시 속도 제한(Rate Limit) 방지를 위한 휴식
        time.sleep(3)

    # 모든 작업 완료 시 상태 파일 삭제
    if os.path.exists(status_file):
        os.remove(status_file)

    print(f"\n최종 완료! 데이터가 {len(merged_data)}개에서 {len(augmented_dataset)}개로 확장되었습니다.")
    print(f"최종 저장 파일: {output_file}")

def generate_rag_dataset(input_path, output_dir="rag_data", status_file="processed_batches.json", BATCH_SIZE=3):

    # API key
    client = genai.Client(api_key=api_key)

    #저장 위치
    os.makedirs(output_dir, exist_ok=True)
    status_file_path = os.path.join(output_dir, status_file)

    if os.path.exists(status_file_path):
        with open(status_file_path, "r") as f:
            processed_batches = json.load(f)
    else:
            processed_batches = []

        #
    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    page_splits = re.split(r'={3,}\s*SOURCE URL\s*\[\d+\]:.*?\s*={3,}', full_text)
    pages = [p.strip() for p in page_splits if p.strip() and len(p.strip()) > 30]
    batches = [pages[i:i + BATCH_SIZE] for i in range(0, len(pages), BATCH_SIZE)]

    print(f"작업 시작: {len(batches)}개의 묶음 확인됨 (저장 경로: {output_dir})")
    doneLLM = False
    for i, batch_pages in enumerate(batches):
        
        batch_num = i + 1
        
        # 건너뛰기 로직
        if batch_num in processed_batches:
            continue
            
        print(f"\n[시도 {batch_num}/{len(batches)}] 분석 중...")
        combined_text = "\n\n--- [페이지 구분] ---\n\n".join(batch_pages)
        prompt = RAG_PROMPT.format(combined_text=combined_text)

        # 5. API 호출 및 예외 처리 (재시도 로직)
        max_retries = 3
        success = False
        
        for attempt in range(max_retries):
            if doneLLM:
                break
            try:
                response = client.models.generate_content(
                    model=CONFIG["MODEL_NAME"],
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json"),
                )
                
                batch_data = json.loads(response.text)
                '''batch_data = [ {
                        "question": "선익시스템은 현재 어떤 디스플레이 장비의 국산화에 역점을 두고 있나요?",
                        "answer": "차세대 IT용 OLED 증착 양산 설비 개발 및 양산화와 OLED 디스플레이 핵심 장비의 국산화에 역점을 두고 있습니다."
                    }]'''

                # 배치별 저장
                with open(os.path.join(output_dir, f"batch_{batch_num}.json"), "w", encoding="utf-8") as f:
                    json.dump(batch_data, f, ensure_ascii=False, indent=4)
                
                # 기록 업데이트
                processed_batches.append(batch_num)
                with open(status_file_path, "w") as f:
                    json.dump(processed_batches, f)
                
                print(f"-> 완료! [이번 추출: {len(batch_data)}개]")
                success = True
                break 
                
            except Exception as e:
                error_msg = str(e)
                print(f"-> [오류] 묶음 {batch_num} 처리 중 (시도 {attempt+1}/{max_retries}): {error_msg}")
                
                if "503" in error_msg :
                    wait_time = (attempt + 1) * 30
                    print(f"-> 서버가 바쁩니다. {wait_time}초 대기 후 재시도합니다.")
                    time.sleep(wait_time)
                elif "429" in error_msg:
                    print("플랜 할당량이 끝났습니다~")
                    doneLLM = True
                    break

                else:
                    break
        if doneLLM:
            break
        time.sleep(3)
    print("\n최종 완료!")


def merge_json_files(base_data_dir="data"):
    if not os.path.exists(base_data_dir):
        print(f"[오류] '{base_data_dir}' 폴더가 존재하지 않습니다.")
        return

    sub_dirs = [d for d in os.listdir(base_data_dir) if os.path.isdir(os.path.join(base_data_dir, d))]
    print(f"총 {len(sub_dirs)}개의 하위 폴더를 찾았습니다: {sub_dirs}\n")

    for sub_dir in sub_dirs:
        current_dir = os.path.join(base_data_dir, sub_dir)
        print(f"[{sub_dir}] 폴더 병합 시작...")

        merged_list = []
        
        file_list = [f for f in os.listdir(current_dir) if f.startswith("batch_") and f.endswith(".json")]
        
        file_list.sort(key=lambda x: int(re.findall(r'\d+', x)[0]) if re.findall(r'\d+', x) else 0)

        if not file_list:
            print(f" -> 병합할 batch_*.json 파일이 없습니다. 건너뜁니다.")
            continue

        for filename in file_list:
            file_path = os.path.join(current_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        merged_list.extend(data)
                    else:
                        merged_list.append(data)
            except Exception as e:
                print(f" -> [오류] {filename} 읽기 실패: {e}")

        output_path = os.path.join(current_dir, f"merged_{sub_dir}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(merged_list, f, ensure_ascii=False, indent=4)

        print(f" -> 완료! 총 {len(merged_list)}개 데이터 통합 -> '{output_path}'\n")

    print("모든 폴더의 병합 작업이 최종 완료되었습니다!")

def get_text_from_urls(url_list):
    scraped_dataset = []

    for i, url in enumerate(url_list, 1):
        try:
            time.sleep(0.5) #
            
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                print(f" [{i}/{len(url_list)}] 접근 실패: {url}")
                continue
                
            response.encoding = response.apparent_encoding
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
            
            text_content = soup.get_text()
            lines = text_content.split('\n')
            clean_lines = [line.strip() for line in lines if line.strip()]
            clean_text = "\n".join(clean_lines)
            
            if clean_text.strip():
                scraped_dataset.append({
                    "url": url,
                    "text": clean_text
                })
                print(f"[{i}/{len(url_list)}] 크롤링 완료: {url}")
            else:
                print(f"[{i}/{len(url_list)}] 빈 페이지 건너뜀: {url}")
                
        except Exception as e:
            print(f"[{i}/{len(url_list)}] 에러 발생 패스: {url} - {e}")
            
    return scraped_dataset


def get_urls_from_sitemap(sitemap_url, print_num=3):

    try:
        
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(sitemap_url+"/sitemap.xml", headers=headers, timeout=10)
        response.raise_for_status() 
        
        soup = BeautifulSoup(response.text, features="xml")
        
        url_list = []
        loc_tags = soup.find_all('loc')
    
        
        for loc in loc_tags:
            url = loc.get_text().strip()
            if url: 
                url_list = Union_filter(url, url_list) 
                
        print(f"총 {len(url_list)}개의 URL을 찾았습니다.\n")
        print(f"{print_num}개의 URL을 출력.\n")
        for i, url in enumerate(url_list[:print_num], 1): 
            print(f"{i}: {url}")
        print(f"\n")
        return url_list

    except Exception as e:
        print(f"사이트맵을 읽어오는 중 오류가 발생했습니다: {e}")
        return []

def Union_filter(url, url_list):
    if url not in url_list:
        url_list.append(url)
    return url_list


def save_data_to_txt(scraped_data, file_name="sunic_all_pages.txt"):

    if not scraped_data:
        print("저장할 데이터가 비어있습니다.")
        return
        
    print(f"{file_name} 파일로 저장을 시작합니다...")
    
    with open(file_name, "w", encoding="utf-8") as f:
        for i, data in enumerate(scraped_data, 1):
            url = data["url"]
            text = data["text"]
            
            # 파일 쓰기 포맷팅
            f.write(f"\n\n========================================\n")
            f.write(f"SOURCE URL [{i}]: {url}\n")
            f.write(f"========================================\n\n")
            f.write(text)
            
    print(f"파일 저장 완료: {file_name} (총 {len(scraped_data)}개 페이지 분량)")