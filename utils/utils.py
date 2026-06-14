import time
import requests
from bs4 import BeautifulSoup

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