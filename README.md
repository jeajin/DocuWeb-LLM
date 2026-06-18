"# DocuWeb-LLM" 

# 🤖 Web-to-RAG QA System with Qwen-2.5

웹사이트에서 데이터를 추출하여 LLM 기반의 QA 데이터셋으로 가공하고, 이를 활용해 경량화된 **Qwen2.5 (4-bit Quantized)** 모델 기반의 RAG(검색 증강 생성) 질의응답 시스템을 구축하는 프로젝트입니다.



## 🚀 파이프라인 개요 (Pipeline)

본 프로젝트는 데이터 수집부터 최종 모델 서빙까지 총 4단계의 파이프라인으로 구성되어 있습니다.
```text
[1. Web Scraping] ➔ [2. LLM QA Generation] ➔ [3. Data Augmentation] ➔ [4. 4-bit Qwen2.5 RAG]
```

---
### 🏃‍♂️ 실행 방법 (Usage)

전체 파이프라인은 데이터 수집, 데이터 가공(QA 변환 및 증강), 최종 RAG 질의 순으로 진행됩니다. `utils` 폴더 내의 Jupyter Notebook(`*.ipynb`) 파일을 순서대로 실행해 주세요.

---

### Step 1. 웹 데이터 수집 및 텍스트 추출
목표 사이트의 웹페이지를 크롤링하여 원본 텍스트 파일(`*.txt`)로 저장합니다.

1. `utils/scraper.ipynb` 파일을 엽니다.
2. 스크랩하고자 하는 대상 URL 설정을 확인한 뒤, 모든 셀을 실행합니다.
3. 실행이 완료되면 `utils/` 폴더 내에 텍스트 데이터 파일(`*.txt`)이 생성됩니다.

---

### Step 2. QA 데이터 변환 및 데이터 증강
추출한 텍스트를 LLM을 통해 질의응답(QA) 형태로 변환하고, RAG 성능 향상을 위해 데이터를 풍부하게 증강(Augmentation)합니다.

1. `utils/rag.ipynb` 파일을 엽니다.
2. 앞서 생성된 `*.txt` 파일을 로드하여 LLM 파이프라인을 통해 QA 데이터셋을 생성합니다.
3. 데이터 증강 과정을 거쳐 최종 결과물을 `data/rag_data3_ko/` 디렉토리에 저장하고, 이를 기반으로 `my_faiss_index/`에 벡터 데이터베이스를 구축합니다.

---

### Step 3. 4-bit Qwen2.5 기반 RAG 질의응답
구축된 FAISS 벡터 DB와 4비트로 압축된 Qwen2.5 모델을 연동하여 최종 질의응답을 수행합니다.

1. `utils/QA.ipynb` 파일을 엽니다.
2. 4-bit 양자화된 `Qwen2.5` 모델과 `my_faiss_index` 벡터 DB가 정상적으로 로드되는지 확인합니다.
3. 노트북 하단의 질의응답 셀에 원하는 질문을 입력하고 실행하여 시스템의 답변을 확인합니다.

### ⚠️ 보안 및 주의 사항 (Security Notice)

본 프로젝트의 `configs/configs.py` 파일에는 시스템 작동에 필요한 LLM 프롬프트 변수, 모델 Configuration, 그리고 **API Key** 설정이 포함되어 있습니다.

- **API Key 유출 방지**: 보안상의 이유로 실제 API Key는 GitHub 저장소에 업로드되지 않으며, `.gitignore`를 통해 관리되거나 비워진 상태로 제공됩니다.
- **실행 전 필수 설정**: 프로젝트를 정상적으로 실행하려면 `configs/configs.py` 파일을 열고 사용하시는 플랫폼의 개인 API Key를 직접 입력한 후 사용하셔야 합니다.

```python
# configs/configs.py 내부 구조 예시

# 1. API KEY 설정 (실행 전 본인의 키로 변경 필수)
API_KEY = "YOUR_ACTUAL_API_KEY_HERE"

# 2. 하이퍼파라미터 및 모델 설정
CONFIG = {
    "MODEL_NAME": "gemini-2.5-flash",
    "BATCH_SIZE": 40,
    "DELAY_SECONDS": 10,
    "MAX_RETRIES": 3
}

# 3. RAG 가공 및 증강 프롬프트 (정의된 변수명)
RAG_PROMPT = """..."""          # 소스 텍스트 기반 QA 추출 프롬프트
AUGMENT_LARGE_PROMPT = """...""" # 추출된 QA 데이터 증강 프롬프트

#### 디렉토리 구조
```text
📁 .
│  📄 .gitignore          # Git 제외 파일 목록 (대용량 데이터 및 보안 설정)
│  📄 README.md           # 프로젝트 개요 및 실행 방법 안내서
│
├─📁 configs
│  │  📄 configs.py       # 프로젝트 설정값 (경로, 모델 파라미터 등)
│
└─📁 utils                # 단계별 핵심 모듈 및 유틸리티 폴더
    │  📄 scraper.ipynb   # [Step 1] 웹사이트 데이터 추출 및 크롤러
    │  📄 QA.ipynb         # [Step 2~3] LLM 기반 QA 데이터 변환 및 증강
    │  📄 rag.ipynb        # [Step 4] 4-bit Qwen2.5 연동 및 RAG 파이프라인 구축
    │  📄 sunic_all_pages.txt # 추출 완료된 원본 텍스트 데이터
    │  📄 utils.py         # 데이터 전처리 및 공통 유틸리티 함수
    │
    ├─📁 data             # 원본 및 가공 데이터셋 저장소
    │  ├─📁 rag_data3      # RAG 데이터셋 (V3)
    │  └─📁 rag_data3_ko   # 한국어 타겟 RAG 데이터셋 (V3)
    │        
    └─📁 my_faiss_index   # 고속 문서 검색을 위한 벡터 데이터베이스 (FAISS)
            📄 index.faiss  # FAISS 벡터 인덱스 파일
            📄 index.pkl    # 텍스트 메타데이터 매핑 파일
```


