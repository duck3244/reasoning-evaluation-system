# LLM 추론 성능 평가 시스템

Oracle DB를 기반으로 한 LLM(Large Language Model) 추론 성능 평가용 데이터셋 수집 및 평가 시스템입니다.

## 주요 기능

### 1. 데이터 수집 및 관리
- **다양한 추론 카테고리**: 수학, 논리, 상식, 독해, 과학 등
- **난이도별 분류**: Easy, Medium, Hard
- **외부 데이터셋 연동**: GSM8K, ARC, HellaSwag 등 
- **다국어 지원**: 한국어, 영어
- **Oracle DB 기반 안정적 저장**

### 2. 평가 시스템
- **자동화된 모델 평가**: 배치 처리 지원
- **다양한 메트릭**: 정확도, 실행시간, 카테고리별 성능
- **평가 결과 저장**: 상세한 평가 로그 및 통계
- **모델 비교**: 여러 모델 성능 비교 분석

### 3. 데이터 관리
- **Import/Export**: JSON, CSV 형태 지원
- **API 연동**: 외부 데이터 소스 연결
- **통계 정보**: 실시간 데이터셋 현황

## 시스템 구조

```
reasoning-evaluation-system/
├── data_models.py          # 데이터 모델 정의
├── database_config.py      # Oracle DB 설정 및 연결
├── data_collector.py       # 데이터 수집 및 저장
├── sample_data_generator.py # 샘플 데이터 생성
├── external_data_loader.py # 외부 데이터셋 로더
├── evaluation_system.py    # 평가 시스템
├── main.py                # 메인 시스템 통합
├── requirements.txt       # 의존성 패키지
└── README.md             # 시스템 설명서
```

## 설치 및 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. Oracle DB 설정
```bash
# 설정 파일 생성
python main.py config
```

`db_config.json` 파일을 수정하여 실제 Oracle DB 정보 입력:
```json
{
  "username": "your_username",
  "password": "your_password", 
  "dsn": "localhost:1521/XE",
  "pool_min": 1,
  "pool_max": 10,
  "pool_increment": 1
}
```

### 3. 시스템 초기화 및 실행
```bash
# 기본 실행 (DB 초기화 + 샘플 데이터 로드 + 평가)
python main.py

# 커스텀 평가 데모
python main.py demo
```

## 사용 방법

### 1. 기본 사용법
```python
from database_config import DatabaseConfig
from main import ReasoningEvaluationSystem

# 시스템 초기화
db_config = DatabaseConfig(
    username="user", 
    password="pass", 
    dsn="localhost:1521/XE"
)
system = ReasoningEvaluationSystem(db_config)

# 데이터베이스 설정
system.setup_database()

# 샘플 데이터 로드
system.load_sample_data()

# 평가 실행
results = system.run_evaluation("my_model", test_size=100)
```

### 2. 커스텀 모델 평가
```python
def my_llm_function(prompt: str) -> str:
    # 실제 LLM API 호출
    # return llm_api.generate(prompt)
    return "모델 응답"

# 평가 실행
results = system.evaluator.evaluate_model(
    model_name="my_custom_llm",
    evaluation_set=evaluation_data,
    model_function=my_llm_function
)
```

### 3. 데이터 관리
```python
# 데이터 추가
from data_models import ReasoningDataPoint

new_problem = ReasoningDataPoint(
    id="",
    category="math",
    difficulty="medium", 
    question="2 + 2 = ?",
    correct_answer="4",
    explanation="2와 2를 더하면 4입니다."
)
system.collector.add_data_point(new_problem)

# 데이터 내보내기
system.export_data("math", "json")  # 수학 문제만 JSON으로
system.export_data(None, "csv")     # 전체 데이터 CSV로

# 데이터 가져오기
count = system.import_data("my_data.json")
```

## 데이터 구조

### ReasoningDataPoint
```python
@dataclass
class ReasoningDataPoint:
    id: str                    # 고유 ID
    category: str             # 카테고리 (math, logic, etc.)
    difficulty: str           # 난이도 (easy, medium, hard)
    question: str             # 문제
    correct_answer: str       # 정답
    explanation: str          # 해설 (선택)
    options: List[str]        # 객관식 선택지 (선택) 
    source: str              # 데이터 출처 (선택)
    created_at: str          # 생성 시간
    metadata: Dict           # 추가 메타데이터 (선택)
```

### 평가 결과
```python
@dataclass  
class EvaluationResult:
    id: str                   # 평가 결과 ID
    data_point_id: str       # 문제 ID
    model_name: str          # 모델명
    predicted_answer: str    # 예측 답변
    is_correct: bool         # 정답 여부
    execution_time: float    # 실행 시간
    created_at: str         # 평가 시간
```

## 지원하는 카테고리

- **math**: 수학 (산술, 기하, 대수)
- **logic**: 논리 추론 (삼단논법, 논리 퍼즐)
- **common_sense**: 상식 추론 (일상 지식)
- **reading_comprehension**: 독해 (문법, 문학)
- **science**: 과학 (물리, 화학, 생물)
- **history**: 역사
- **language**: 언어

## 외부 데이터셋 지원

- **GSM8K**: 초등학교 수준 수학 문제
- **ARC**: AI2 Reasoning Challenge (과학 추론)
- **HellaSwag**: 상식 추론
- **커스텀 API**: REST API 연동
- **한국어 데이터셋**: 한국어 추론 문제

## 평가 메트릭

- **전체 정확도**: 올바른 답변 비율
- **카테고리별 정확도**: 각 추론 영역별 성능
- **난이도별 정확도**: 문제 난이도별 성능  
- **실행 시간**: 평균 응답 속도
- **모델 비교**: 여러 모델 성능 대조

## Oracle DB 스키마

### REASONING_DATA 테이블
```sql
CREATE TABLE REASONING_DATA (
    ID VARCHAR2(32) PRIMARY KEY,
    CATEGORY VARCHAR2(50) NOT NULL,
    DIFFICULTY VARCHAR2(20) NOT NULL,
    QUESTION CLOB NOT NULL,
    CORRECT_ANSWER CLOB NOT NULL,
    EXPLANATION CLOB,
    OPTIONS CLOB,
    SOURCE VARCHAR2(100),
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    METADATA CLOB
);
```

### EVALUATION_RESULTS 테이블
```sql
CREATE TABLE EVALUATION_RESULTS (
    ID VARCHAR2(32) PRIMARY KEY,
    DATA_POINT_ID VARCHAR2(32) NOT NULL,
    MODEL_NAME VARCHAR2(100) NOT NULL,
    PREDICTED_ANSWER CLOB NOT NULL,
    IS_CORRECT NUMBER(1),
    EXECUTION_TIME NUMBER(10,3),
    CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    METADATA CLOB
);
```

## 확장 가능성

- **새로운 카테고리 추가**: 상수 파일 수정으로 쉽게 확장
- **다국어 지원**: 언어별 데이터셋 추가
- **고급 평가 메트릭**: F1-score, BLEU 등 추가 가능
- **실시간 평가**: 스트리밍 평가 지원
- **웹 인터페이스**: 대시보드 구축 가능

## 문제 해결

### 일반적인 오류들

1. **Oracle DB 연결 실패**
   - `db_config.json` 설정 확인
   - Oracle 서버 상태 확인
   - 네트워크 연결 확인

2. **메모리 부족**
   - 배치 크기 조정
   - 연결 풀 크기 조정

3. **데이터 형식 오류**
   - JSON/CSV 파일 형식 확인
   - 필수 필드 누락 확인