# LLM 추론 성능 평가 시스템

Oracle 데이터베이스를 기반으로 한 대규모 LLM(Large Language Model) 추론 성능 평가 시스템입니다. 
수학, 논리, 상식, 독해, 과학 등 다양한 영역의 문제를 통해 모델의 추론 능력을 체계적으로 측정하고 분석할 수 있습니다.

## 🌟 주요 기능

### 📊 데이터 관리
- **다양한 데이터 소스 지원**: GSM8K, HellaSwag, ARC, 한국어 데이터셋
- **배치 처리 최적화**: 대용량 데이터 효율적 처리 (적응적 배치 크기)
- **데이터 검증**: 자동 데이터 품질 검사 및 정리
- **백업/복원**: 데이터 안전성 보장

### 🎯 평가 시스템
- **균형잡힌 테스트셋**: 카테고리별, 난이도별 균등 분배
- **실시간 평가**: 모델 응답 시간 및 정확도 측정
- **다양한 지표**: 정확도, 카테고리별 성능, 실행 시간 분석
- **비교 분석**: 여러 모델 간 성능 비교

### 🚀 성능 최적화
- **메모리 관리**: 적응적 메모리 정리 및 모니터링
- **병렬 처리**: 멀티스레드 배치 처리
- **쿼리 최적화**: Oracle 힌트 활용 및 인덱스 최적화
- **캐싱**: 쿼리 결과 캐싱으로 성능 향상

### 📈 모니터링 및 로깅
- **구조화된 로깅**: JSON 형태의 상세 로그
- **실시간 알림**: 에러율, 성능 임계값 모니터링
- **성능 추적**: 함수별 실행 시간 및 메모리 사용량
- **로그 분석**: 패턴 분석 및 리포트 생성

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Loaders  │    │  Evaluation     │    │   Monitoring    │
│                 │    │    Engine       │    │    System       │
│ • GSM8K         │    │                 │    │                 │
│ • HellaSwag     │────┤ • Model Eval    │────┤ • Logging       │
│ • ARC           │    │ • Metrics       │    │ • Alerting      │
│ • Korean Data   │    │ • Comparison    │    │ • Analytics     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Core System    │
                    │                 │
                    │ • Data Models   │
                    │ • DB Manager    │
                    │ • Optimizer     │
                    │ • Collector     │
                    └─────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Oracle Database │
                    │                 │
                    │ • Reasoning Data│
                    │ • Eval Results  │
                    │ • Statistics    │
                    └─────────────────┘
```

## 🛠️ 설치 및 설정

### 1. 환경 요구사항
```bash
Python 3.8+
Oracle Database 12c+
최소 4GB RAM (권장 8GB+)
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. 데이터베이스 설정
```bash
# 설정 파일 생성
cp config/db_config.json.example config/db_config.json
# db_config.json 편집하여 실제 DB 정보 입력

# 데이터베이스 초기화
python scripts/init_database.py
```

### 4. 기본 설정 확인
```bash
# 연결 테스트
python -c "from core.database_config import load_db_config_from_file, DatabaseManager; 
config = load_db_config_from_file('config/db_config.json');
db = DatabaseManager(config);
print('DB 연결:', db.db_config.test_connection())"
```

## 🚀 빠른 시작

### 기본 사용법
```python
from main import ReasoningEvaluationSystem
from core.database_config import load_db_config_from_file

# 시스템 초기화
db_config = load_db_config_from_file("config/db_config.json")
system = ReasoningEvaluationSystem(db_config)

# 데이터베이스 설정
system.setup_database()

# 샘플 데이터 로드
results = system.load_sample_data()
print(f"로드된 데이터: {sum(results.values())}개")

# 모델 평가 실행
def my_model(prompt):
    # 여기에 실제 LLM 모델 호출 코드 작성
    return "model_response"

eval_results = system.evaluator.evaluate_model(
    model_name="my_model_v1",
    evaluation_set=system.evaluator.create_evaluation_set(test_size=50),
    model_function=my_model
)

print(f"정확도: {eval_results['accuracy']:.2%}")
```

### 대용량 데이터 처리
```python
from core.performance_optimizer import OptimizedDataProcessor
from core.data_collector import ReasoningDatasetCollector

# 최적화된 프로세서 사용
processor = OptimizedDataProcessor(db_manager, batch_size=2000)

# 대량 데이터 처리
large_dataset = [...]  # 대량의 ReasoningDataPoint 객체들
result = processor.process_large_dataset(large_dataset, process_func)
```

### 실시간 모니터링
```python
from monitoring.logging_system import LogMonitor, setup_application_logging

# 로깅 시스템 설정
setup_application_logging({
    'log_level': 'INFO',
    'log_format': 'structured',
    'log_file': 'logs/application.log'
})

# 실시간 모니터링 시작
monitor = LogMonitor('logs/application.log')
monitor.monitor_realtime()  # 백그라운드에서 실행
```

## 📊 데이터 형식

### 추론 데이터 포인트
```python
from core.data_models import ReasoningDataPoint

data_point = ReasoningDataPoint(
    id="unique_id",
    category="math",           # math, logic, common_sense, reading_comprehension, science
    difficulty="medium",       # easy, medium, hard
    question="2 + 2 = ?",
    correct_answer="4",
    explanation="2와 2를 더하면 4입니다.",
    options=["2", "3", "4", "5"],  # 객관식인 경우
    source="gsm8k",
    metadata={"type": "arithmetic"}
)
```

### 평가 결과
```python
{
    "total_questions": 100,
    "correct_answers": 85,
    "accuracy": 0.85,
    "category_accuracy": {
        "math": {"accuracy": 0.90, "correct": 18, "total": 20},
        "logic": {"accuracy": 0.80, "correct": 16, "total": 20}
    },
    "difficulty_accuracy": {
        "easy": {"accuracy": 0.95, "correct": 19, "total": 20},
        "medium": {"accuracy": 0.85, "correct": 17, "total": 20},
        "hard": {"accuracy": 0.75, "correct": 15, "total": 20}
    },
    "average_execution_time": 1.25,
    "evaluation_date": "2024-01-01T10:00:00"
}
```

## 🎯 지원하는 데이터셋

### 수학 (Math)
- **GSM8K**: 초등학교 수준 수학 문제
- **기하학**: 도형, 넓이, 부피 계산
- **대수학**: 방정식, 함수

### 논리 (Logic)  
- **삼단논법**: 논리적 추론
- **조합론**: 순열, 조합 문제
- **패턴 인식**: 규칙 찾기

### 상식 (Common Sense)
- **HellaSwag**: 상황 추론
- **일상 상식**: 물리 법칙, 사회 규범
- **한국 문화**: 한국 특화 상식

### 독해 (Reading Comprehension)
- **주제 파악**: 글의 요지, 주장
- **세부 정보**: 구체적 내용 이해
- **추론**: 행간의 의미

### 과학 (Science)
- **ARC Challenge**: 과학적 추론
- **물리, 화학, 생물**: 기본 과학 지식
- **의학**: 기초 의학 상식

## 🔧 고급 기능

### 1. 커스텀 평가 지표 추가
```python
def custom_metric(predictions, ground_truth, metadata):
    """커스텀 평가 지표 함수"""
    # 사용자 정의 평가 로직
    return score

# 평가 시스템에 추가
evaluator.add_custom_metric("my_metric", custom_metric)
```

### 2. 병렬 처리 최적화
```python
from core.performance_optimizer import ParallelProcessor

processor = ParallelProcessor(max_workers=8)
result = processor.process_parallel_batches(batches, process_func)
```

### 3. 실시간 성능 모니터링
```python
from monitoring.logging_system import log_performance

@log_performance()
def my_model_function(prompt):
    # 모델 실행 시간과 메모리 사용량이 자동으로 로깅됨
    return model_response
```

### 4. 데이터 품질 관리
```python
# 데이터 품질 리포트 생성
quality_report = collector.get_data_quality_report()

# 중복 데이터 정리
optimization_results = collector.optimize_storage()

# 오래된 데이터 정리
cleaned_count = collector.cleanup_old_data(days_old=30)
```

## 📈 성능 최적화 가이드

### 데이터베이스 최적화
```python
from core.performance_optimizer import IndexOptimizer

# 복합 인덱스 생성
optimizer = IndexOptimizer(db_manager)
optimizer.create_composite_indexes()

# 테이블 통계 갱신  
optimizer.analyze_table_statistics()
```

### 메모리 관리
```python
from core.performance_optimizer import MemoryManager

memory_manager = MemoryManager(memory_threshold_percent=80.0)

# 자동 메모리 정리
memory_manager.cleanup_if_needed()

# 메모리 사용량 체크
memory_info = memory_manager.check_memory_usage()
```

### 배치 크기 최적화
```python
from core.performance_optimizer import BatchOptimizer

optimizer = BatchOptimizer()

# 시스템 상황에 따른 적응적 배치 크기
optimal_size = optimizer.adaptive_batch_size(total_items=10000)
```

## 📋 API 참조

### 주요 클래스

#### ReasoningDatasetCollector
```python
# 데이터 추가
collector.add_data_point(data_point)
collector.add_batch_data_points(data_points)

# 데이터 조회
data = collector.get_data(category="math", difficulty="easy", limit=100)
item = collector.get_data_by_id("data_id")

# 데이터 내보내기/가져오기
collector.export_to_json("export.json")
collector.load_from_json("import.json")

# 통계 정보
stats = collector.get_statistics()
```

#### ReasoningEvaluator
```python
# 평가 데이터셋 생성
eval_set = evaluator.create_evaluation_set(test_size=100, balance_categories=True)

# 모델 평가
results = evaluator.evaluate_model(model_name, eval_set, model_function)

# 성능 비교
comparison = evaluator.compare_models(["model1", "model2"])
```

#### ExternalDatasetLoader
```python
# 외부 데이터셋 로드
loader.load_gsm8k_sample()
loader.load_arc_sample()
loader.load_korean_datasets()

# 모든 샘플 로드
results = loader.load_all_samples()
```

## 🧪 테스트

### 단위 테스트 실행
```bash
python -m pytest tests/test_data_collector.py -v
```

### 통합 테스트 실행
```bash
python -m pytest tests/test_integration.py -v
```

### 전체 테스트 실행
```bash
python -m pytest tests/ -v --cov=core --cov=evaluation --cov=data_loaders
```

## 🚨 트러블슈팅

### 일반적인 문제들

#### 1. 데이터베이스 연결 실패
```bash
# 연결 테스트
python -c "from core.database_config import load_db_config_from_file, DatabaseManager; 
config = load_db_config_from_file('config/db_config.json');
print('Config loaded:', config.test_connection())"
```

#### 2. 메모리 부족 오류
```python
# 배치 크기 조정
collector = ReasoningDatasetCollector(db_manager, batch_size=500)  # 기본값 1000에서 감소
```

#### 3. 성능 저하
```python
# 쿼리 캐시 정리
query_optimizer.clear_query_cache()

# 인덱스 통계 갱신
index_optimizer.analyze_table_statistics()
```

#### 4. 로그 파일 용량 증가
```python
# 로그 정리 및 압축
from monitoring.logging_system import LogRotationManager

rotation_manager = LogRotationManager("logs", max_age_days=30)
rotation_manager.cleanup_old_logs()
rotation_manager.compress_old_logs()
```

### 성능 모니터링

#### 시스템 리소스 체크
```python
from core.performance_optimizer import MemoryManager

memory_manager = MemoryManager()
memory_info = memory_manager.check_memory_usage()
print(f"메모리 사용률: {memory_info['system_memory_percent']:.1f}%")
```

#### 로그 분석
```python
from monitoring.logging_system import LogAnalyzer

analyzer = LogAnalyzer('logs/application.log')
summary = analyzer.generate_log_summary(hours=24)
print(f"에러율: {summary['error_rate']:.2%}")
```
---