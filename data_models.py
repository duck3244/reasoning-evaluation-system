"""
데이터 모델 정의
"""
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json


@dataclass
class ReasoningDataPoint:
    """추론 평가용 데이터 포인트"""
    id: str
    category: str  # "math", "logic", "common_sense", "reading_comprehension", "science"
    difficulty: str  # "easy", "medium", "hard"
    question: str
    correct_answer: str
    explanation: Optional[str] = None
    options: Optional[List[str]] = None  # 객관식의 경우
    source: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """초기화 후 처리"""
        if not self.id:
            self.id = self.generate_id()

        if not self.created_at:
            self.created_at = datetime.now().isoformat()

        self.updated_at = datetime.now().isoformat()

    def generate_id(self) -> str:
        """질문과 답변으로 고유 ID 생성"""
        content = f"{self.question}_{self.correct_answer}_{self.category}"
        return hashlib.md5(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReasoningDataPoint':
        """딕셔너리에서 객체 생성"""
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'ReasoningDataPoint':
        """JSON 문자열에서 객체 생성"""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class EvaluationResult:
    """평가 결과 데이터 모델"""
    id: str
    data_point_id: str
    model_name: str
    predicted_answer: str
    is_correct: bool
    confidence_score: Optional[float] = None
    reasoning_steps: Optional[List[str]] = None
    execution_time: Optional[float] = None
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class DatasetStatistics:
    """데이터셋 통계 정보"""
    total_count: int
    category_counts: Dict[str, int]
    difficulty_counts: Dict[str, int]
    source_counts: Dict[str, int]
    created_at: str

    def __post_init__(self):
        if not hasattr(self, 'created_at') or not self.created_at:
            self.created_at = datetime.now().isoformat()


# 상수 정의
class Constants:
    """시스템 상수"""

    # 카테고리
    CATEGORIES = [
        "math",
        "logic",
        "common_sense",
        "reading_comprehension",
        "science",
        "history",
        "language"
    ]

    # 난이도
    DIFFICULTIES = ["easy", "medium", "hard"]

    # 데이터 소스
    SOURCES = [
        "manual",
        "gsm8k",
        "hellaswag",
        "arc_challenge",
        "korean_dataset",
        "custom_api"
    ]

    # 테이블 이름
    TABLE_REASONING_DATA = "REASONING_DATA"
    TABLE_EVALUATION_RESULTS = "EVALUATION_RESULTS"
    TABLE_DATASET_STATS = "DATASET_STATISTICS"