"""
샘플 데이터 생성기
"""
from typing import List
import logging
from core.data_models import ReasoningDataPoint
from core.data_collector import ReasoningDatasetCollector

logger = logging.getLogger(__name__)


class SampleDataGenerator:
    """샘플 추론 데이터 생성기"""

    def __init__(self, collector: ReasoningDatasetCollector):
        self.collector = collector

    def generate_math_problems(self) -> List[ReasoningDataPoint]:
        """수학 문제 생성"""
        math_problems = [
            ReasoningDataPoint(
                id="",
                category="math",
                difficulty="easy",
                question="15 + 23 = ?",
                correct_answer="38",
                explanation="15와 23을 더하면 38입니다.",
                source="basic_arithmetic"
            ),
            ReasoningDataPoint(
                id="",
                category="math",
                difficulty="easy",
                question="7 × 8 = ?",
                correct_answer="56",
                explanation="7에 8을 곱하면 56입니다.",
                source="basic_arithmetic"
            ),
            ReasoningDataPoint(
                id="",
                category="math",
                difficulty="medium",
                question="한 직사각형의 길이가 12cm, 너비가 7cm일 때 둘레는?",
                correct_answer="38",
                explanation="직사각형 둘레 = 2 × (길이 + 너비) = 2 × (12 + 7) = 38 cm",
                source="geometry"
            ),
            ReasoningDataPoint(
                id="",
                category="math",
                difficulty="medium",
                question="원주율을 3.14로 할 때, 반지름이 5cm인 원의 넓이는?",
                correct_answer="78.5",
                explanation="원의 넓이 = π × r² = 3.14 × 5² = 3.14 × 25 = 78.5 cm²",
                source="geometry"
            ),
            ReasoningDataPoint(
                id="",
                category="math",
                difficulty="hard",
                question="3x + 7 = 22일 때, x의 값은?",
                correct_answer="5",
                explanation="3x = 22 - 7 = 15, 따라서 x = 15 ÷ 3 = 5",
                source="algebra"
            ),
            ReasoningDataPoint(
                id="",
                category="math",
                difficulty="hard",
                question="이차방정식 x² - 5x + 6 = 0의 해는?",
                correct_answer="x = 2 또는 x = 3",
                explanation="(x - 2)(x - 3) = 0이므로 x = 2 또는 x = 3",
                source="algebra"
            ),
        ]
        return math_problems

    def generate_logic_problems(self) -> List[ReasoningDataPoint]:
        """논리 문제 생성"""
        logic_problems = [
            ReasoningDataPoint(
                id="",
                category="logic",
                difficulty="easy",
                question="모든 새는 날 수 있다. 독수리는 새다. 따라서 독수리는 날 수 있다. 이 추론이 올바른가?",
                correct_answer="예",
                explanation="삼단논법의 올바른 형태입니다. 대전제, 소전제, 결론이 논리적으로 연결됩니다.",
                source="syllogism"
            ),
            ReasoningDataPoint(
                id="",
                category="logic",
                difficulty="medium",
                question="A, B, C, D 네 사람이 있다. A는 B보다 키가 크고, B는 C보다 키가 크며, C는 D보다 키가 크다. 누가 가장 키가 큰가?",
                correct_answer="A",
                explanation="A > B > C > D 순서이므로 A가 가장 키가 큽니다.",
                source="logical_reasoning"
            ),
            ReasoningDataPoint(
                id="",
                category="logic",
                difficulty="medium",
                question="다음 중 논리적으로 올바른 것은? 1) 모든 고양이는 동물이다. 2) 일부 동물은 고양이가 아니다. 3) 따라서 일부 고양이는 동물이 아니다.",
                correct_answer="올바르지 않다",
                explanation="결론이 전제와 모순됩니다. 모든 고양이는 동물이므로 고양이가 동물이 아닐 수 없습니다.",
                source="logical_reasoning"
            ),
            ReasoningDataPoint(
                id="",
                category="logic",
                difficulty="hard",
                question="5명이 일렬로 앉는데, A는 B 옆에 앉지 않고, C는 맨 끝에 앉지 않는다. 가능한 배치 방법의 수는?",
                correct_answer="54",
                explanation="전체 경우의 수에서 제약 조건을 만족하지 않는 경우를 제외해야 합니다.",
                source="combinatorics"
            ),
        ]
        return logic_problems

    def generate_reading_comprehension_problems(self) -> List[ReasoningDataPoint]:
        """독해 문제 생성"""
        reading_problems = [
            ReasoningDataPoint(
                id="",
                category="reading_comprehension",
                difficulty="easy",
                question="다음 문장에서 주어는 무엇인가? '강아지가 공원에서 뛰어다니고 있다.'",
                correct_answer="강아지",
                explanation="'강아지가'에서 '강아지'가 동작의 주체인 주어입니다.",
                source="grammar"
            ),
            ReasoningDataPoint(
                id="",
                category="reading_comprehension",
                difficulty="medium",
                question="다음 글의 주제는? '인공지능 기술의 발전으로 우리 생활이 편리해지고 있다. 스마트폰, 자율주행차, 음성인식 등 다양한 분야에서 활용되고 있다.'",
                correct_answer="인공지능 기술이 생활에 미치는 영향",
                explanation="글 전체가 인공지능 기술이 생활을 편리하게 만드는 다양한 사례들을 소개하고 있습니다.",
                source="reading_comprehension"
            ),
            ReasoningDataPoint(
                id="",
                category="reading_comprehension",
                difficulty="medium",
                question="다음 글에서 글쓴이의 의견은? '최근 환경 문제가 심각해지고 있다. 우리 모두가 일회용품 사용을 줄이고 재활용을 실천해야 한다.'",
                correct_answer="환경보호를 위해 개인의 실천이 필요하다",
                explanation="글쓴이는 환경 문제 해결을 위해 개인 차원의 노력이 중요하다고 주장하고 있습니다.",
                source="reading_comprehension"
            ),
            ReasoningDataPoint(
                id="",
                category="reading_comprehension",
                difficulty="hard",
                question="다음 시의 화자의 정서는? '낙엽이 떨어지는 가을날, 혼자 걷는 길이 쓸쓸하다. 지나간 시간들이 아련하게 떠오른다.'",
                correct_answer="그리움과 쓸쓸함",
                explanation="가을의 정취와 혼자만의 시간을 통해 화자의 그리움과 쓸쓸한 감정이 드러납니다.",
                source="literature"
            ),
        ]
        return reading_problems

    def generate_common_sense_problems(self) -> List[ReasoningDataPoint]:
        """상식 문제 생성"""
        common_sense_problems = [
            ReasoningDataPoint(
                id="",
                category="common_sense",
                difficulty="easy",
                question="비가 오면 사람들은 무엇을 사용하는가?",
                correct_answer="우산",
                explanation="비를 피하기 위해 우산을 사용합니다.",
                source="daily_life"
            ),
            ReasoningDataPoint(
                id="",
                category="common_sense",
                difficulty="medium",
                question="음식을 오래 보관하려면 어디에 두어야 하는가?",
                correct_answer="냉장고",
                explanation="냉장고의 낮은 온도가 음식의 부패를 지연시킵니다.",
                source="daily_life"
            ),
            ReasoningDataPoint(
                id="",
                category="common_sense",
                difficulty="medium",
                question="다음 중 가장 무거운 것은? A) 종이 B) 물 C) 공기 D) 철",
                correct_answer="D) 철",
                options=["종이", "물", "공기", "철"],
                explanation="같은 부피에서 철의 밀도가 가장 높습니다.",
                source="physics_common_sense"
            ),
            ReasoningDataPoint(
                id="",
                category="common_sense",
                difficulty="hard",
                question="사람이 물 없이 생존할 수 있는 대략적인 기간은?",
                correct_answer="3-5일",
                explanation="개인차와 환경에 따라 다르지만 일반적으로 3-5일 정도입니다.",
                source="survival_knowledge"
            ),
        ]
        return common_sense_problems

    def generate_science_problems(self) -> List[ReasoningDataPoint]:
        """과학 문제 생성"""
        science_problems = [
            ReasoningDataPoint(
                id="",
                category="science",
                difficulty="easy",
                question="물의 끓는점은 섭씨 몇 도인가?",
                correct_answer="100도",
                explanation="1기압에서 물의 끓는점은 섭씨 100도입니다.",
                source="chemistry"
            ),
            ReasoningDataPoint(
                id="",
                category="science",
                difficulty="medium",
                question="빛의 속도는 초당 약 몇 km인가?",
                correct_answer="300,000km",
                explanation="진공에서 빛의 속도는 초당 약 30만 km입니다.",
                source="physics"
            ),
            ReasoningDataPoint(
                id="",
                category="science",
                difficulty="medium",
                question="사람의 심장은 몇 개의 방으로 구성되어 있는가?",
                correct_answer="4개",
                explanation="심장은 좌심방, 우심방, 좌심실, 우심실의 4개 방으로 구성됩니다.",
                source="biology"
            ),
            ReasoningDataPoint(
                id="",
                category="science",
                difficulty="hard",
                question="DNA의 이중나선 구조를 발견한 과학자는?",
                correct_answer="왓슨과 크릭",
                explanation="제임스 왓슨과 프랜시스 크릭이 1953년 DNA의 이중나선 구조를 발견했습니다.",
                source="biology"
            ),
        ]
        return science_problems

    def add_all_sample_data(self) -> int:
        """모든 샘플 데이터 추가"""
        try:
            total_count = 0

            # 각 카테고리별 샘플 데이터 생성 및 추가
            categories = [
                ("수학", self.generate_math_problems()),
                ("논리", self.generate_logic_problems()),
                ("독해", self.generate_reading_comprehension_problems()),
                ("상식", self.generate_common_sense_problems()),
                ("과학", self.generate_science_problems()),
            ]

            for category_name, problems in categories:
                count = self.collector.add_batch_data_points(problems)
                total_count += count
                logger.info(f"{category_name} 문제 {count}개 추가")

            logger.info(f"총 {total_count}개의 샘플 데이터 추가 완료")
            return total_count

        except Exception as e:
            logger.error(f"샘플 데이터 추가 오류: {e}")
            return 0

    def create_custom_problem(self,
                              category: str,
                              difficulty: str,
                              question: str,
                              correct_answer: str,
                              explanation: str = None,
                              options: List[str] = None,
                              source: str = "manual") -> ReasoningDataPoint:
        """커스텀 문제 생성"""
        return ReasoningDataPoint(
            id="",
            category=category,
            difficulty=difficulty,
            question=question,
            correct_answer=correct_answer,
            explanation=explanation,
            options=options,
            source=source
        )