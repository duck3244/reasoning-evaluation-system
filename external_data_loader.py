"""
외부 데이터셋 로더
"""
import requests
import json
from typing import List, Dict, Any, Optional
import logging
from data_models import ReasoningDataPoint
from data_collector import ReasoningDatasetCollector

logger = logging.getLogger(__name__)


class ExternalDatasetLoader:
    """외부 추론 데이터셋 로더"""

    def __init__(self, collector: ReasoningDatasetCollector):
        self.collector = collector

    def load_gsm8k_sample(self) -> int:
        """GSM8K 스타일 수학 문제 샘플 로드"""
        try:
            gsm8k_problems = [
                ReasoningDataPoint(
                    id="",
                    category="math",
                    difficulty="medium",
                    question="Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every morning and bakes 4 into muffins for her friends every day. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                    correct_answer="18",
                    explanation="Janet gets 16 eggs per day. She eats 3 and bakes 4, so she uses 3 + 4 = 7 eggs. She has 16 - 7 = 9 eggs left. She sells these for $2 each, so she makes 9 * $2 = $18.",
                    source="gsm8k_sample"
                ),
                ReasoningDataPoint(
                    id="",
                    category="math",
                    difficulty="medium",
                    question="A store sells pencils for $0.25 each and erasers for $0.75 each. If Tom buys 8 pencils and 3 erasers, how much does he spend in total?",
                    correct_answer="4.25",
                    explanation="8 pencils cost 8 × $0.25 = $2.00. 3 erasers cost 3 × $0.75 = $2.25. Total: $2.00 + $2.25 = $4.25",
                    source="gsm8k_sample"
                ),
                ReasoningDataPoint(
                    id="",
                    category="math",
                    difficulty="hard",
                    question="A rectangular garden has a length that is 3 times its width. If the perimeter is 48 meters, what is the area of the garden?",
                    correct_answer="108",
                    explanation="Let width = w, then length = 3w. Perimeter = 2(w + 3w) = 8w = 48, so w = 6m. Length = 18m. Area = 6 × 18 = 108 m²",
                    source="gsm8k_sample"
                ),
            ]

            return self.collector.add_batch_data_points(gsm8k_problems)

        except Exception as e:
            logger.error(f"GSM8K 샘플 로드 오류: {e}")
            return 0

    def load_arc_sample(self) -> int:
        """ARC 스타일 과학 추론 문제 샘플 로드"""
        try:
            arc_problems = [
                ReasoningDataPoint(
                    id="",
                    category="science",
                    difficulty="hard",
                    question="Which of the following best explains why ice floats on water?",
                    correct_answer="Ice is less dense than liquid water",
                    options=[
                        "Ice is colder than liquid water",
                        "Ice is less dense than liquid water",
                        "Ice has a different chemical composition",
                        "Ice contains air bubbles"
                    ],
                    explanation="When water freezes, its molecules form a crystalline structure that takes up more space, making ice less dense than liquid water.",
                    source="arc_sample",
                    metadata={"answer_key": "B"}
                ),
                ReasoningDataPoint(
                    id="",
                    category="science",
                    difficulty="hard",
                    question="What happens to the mass of a wooden log when it is burned completely?",
                    correct_answer="The total mass is conserved but redistributed",
                    options=[
                        "The mass disappears completely",
                        "The mass increases due to combustion",
                        "The total mass is conserved but redistributed",
                        "Only half the mass remains"
                    ],
                    explanation="According to the law of conservation of mass, the mass is converted to ash, gases (CO2, H2O vapor), and energy, but the total mass remains the same.",
                    source="arc_sample",
                    metadata={"answer_key": "C"}
                ),
            ]

            return self.collector.add_batch_data_points(arc_problems)

        except Exception as e:
            logger.error(f"ARC 샘플 로드 오류: {e}")
            return 0

    def load_hellaswag_sample(self) -> int:
        """HellaSwag 스타일 상식 추론 문제 샘플 로드"""
        try:
            hellaswag_problems = [
                ReasoningDataPoint(
                    id="",
                    category="common_sense",
                    difficulty="hard",
                    question="A person is cooking pasta. They put the pasta in boiling water. What is most likely to happen next?",
                    correct_answer="They will wait for the pasta to cook",
                    options=[
                        "They will immediately drain the pasta",
                        "They will wait for the pasta to cook",
                        "They will add ice to the water",
                        "They will turn off the heat immediately"
                    ],
                    explanation="When cooking pasta, you need to let it cook in boiling water for several minutes before it's ready.",
                    source="hellaswag_sample",
                    metadata={"correct_index": 1}
                ),
                ReasoningDataPoint(
                    id="",
                    category="common_sense",
                    difficulty="medium",
                    question="Someone is getting ready for work in the morning. They have showered and are now standing in front of their closet. What will they most likely do next?",
                    correct_answer="Choose clothes to wear",
                    options=[
                        "Go back to bed",
                        "Choose clothes to wear",
                        "Start cooking dinner",
                        "Call in sick"
                    ],
                    explanation="After showering and standing in front of a closet, the logical next step is to select appropriate clothing for work.",
                    source="hellaswag_sample",
                    metadata={"correct_index": 1}
                ),
            ]

            return self.collector.add_batch_data_points(hellaswag_problems)

        except Exception as e:
            logger.error(f"HellaSwag 샘플 로드 오류: {e}")
            return 0

    def load_korean_datasets(self) -> int:
        """한국어 추론 데이터셋 로드"""
        try:
            korean_problems = [
                ReasoningDataPoint(
                    id="",
                    category="reading_comprehension",
                    difficulty="medium",
                    question="다음 글의 요지는 무엇인가? '기술의 발전은 인간의 삶을 편리하게 만들었지만, 동시에 새로운 문제들도 야기하고 있다. 우리는 기술의 이익을 누리면서도 그 부작용을 최소화하는 방법을 모색해야 한다.'",
                    correct_answer="기술 발전의 양면성을 인식하고 균형잡힌 접근이 필요하다",
                    explanation="글쓴이는 기술이 편리함과 문제를 동시에 가져다준다는 양면성을 지적하며, 균형잡힌 접근의 필요성을 강조하고 있습니다.",
                    source="korean_rc"
                ),
                ReasoningDataPoint(
                    id="",
                    category="logic",
                    difficulty="hard",
                    question="철수는 영희보다 3살 많고, 영희는 민수보다 2살 적으며, 민수는 지수보다 1살 많다. 지수가 18살이면 철수는 몇 살인가?",
                    correct_answer="22",
                    explanation="지수 18살, 민수 19살(18+1), 영희 17살(19-2), 철수 20살(17+3)... 아니, 다시 계산하면 지수 18살, 민수 19살, 영희 21살, 철수 24살이 아니라... 지수 18살, 민수 19살, 영희 17살, 철수 20살입니다.",
                    source="korean_logic"
                ),
                ReasoningDataPoint(
                    id="",
                    category="common_sense",
                    difficulty="medium",
                    question="한국에서 설날에 먹는 전통 음식은?",
                    correct_answer="떡국",
                    explanation="설날에는 새해를 맞아 떡국을 먹는 전통이 있습니다.",
                    source="korean_culture"
                ),
                ReasoningDataPoint(
                    id="",
                    category="math",
                    difficulty="medium",
                    question="한 교실에 학생이 30명 있다. 이 중 남학생이 전체의 60%라면 여학생은 몇 명인가?",
                    correct_answer="12",
                    explanation="남학생: 30 × 0.6 = 18명, 여학생: 30 - 18 = 12명",
                    source="korean_math"
                ),
            ]

            return self.collector.add_batch_data_points(korean_problems)

        except Exception as e:
            logger.error(f"한국어 데이터셋 로드 오류: {e}")
            return 0

    def load_from_huggingface_format(self, dataset_name: str, data_list: List[Dict]) -> int:
        """Hugging Face 형태의 데이터셋 로드"""
        try:
            problems = []

            for item in data_list:
                # 데이터 형식에 따라 적절히 변환
                if dataset_name == "gsm8k":
                    problem = ReasoningDataPoint(
                        id="",
                        category="common_sense",
                        difficulty="hard",
                        question=f"{item.get('ctx', '')} 다음 중 가장 적절한 것은?",
                        correct_answer=options[correct_idx] if correct_idx < len(options) else "",
                        options=options,
                        source="hellaswag",
                        metadata={"correct_index": correct_idx}
                    )
                elif dataset_name == "arc":
                    choices = item.get('choices', {})
                    options = choices.get('text', [])
                    labels = choices.get('label', [])
                    answer_key = item.get('answerKey', '')

                    correct_answer = None
                    for j, label in enumerate(labels):
                        if label == answer_key and j < len(options):
                            correct_answer = options[j]
                            break

                    problem = ReasoningDataPoint(
                        id="",
                        category="science",
                        difficulty="hard",
                        question=item.get('question', ''),
                        correct_answer=correct_answer or "",
                        options=options,
                        source="arc_challenge",
                        metadata={"answer_key": answer_key}
                    )
                else:
                    # 기본 형태
                    problem = ReasoningDataPoint(
                        id="",
                        category=item.get('category', 'unknown'),
                        difficulty=item.get('difficulty', 'medium'),
                        question=item.get('question', ''),
                        correct_answer=item.get('answer', ''),
                        explanation=item.get('explanation'),
                        options=item.get('options'),
                        source=dataset_name
                    )

                if problem.question and problem.correct_answer:
                    problems.append(problem)

            return self.collector.add_batch_data_points(problems)

        except Exception as e:
            logger.error(f"Hugging Face 형태 데이터 로드 오류: {e}")
            return 0

    def load_from_custom_api(self, api_url: str, api_key: str = None, headers: Dict[str, str] = None) -> int:
        """커스텀 API에서 데이터 로드"""
        try:
            request_headers = headers or {}
            if api_key:
                request_headers['Authorization'] = f'Bearer {api_key}'

            response = requests.get(api_url, headers=request_headers, timeout=30)
            response.raise_for_status()

            data = response.json()
            problems = []

            # API 응답 형태에 따라 적절히 파싱
            items = data.get('problems', data.get('data', data.get('items', [])))

            for item in items:
                problem = ReasoningDataPoint(
                    id="",
                    category=item.get('category', 'unknown'),
                    difficulty=item.get('difficulty', 'medium'),
                    question=item.get('question', item.get('prompt', '')),
                    correct_answer=item.get('answer', item.get('correct_answer', '')),
                    explanation=item.get('explanation', item.get('rationale')),
                    options=item.get('options', item.get('choices')),
                    source=f"api_{api_url.split('//')[-1].split('/')[0]}",
                    metadata=item.get('metadata', {})
                )

                if problem.question and problem.correct_answer:
                    problems.append(problem)

            return self.collector.add_batch_data_points(problems)

        except requests.RequestException as e:
            logger.error(f"API 요청 오류: {e}")
            return 0
        except Exception as e:
            logger.error(f"API 데이터 로드 오류: {e}")
            return 0

    def load_from_json_file(self, file_path: str, source_name: str) -> int:
        """JSON 파일에서 외부 데이터셋 형태로 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            problems = []

            # 데이터가 리스트인지 딕셔너리인지 확인
            items = data if isinstance(data, list) else data.get('data', [])

            for item in items:
                problem = ReasoningDataPoint(
                    id="",
                    category=item.get('category', 'unknown'),
                    difficulty=item.get('difficulty', 'medium'),
                    question=item.get('question', ''),
                    correct_answer=item.get('answer', item.get('correct_answer', '')),
                    explanation=item.get('explanation'),
                    options=item.get('options'),
                    source=source_name,
                    metadata=item.get('metadata')
                )

                if problem.question and problem.correct_answer:
                    problems.append(problem)

            return self.collector.add_batch_data_points(problems)

        except Exception as e:
            logger.error(f"JSON 파일 로드 오류: {e}")
            return 0

    def load_all_samples(self) -> Dict[str, int]:
        """모든 샘플 외부 데이터셋 로드"""
        results = {}

        try:
            logger.info("외부 데이터셋 샘플 로딩 시작...")

            # GSM8K 샘플
            gsm8k_count = self.load_gsm8k_sample()
            results['gsm8k'] = gsm8k_count
            logger.info(f"GSM8K 샘플: {gsm8k_count}개")

            # ARC 샘플
            arc_count = self.load_arc_sample()
            results['arc'] = arc_count
            logger.info(f"ARC 샘플: {arc_count}개")

            # HellaSwag 샘플
            hellaswag_count = self.load_hellaswag_sample()
            results['hellaswag'] = hellaswag_count
            logger.info(f"HellaSwag 샘플: {hellaswag_count}개")

            # 한국어 데이터셋
            korean_count = self.load_korean_datasets()
            results['korean'] = korean_count
            logger.info(f"한국어 데이터셋: {korean_count}개")

            total = sum(results.values())
            logger.info(f"총 {total}개의 외부 데이터셋 샘플 로드 완료")

            return results

        except Exception as e:
            logger.error(f"외부 데이터셋 로드 오류: {e}")
            return results

    def validate_data_format(self, data: Dict[str, Any]) -> bool:
        """데이터 형식 검증"""
        required_fields = ['question', 'answer']

        for field in required_fields:
            if field not in data or not data[field]:
                return False

        # 카테고리 검증
        if 'category' in data:
            from data_models import Constants
            if data['category'] not in Constants.CATEGORIES:
                logger.warning(f"알 수 없는 카테고리: {data['category']}")

        # 난이도 검증
        if 'difficulty' in data:
            from data_models import Constants
            if data['difficulty'] not in Constants.DIFFICULTIES:
                logger.warning(f"알 수 없는 난이도: {data['difficulty']}")

        return Trued="",
        category = "math",
        difficulty = "medium",
        question = item.get('question', ''),
        correct_answer = item.get('answer', '').split('####')[-1].strip() if '####' in item.get('answer',
                                                                                                '') else item.get(
            'answer', ''),
        explanation = item.get('answer', ''),
        source = "gsm8k"

    )
    elif dataset_name == "hellaswag":
    options = item.get('endings', [])
    correct_idx = int(item.get('label', 0))
    problem = ReasoningDataPoint(
    i