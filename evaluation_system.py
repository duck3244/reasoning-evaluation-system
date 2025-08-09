"""
LLM 추론 성능 평가 시스템
"""
import json
import time
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging
from data_models import ReasoningDataPoint, EvaluationResult, Constants
from data_collector import ReasoningDatasetCollector
from database_config import DatabaseManager

logger = logging.getLogger(__name__)


class ReasoningEvaluator:
    """추론 성능 평가기"""

    def __init__(self, db_manager: DatabaseManager, collector: ReasoningDatasetCollector):
        self.db_manager = db_manager
        self.collector = collector

    def create_evaluation_set(self,
                              test_size: int = 100,
                              balance_categories: bool = True,
                              balance_difficulties: bool = True) -> List[ReasoningDataPoint]:
        """평가용 데이터셋 생성"""
        try:
            if balance_categories:
                # 카테고리별로 균등하게 선택
                categories = Constants.CATEGORIES
                per_category = max(1, test_size // len(categories))

                evaluation_set = []
                for category in categories:
                    if balance_difficulties:
                        # 각 카테고리 내에서 난이도별로 균등 분배
                        difficulties = Constants.DIFFICULTIES
                        per_difficulty = max(1, per_category // len(difficulties))

                        for difficulty in difficulties:
                            data = self.collector.get_data(
                                category=category,
                                difficulty=difficulty,
                                limit=per_difficulty
                            )
                            evaluation_set.extend(data)
                    else:
                        data = self.collector.get_data(category=category, limit=per_category)
                        evaluation_set.extend(data)

                # 목표 크기에 맞게 조정
                if len(evaluation_set) > test_size:
                    evaluation_set = evaluation_set[:test_size]
                elif len(evaluation_set) < test_size:
                    # 부족한 경우 추가 데이터로 채움
                    remaining = test_size - len(evaluation_set)
                    additional_data = self.collector.get_data(limit=remaining)
                    # 중복 제거
                    existing_ids = {item.id for item in evaluation_set}
                    for item in additional_data:
                        if item.id not in existing_ids and len(evaluation_set) < test_size:
                            evaluation_set.append(item)
            else:
                # 단순히 최신 데이터 선택
                evaluation_set = self.collector.get_data(limit=test_size)

            logger.info(f"평가 데이터셋 생성 완료: {len(evaluation_set)}개")
            return evaluation_set

        except Exception as e:
            logger.error(f"평가 데이터셋 생성 오류: {e}")
            return []

    def evaluate_model(self,
                       model_name: str,
                       evaluation_set: List[ReasoningDataPoint],
                       model_function: Callable[[str], str],
                       save_results: bool = True) -> Dict[str, Any]:
        """모델 평가 실행"""
        results = []
        start_time = time.time()

        try:
            for i, data_point in enumerate(evaluation_set):
                try:
                    # 모델에게 질문
                    prompt = self._create_prompt(data_point)

                    # 실행 시간 측정
                    question_start = time.time()
                    predicted_answer = model_function(prompt)
                    execution_time = time.time() - question_start

                    # 정답 확인
                    is_correct = self._check_answer(
                        predicted_answer,
                        data_point.correct_answer,
                        data_point.category
                    )

                    # 평가 결과 생성
                    eval_result = EvaluationResult(
                        id=f"{model_name}_{data_point.id}_{int(time.time())}",
                        data_point_id=data_point.id,
                        model_name=model_name,
                        predicted_answer=predicted_answer,
                        is_correct=is_correct,
                        execution_time=execution_time,
                        metadata={
                            "question": data_point.question,
                            "expected_answer": data_point.correct_answer,
                            "category": data_point.category,
                            "difficulty": data_point.difficulty
                        }
                    )

                    results.append(eval_result)

                    if (i + 1) % 10 == 0:
                        logger.info(f"평가 진행: {i + 1}/{len(evaluation_set)}")

                except Exception as e:
                    logger.error(f"문제 {i} 평가 중 오류: {e}")
                    continue

            total_time = time.time() - start_time

            # 결과 저장
            if save_results:
                self._save_evaluation_results(results)

            # 통계 계산
            stats = self._calculate_statistics(results, total_time)

            logger.info(f"모델 '{model_name}' 평가 완료: 정확도 {stats['accuracy']:.2%}")
            return stats

        except Exception as e:
            logger.error(f"모델 평가 오류: {e}")
            return {}

    def _create_prompt(self, data_point: ReasoningDataPoint) -> str:
        """평가용 프롬프트 생성"""
        prompt = f"다음 문제를 풀어주세요:\n\n{data_point.question}"

        if data_point.options:
            prompt += "\n\n선택지:"
            for i, option in enumerate(data_point.options):
                prompt += f"\n{chr(65 + i)}) {option}"
            prompt += "\n\n정답을 선택하고 간단히 설명해주세요."
        else:
            prompt += "\n\n답변과 함께 풀이 과정을 설명해주세요."

        return prompt

    def _check_answer(self, predicted: str, correct: str, category: str) -> bool:
        """정답 확인"""
        try:
            # 기본적인 문자열 정리
            predicted_clean = predicted.strip().lower()
            correct_clean = correct.strip().lower()

            # 완전 일치
            if predicted_clean == correct_clean:
                return True

            # 카테고리별 특별 처리
            if category == "math":
                return self._check_math_answer(predicted_clean, correct_clean)
            elif category == "logic":
                return self._check_logic_answer(predicted_clean, correct_clean)
            else:
                # 부분 일치 확인
                return correct_clean in predicted_clean or predicted_clean in correct_clean

        except Exception as e:
            logger.error(f"정답 확인 오류: {e}")
            return False

    def _check_math_answer(self, predicted: str, correct: str) -> bool:
        """수학 답안 확인"""
        try:
            # 숫자 추출 시도
            import re

            pred_numbers = re.findall(r'-?\d+\.?\d*', predicted)
            correct_numbers = re.findall(r'-?\d+\.?\d*', correct)

            if pred_numbers and correct_numbers:
                # 첫 번째 숫자 비교
                try:
                    pred_val = float(pred_numbers[0])
                    correct_val = float(correct_numbers[0])
                    return abs(pred_val - correct_val) < 0.01
                except ValueError:
                    pass

            return predicted == correct

        except Exception:
            return predicted == correct

    def _check_logic_answer(self, predicted: str, correct: str) -> bool:
        """논리 답안 확인"""
        # 예/아니오 형태 답변 처리
        yes_patterns = ['예', 'yes', '맞다', '참', 'true', '올바르다']
        no_patterns = ['아니오', 'no', '틀렸다', '거짓', 'false', '올바르지 않다']

        pred_is_yes = any(pattern in predicted for pattern in yes_patterns)
        pred_is_no = any(pattern in predicted for pattern in no_patterns)

        correct_is_yes = any(pattern in correct for pattern in yes_patterns)
        correct_is_no = any(pattern in correct for pattern in no_patterns)

        if (pred_is_yes and correct_is_yes) or (pred_is_no and correct_is_no):
            return True

        return predicted == correct

    def _save_evaluation_results(self, results: List[EvaluationResult]) -> bool:
        """평가 결과 저장"""
        try:
            sql = f"""
            INSERT INTO {Constants.TABLE_EVALUATION_RESULTS}
            (ID, DATA_POINT_ID, MODEL_NAME, PREDICTED_ANSWER, IS_CORRECT,
             CONFIDENCE_SCORE, REASONING_STEPS, EXECUTION_TIME, CREATED_AT, METADATA)
            VALUES (:1, :2, :3, :4, :5, :6, :7, :8, CURRENT_TIMESTAMP, :9)
            """

            params_list = []
            for result in results:
                params = [
                    result.id,
                    result.data_point_id,
                    result.model_name,
                    result.predicted_answer,
                    1 if result.is_correct else 0,
                    result.confidence_score,
                    json.dumps(result.reasoning_steps) if result.reasoning_steps else None,
                    result.execution_time,
                    json.dumps(result.metadata) if result.metadata else None
                ]
                params_list.append(params)

            self.db_manager.execute_batch_dml(sql, params_list)
            logger.info(f"평가 결과 {len(results)}개 저장 완료")
            return True

        except Exception as e:
            logger.error(f"평가 결과 저장 오류: {e}")
            return False

    def _calculate_statistics(self, results: List[EvaluationResult], total_time: float) -> Dict[str, Any]:
        """평가 통계 계산"""
        if not results:
            return {}

        total_count = len(results)
        correct_count = sum(1 for r in results if r.is_correct)
        accuracy = correct_count / total_count

        # 카테고리별 정확도
        category_stats = {}
        for result in results:
            category = result.metadata.get('category', 'unknown')
            if category not in category_stats:
                category_stats[category] = {'correct': 0, 'total': 0}

            category_stats[category]['total'] += 1
            if result.is_correct:
                category_stats[category]['correct'] += 1

        for category in category_stats:
            category_stats[category]['accuracy'] = (
                    category_stats[category]['correct'] / category_stats[category]['total']
            )

        # 난이도별 정확도
        difficulty_stats = {}
        for result in results:
            difficulty = result.metadata.get('difficulty', 'unknown')
            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'correct': 0, 'total': 0}

            difficulty_stats[difficulty]['total'] += 1
            if result.is_correct:
                difficulty_stats[difficulty]['correct'] += 1

        for difficulty in difficulty_stats:
            difficulty_stats[difficulty]['accuracy'] = (
                    difficulty_stats[difficulty]['correct'] / difficulty_stats[difficulty]['total']
            )

        # 실행 시간 통계
        execution_times = [r.execution_time for r in results if r.execution_time]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0

        return {
            'total_questions': total_count,
            'correct_answers': correct_count,
            'accuracy': accuracy,
            'category_accuracy': category_stats,
            'difficulty_accuracy': difficulty_stats,
            'total_evaluation_time': total_time,
            'average_execution_time': avg_execution_time,
            'evaluation_date': datetime.now().isoformat()
        }

    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """모델 성능 기록 조회"""
        try:
            sql = f"""
            SELECT COUNT(*) as total,
                   SUM(IS_CORRECT) as correct,
                   AVG(EXECUTION_TIME) as avg_time,
                   MIN(CREATED_AT) as first_eval,
                   MAX(CREATED_AT) as last_eval
            FROM {Constants.TABLE_EVALUATION_RESULTS}
            WHERE MODEL_NAME = :1
            """

            result = self.db_manager.execute_query(sql, [model_name])
            if result and result[0][0] > 0:
                row = result[0]
                return {
                    'model_name': model_name,
                    'total_evaluations': row[0],
                    'correct_answers': row[1],
                    'accuracy': row[1] / row[0] if row[0] > 0 else 0,
                    'average_execution_time': row[2],
                    'first_evaluation': row[3],
                    'last_evaluation': row[4]
                }
            else:
                return {'model_name': model_name, 'message': '평가 기록이 없습니다.'}

        except Exception as e:
            logger.error(f"모델 성능 조회 오류: {e}")
            return {}

    def compare_models(self, model_names: List[str]) -> Dict[str, Any]:
        """여러 모델 성능 비교"""
        comparison = {}

        for model_name in model_names:
            performance = self.get_model_performance(model_name)
            comparison[model_name] = performance

        return comparison

    def save_evaluation_format(self,
                               output_path: str,
                               test_size: int = 100,
                               format_type: str = "json") -> int:
        """LLM 평가용 포맷으로 저장"""
        try:
            eval_data = self.create_evaluation_set(test_size)

            # 평가용 포맷으로 변환
            formatted_data = []
            for item in eval_data:
                formatted_item = {
                    "id": item.id,
                    "category": item.category,
                    "difficulty": item.difficulty,
                    "prompt": self._create_prompt(item),
                    "expected_answer": item.correct_answer,
                    "explanation": item.explanation,
                    "metadata": {
                        "source": item.source,
                        "options": item.options,
                        "created_at": item.created_at
                    }
                }
                formatted_data.append(formatted_item)

            if format_type == "json":
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(formatted_data, f, ensure_ascii=False, indent=2)

            logger.info(f"평가용 데이터셋 저장 완료: {output_path} ({len(formatted_data)}개)")
            return len(formatted_data)

        except Exception as e:
            logger.error(f"평가용 포맷 저장 오류: {e}")
            return 0


# 더미 모델 함수들 (테스트용)
def dummy_model_simple(prompt: str) -> str:
    """간단한 더미 모델 (테스트용)"""
    if "+" in prompt:
        return "계산 결과입니다"
    elif "누구" in prompt or "무엇" in prompt:
        return "모르겠습니다"
    else:
        return "답변"


def dummy_model_random(prompt: str) -> str:
    """랜덤 더미 모델 (테스트용)"""
    import random
    responses = ["A", "B", "C", "D", "예", "아니오", "1", "2", "3"]
    return random.choice(responses)