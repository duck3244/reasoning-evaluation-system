"""
평가 지표 계산 모듈
"""
import re
import math
import json
import logging
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class MetricResult:
    """지표 결과 클래스"""
    name: str
    value: float
    details: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


class BaseMetric:
    """기본 지표 클래스"""

    def __init__(self, name: str):
        self.name = name

    def calculate(self, predictions: List[str], ground_truth: List[str],
                  metadata: List[Dict[str, Any]] = None) -> MetricResult:
        """지표 계산 (하위 클래스에서 구현)"""
        raise NotImplementedError

    def _validate_inputs(self, predictions: List[str], ground_truth: List[str]):
        """입력 데이터 검증"""
        if len(predictions) != len(ground_truth):
            raise ValueError(f"예측값과 정답의 개수가 다릅니다: {len(predictions)} vs {len(ground_truth)}")

        if not predictions:
            raise ValueError("입력 데이터가 비어있습니다")


class AccuracyMetric(BaseMetric):
    """정확도 지표"""

    def __init__(self, case_sensitive: bool = False, normalize: bool = True):
        super().__init__("accuracy")
        self.case_sensitive = case_sensitive
        self.normalize = normalize

    def calculate(self, predictions: List[str], ground_truth: List[str],
                  metadata: List[Dict[str, Any]] = None) -> MetricResult:
        self._validate_inputs(predictions, ground_truth)

        correct_count = 0
        total_count = len(predictions)

        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            if self._is_correct(pred, truth, metadata[i] if metadata else None):
                correct_count += 1

        accuracy = correct_count / total_count if total_count > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=accuracy,
            details={
                'correct_count': correct_count,
                'total_count': total_count,
                'incorrect_count': total_count - correct_count
            }
        )

    def _is_correct(self, prediction: str, ground_truth: str, metadata: Dict[str, Any] = None) -> bool:
        """정답 여부 판단"""
        if not self.case_sensitive:
            prediction = prediction.lower()
            ground_truth = ground_truth.lower()

        if self.normalize:
            prediction = self._normalize_text(prediction)
            ground_truth = self._normalize_text(ground_truth)

        return prediction.strip() == ground_truth.strip()

    def _normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 공백 정리
        text = re.sub(r'\s+', ' ', text)
        # 특수문자 제거 (선택적)
        text = re.sub(r'[^\w\s가-힣]', '', text)
        return text.strip()


class CategoryAccuracyMetric(BaseMetric):
    """카테고리별 정확도 지표"""

    def __init__(self):
        super().__init__("category_accuracy")

    def calculate(self, predictions: List[str], ground_truth: List[str],
                  metadata: List[Dict[str, Any]] = None) -> MetricResult:
        self._validate_inputs(predictions, ground_truth)

        if not metadata:
            raise ValueError("카테고리별 정확도 계산에는 메타데이터가 필요합니다")

        category_stats = {}

        for pred, truth, meta in zip(predictions, ground_truth, metadata):
            category = meta.get('category', 'unknown')

            if category not in category_stats:
                category_stats[category] = {'correct': 0, 'total': 0}

            category_stats[category]['total'] += 1
            if pred.strip().lower() == truth.strip().lower():
                category_stats[category]['correct'] += 1

        # 카테고리별 정확도 계산
        category_accuracies = {}
        overall_correct = 0
        overall_total = 0

        for category, stats in category_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            category_accuracies[category] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }
            overall_correct += stats['correct']
            overall_total += stats['total']

        overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=overall_accuracy,
            details={
                'category_accuracies': category_accuracies,
                'overall_correct': overall_correct,
                'overall_total': overall_total
            }
        )


class DifficultyAccuracyMetric(BaseMetric):
    """난이도별 정확도 지표"""

    def __init__(self):
        super().__init__("difficulty_accuracy")

    def calculate(self, predictions: List[str], ground_truth: List[str],
                  metadata: List[Dict[str, Any]] = None) -> MetricResult:
        self._validate_inputs(predictions, ground_truth)

        if not metadata:
            raise ValueError("난이도별 정확도 계산에는 메타데이터가 필요합니다")

        difficulty_stats = {}

        for pred, truth, meta in zip(predictions, ground_truth, metadata):
            difficulty = meta.get('difficulty', 'unknown')

            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {'correct': 0, 'total': 0}

            difficulty_stats[difficulty]['total'] += 1
            if pred.strip().lower() == truth.strip().lower():
                difficulty_stats[difficulty]['correct'] += 1

        # 난이도별 정확도 계산
        difficulty_accuracies = {}

        for difficulty, stats in difficulty_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
            difficulty_accuracies[difficulty] = {
                'accuracy': accuracy,
                'correct': stats['correct'],
                'total': stats['total']
            }

        # 전체 정확도 계산
        total_correct = sum(stats['correct'] for stats in difficulty_stats.values())
        total_count = sum(stats['total'] for stats in difficulty_stats.values())
        overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

        return MetricResult(
            name=self.name,
            value=overall_accuracy,
            details={
                'difficulty_accuracies': difficulty_accuracies,
                'total_correct': total_correct,
                'total_count': total_count
            }
        )


class MathAccuracyMetric(BaseMetric):
    """수학 문제 전용 정확도 지표"""

    def __init__(self, tolerance: float = 1e-6):
        super().__init__("math_accuracy")
        self.tolerance = tolerance

    def calculate(self, predictions: List[str], ground_truth: List[str],
                  metadata: List[Dict[str, Any]] = None) -> MetricResult:
        self._validate_inputs(predictions, ground_truth)

        correct_count = 0
        numeric_comparisons = 0
        exact_matches = 0

        for pred, truth in zip(predictions, ground_truth):
            if self._is_math_correct(pred, truth):
                correct_count += 1

                # 숫자 비교인지 확인
                if self._contains_number(pred) and self._contains_number(truth):
                    numeric_comparisons += 1
                else:
                    exact_matches += 1

        accuracy = correct_count / len(predictions) if predictions else 0.0

        return MetricResult(
            name=self.name,
            value=accuracy,
            details={
                'correct_count': correct_count,
                'total_count': len(predictions),
                'numeric_comparisons': numeric_comparisons,
                'exact_matches': exact_matches,
                'tolerance': self.tolerance
            }
        )

    def _is_math_correct(self, prediction: str, ground_truth: str) -> bool:
        """수학 정답 여부 판단"""
        # 숫자 추출 시도
        pred_numbers = self._extract_numbers(prediction)
        truth_numbers = self._extract_numbers(ground_truth)

        if pred_numbers and truth_numbers:
            # 첫 번째 숫자 비교
            return abs(pred_numbers[0] - truth_numbers[0]) < self.tolerance

        # 숫자가 없으면 텍스트 비교
        return prediction.strip().lower() == ground_truth.strip().lower()

    def _extract_numbers(self, text: str) -> List[float]:
        """텍스트에서 숫자 추출"""
        # 분수 패턴 (예: 3/4)
        fraction_pattern = r'(\d+)/(\d+)'
        fractions = re.findall(fraction_pattern, text)

        numbers = []
        for numerator, denominator in fractions:
            if int(denominator) != 0:
                numbers.append(float(numerator) / float(denominator))

        # 일반 숫자 패턴
        number_pattern = r'-?\d+\.?\d*'
        number_matches = re.findall(number_pattern, text)

        for match in number_matches:
            try:
                numbers.append(float(match))
            except ValueError:
                continue

        return numbers

    def _contains_number(self, text: str) -> bool:
        """텍스트에 숫자가 포함되어 있는지 확인"""
        return bool(re.search(r'\d', text))


class BleuMetric(BaseMetric):
    """BLEU 점수 지표 (간단한 구현)"""

    def __init__(self, n_gram: int = 4):
        super().__init__("bleu")
        self.n_gram = n_gram

    def calculate(self, predictions: List[str], ground_truth: List[str],
                  metadata: List[Dict[str, Any]] = None) -> MetricResult:
        self._validate_inputs(predictions, ground_truth)

        bleu_scores = []

        for pred, truth in zip(predictions, ground_truth):
            score = self._calculate_bleu_score(pred, truth)
            bleu_scores.append(score)

        avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0

        return MetricResult(
            name=self.name,
            value=avg_bleu,
            details={
                'individual_scores': bleu_scores,
                'n_gram': self.n_gram,
                'min_score': min(bleu_scores) if bleu_scores else 0.0,
                'max_score': max(bleu_scores) if bleu_scores else 0.0
            }
        )

    def _calculate_bleu_score(self, prediction: str, reference: str) -> float:
        """단일 BLEU 점수 계산"""
        pred_tokens = prediction.lower().split()
        ref_tokens = reference.lower().split()

        if not pred_tokens or not ref_tokens:
            return 0.0

        # 1-gram ~ n-gram 정밀도 계산
        precisions = []

        for n in range(1, min(self.n_gram + 1, len(pred_tokens) + 1)):
            pred_ngrams = self._get_ngrams(pred_tokens, n)
            ref_ngrams = self._get_ngrams(ref_tokens, n)

            if not pred_ngrams:
                precisions.append(0.0)
                continue

            matches = 0
            for ngram in pred_ngrams:
                if ngram in ref_ngrams:
                    matches += min(pred_ngrams[ngram], ref_ngrams[ngram])

            precision = matches / sum(pred_ngrams.values())
            precisions.append(precision)

        if not precisions or all(p == 0 for p in precisions):
            return 0.0

        # 기하 평균
        geo_mean = math.exp(sum(math.log(max(p, 1e-10)) for p in precisions) / len(precisions))

        # Brevity penalty
        bp = min(1.0, math.exp(1 - len(ref_tokens) / len(pred_tokens)))

        return bp * geo_mean

    def _get_ngrams(self, tokens: List[str], n: int) -> Dict[tuple, int]:
        """n-gram 딕셔너리 생성"""
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams


class PerformanceMetric(BaseMetric):
    """성능 지표 (실행 시간, 처리량 등)"""

    def __init__(self):
        super().__init__("performance")

    def calculate(self, predictions: List[str], ground_truth: List[str],
                  metadata: List[Dict[str, Any]] = None) -> MetricResult:
        if not metadata:
            raise ValueError("성능 지표 계산에는 메타데이터가 필요합니다")

        execution_times = []
        memory_usages = []

        for meta in metadata:
            if 'execution_time' in meta:
                execution_times.append(meta['execution_time'])
            if 'memory_usage' in meta:
                memory_usages.append(meta['memory_usage'])

        performance_stats = {
            'total_questions': len(predictions),
            'avg_execution_time': statistics.mean(execution_times) if execution_times else 0.0,
            'median_execution_time': statistics.median(execution_times) if execution_times else 0.0,
            'min_execution_time': min(execution_times) if execution_times else 0.0,
            'max_execution_time': max(execution_times) if execution_times else 0.0,
            'total_execution_time': sum(execution_times) if execution_times else 0.0,
        }

        if memory_usages:
            performance_stats.update({
                'avg_memory_usage': statistics.mean(memory_usages),
                'max_memory_usage': max(memory_usages),
                'min_memory_usage': min(memory_usages),
            })

        # 처리량 계산 (questions per second)
        total_time = sum(execution_times) if execution_times else 1.0
        throughput = len(predictions) / total_time

        return MetricResult(
            name=self.name,
            value=throughput,  # 처리량을 주 지표로 사용
            details=performance_stats
        )


class CustomMetric(BaseMetric):
    """사용자 정의 지표"""

    def __init__(self, name: str, calculate_func: Callable):
        super().__init__(name)
        self.calculate_func = calculate_func

    def calculate(self, predictions: List[str], ground_truth: List[str],
                  metadata: List[Dict[str, Any]] = None) -> MetricResult:
        try:
            result = self.calculate_func(predictions, ground_truth, metadata)

            if isinstance(result, (int, float)):
                return MetricResult(name=self.name, value=float(result))
            elif isinstance(result, dict):
                value = result.get('value', 0.0)
                details = result.get('details', {})
                return MetricResult(name=self.name, value=float(value), details=details)
            else:
                raise ValueError("커스텀 지표 함수는 숫자 또는 딕셔너리를 반환해야 합니다")

        except Exception as e:
            logger.error(f"커스텀 지표 '{self.name}' 계산 오류: {e}")
            return MetricResult(name=self.name, value=0.0, details={'error': str(e)})


class MetricsCalculator:
    """지표 계산기"""

    def __init__(self):
        self.metrics = {}
        self._register_default_metrics()

    def _register_default_metrics(self):
        """기본 지표들 등록"""
        self.register_metric(AccuracyMetric())
        self.register_metric(CategoryAccuracyMetric())
        self.register_metric(DifficultyAccuracyMetric())
        self.register_metric(MathAccuracyMetric())
        self.register_metric(BleuMetric())
        self.register_metric(PerformanceMetric())

    def register_metric(self, metric: BaseMetric):
        """지표 등록"""
        self.metrics[metric.name] = metric
        logger.info(f"지표 등록: {metric.name}")

    def register_custom_metric(self, name: str, calculate_func: Callable):
        """커스텀 지표 등록"""
        metric = CustomMetric(name, calculate_func)
        self.register_metric(metric)

    def calculate_all(self, predictions: List[str], ground_truth: List[str],
                      metadata: List[Dict[str, Any]] = None,
                      selected_metrics: List[str] = None) -> Dict[str, MetricResult]:
        """모든 지표 계산"""
        if selected_metrics is None:
            selected_metrics = list(self.metrics.keys())

        results = {}

        for metric_name in selected_metrics:
            if metric_name not in self.metrics:
                logger.warning(f"등록되지 않은 지표: {metric_name}")
                continue

            try:
                metric = self.metrics[metric_name]
                result = metric.calculate(predictions, ground_truth, metadata)
                results[metric_name] = result

                logger.debug(f"지표 계산 완료: {metric_name} = {result.value:.4f}")

            except Exception as e:
                logger.error(f"지표 '{metric_name}' 계산 오류: {e}")
                results[metric_name] = MetricResult(
                    name=metric_name,
                    value=0.0,
                    details={'error': str(e)}
                )

        return results

    def calculate_single(self, metric_name: str, predictions: List[str],
                         ground_truth: List[str],
                         metadata: List[Dict[str, Any]] = None) -> MetricResult:
        """단일 지표 계산"""
        if metric_name not in self.metrics:
            raise ValueError(f"등록되지 않은 지표: {metric_name}")

        metric = self.metrics[metric_name]
        return metric.calculate(predictions, ground_truth, metadata)

    def get_metric_names(self) -> List[str]:
        """등록된 지표 이름 목록"""
        return list(self.metrics.keys())

    def remove_metric(self, metric_name: str):
        """지표 제거"""
        if metric_name in self.metrics:
            del self.metrics[metric_name]
            logger.info(f"지표 제거: {metric_name}")


class MetricsReport:
    """지표 리포트 생성기"""

    @staticmethod
    def generate_summary(results: Dict[str, MetricResult]) -> Dict[str, Any]:
        """지표 요약 생성"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'metrics_count': len(results),
            'primary_metrics': {},
            'detailed_results': {}
        }

        # 주요 지표 추출
        primary_metrics = ['accuracy', 'category_accuracy', 'difficulty_accuracy', 'performance']

        for metric_name in primary_metrics:
            if metric_name in results:
                result = results[metric_name]
                summary['primary_metrics'][metric_name] = {
                    'value': result.value,
                    'formatted': f"{result.value:.4f}" if isinstance(result.value, float) else str(result.value)
                }

        # 상세 결과
        for metric_name, result in results.items():
            summary['detailed_results'][metric_name] = {
                'value': result.value,
                'details': result.details,
                'metadata': result.metadata
            }

        return summary

    @staticmethod
    def format_for_display(results: Dict[str, MetricResult]) -> str:
        """디스플레이용 포맷팅"""
        lines = []
        lines.append("=" * 60)
        lines.append("평가 결과 요약")
        lines.append("=" * 60)

        # 주요 지표
        if 'accuracy' in results:
            acc = results['accuracy']
            lines.append(f"전체 정확도: {acc.value:.2%}")
            if acc.details:
                lines.append(f"  정답: {acc.details.get('correct_count', 0)}")
                lines.append(f"  전체: {acc.details.get('total_count', 0)}")

        # 카테고리별 정확도
        if 'category_accuracy' in results:
            cat_acc = results['category_accuracy']
            if cat_acc.details and 'category_accuracies' in cat_acc.details:
                lines.append("\n카테고리별 정확도:")
                for category, stats in cat_acc.details['category_accuracies'].items():
                    lines.append(f"  {category}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

        # 난이도별 정확도
        if 'difficulty_accuracy' in results:
            diff_acc = results['difficulty_accuracy']
            if diff_acc.details and 'difficulty_accuracies' in diff_acc.details:
                lines.append("\n난이도별 정확도:")
                for difficulty, stats in diff_acc.details['difficulty_accuracies'].items():
                    lines.append(f"  {difficulty}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

        # 성능 지표
        if 'performance' in results:
            perf = results['performance']
            if perf.details:
                lines.append("\n성능 지표:")
                lines.append(f"  처리량: {perf.value:.2f} questions/sec")
                lines.append(f"  평균 실행 시간: {perf.details.get('avg_execution_time', 0):.3f}초")
                lines.append(f"  총 실행 시간: {perf.details.get('total_execution_time', 0):.3f}초")

        # 기타 지표들
        other_metrics = [name for name in results.keys()
                         if name not in ['accuracy', 'category_accuracy', 'difficulty_accuracy', 'performance']]

        if other_metrics:
            lines.append("\n기타 지표:")
            for metric_name in other_metrics:
                result = results[metric_name]
                lines.append(f"  {metric_name}: {result.value:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)

    @staticmethod
    def export_to_json(results: Dict[str, MetricResult], file_path: str) -> bool:
        """JSON 파일로 내보내기"""
        try:
            import json

            export_data = {
                'timestamp': datetime.now().isoformat(),
                'results': {}
            }

            for metric_name, result in results.items():
                export_data['results'][metric_name] = {
                    'name': result.name,
                    'value': result.value,
                    'details': result.details,
                    'metadata': result.metadata
                }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"지표 결과를 JSON으로 내보냄: {file_path}")
            return True

        except Exception as e:
            logger.error(f"JSON 내보내기 실패: {e}")
            return False

    @staticmethod
    def export_to_csv(results: Dict[str, MetricResult], file_path: str) -> bool:
        """CSV 파일로 내보내기"""
        try:
            import csv

            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                # 헤더
                writer.writerow(['Metric', 'Value', 'Details'])

                # 데이터
                for metric_name, result in results.items():
                    details_str = json.dumps(result.details) if result.details else ""
                    writer.writerow([result.name, result.value, details_str])

            logger.info(f"지표 결과를 CSV로 내보냄: {file_path}")
            return True

        except Exception as e:
            logger.error(f"CSV 내보내기 실패: {e}")
            return False


# 유틸리티 함수들
def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    """신뢰구간 계산"""
    if not values:
        return (0.0, 0.0)

    import scipy.stats as stats

    mean = statistics.mean(values)
    sem = stats.sem(values)  # 표준오차
    interval = stats.t.interval(confidence, len(values) - 1, loc=mean, scale=sem)

    return interval


def calculate_statistical_significance(group1: List[float], group2: List[float]) -> Dict[str, Any]:
    """통계적 유의성 검정"""
    try:
        import scipy.stats as stats

        # t-검정
        t_stat, p_value = stats.ttest_ind(group1, group2)

        # 효과 크기 (Cohen's d)
        pooled_std = math.sqrt(((len(group1) - 1) * statistics.variance(group1) +
                                (len(group2) - 1) * statistics.variance(group2)) /
                               (len(group1) + len(group2) - 2))

        cohens_d = (statistics.mean(group1) - statistics.mean(group2)) / pooled_std if pooled_std > 0 else 0

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'cohens_d': cohens_d,
            'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
        }

    except ImportError:
        logger.warning("scipy가 설치되지 않아 통계적 유의성 검정을 수행할 수 없습니다")
        return {'error': 'scipy not available'}
    except Exception as e:
        logger.error(f"통계적 유의성 검정 오류: {e}")
        return {'error': str(e)}


# 사용 예시 및 테스트
if __name__ == "__main__":
    # 샘플 데이터
    predictions = ["4", "답: 파리", "예", "3.14", "물"]
    ground_truth = ["4", "파리", "예", "3.14159", "H2O"]
    metadata = [
        {"category": "math", "difficulty": "easy", "execution_time": 0.5},
        {"category": "common_sense", "difficulty": "easy", "execution_time": 0.3},
        {"category": "logic", "difficulty": "medium", "execution_time": 0.8},
        {"category": "math", "difficulty": "hard", "execution_time": 1.2},
        {"category": "science", "difficulty": "medium", "execution_time": 0.6}
    ]

    # 지표 계산기 생성
    calculator = MetricsCalculator()

    # 모든 지표 계산
    results = calculator.calculate_all(predictions, ground_truth, metadata)

    # 결과 출력
    print(MetricsReport.format_for_display(results))


    # 커스텀 지표 등록 예시
    def korean_accuracy(predictions, ground_truth, metadata):
        """한국어 특화 정확도"""
        correct = 0
        for pred, truth in zip(predictions, ground_truth):
            # 한국어 특화 비교 로직
            if pred.replace(" ", "") == truth.replace(" ", ""):
                correct += 1
        return correct / len(predictions)


    calculator.register_custom_metric("korean_accuracy", korean_accuracy)

    # 커스텀 지표 계산
    korean_result = calculator.calculate_single("korean_accuracy", predictions, ground_truth, metadata)
    print(f"\n한국어 정확도: {korean_result.value:.2%}")

    # JSON으로 내보내기
    MetricsReport.export_to_json(results, "metrics_results.json")

    print("\n지표 계산 완료!")