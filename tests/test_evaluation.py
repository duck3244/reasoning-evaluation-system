"""
평가 시스템 단위 테스트
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import time
from datetime import datetime

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_models import ReasoningDataPoint, EvaluationResult
from evaluation.evaluation_system import ReasoningEvaluator
from evaluation.metrics import (
    AccuracyMetric, CategoryAccuracyMetric, DifficultyAccuracyMetric,
    MathAccuracyMetric, BleuMetric, PerformanceMetric, MetricsCalculator,
    MetricsReport, MetricResult
)


class TestReasoningEvaluator(unittest.TestCase):
    """ReasoningEvaluator 단위 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.mock_db_manager = Mock()
        self.mock_collector = Mock()
        self.evaluator = ReasoningEvaluator(self.mock_db_manager, self.mock_collector)

        # 샘플 테스트 데이터
        self.sample_data = [
            ReasoningDataPoint(
                id="test_1",
                category="math",
                difficulty="easy",
                question="2 + 2 = ?",
                correct_answer="4",
                source="test"
            ),
            ReasoningDataPoint(
                id="test_2",
                category="logic",
                difficulty="medium",
                question="참 또는 거짓?",
                correct_answer="참",
                source="test"
            ),
            ReasoningDataPoint(
                id="test_3",
                category="science",
                difficulty="hard",
                question="물의 화학식은?",
                correct_answer="H2O",
                source="test"
            )
        ]

    def test_create_evaluation_set_balanced(self):
        """균형잡힌 평가 데이터셋 생성 테스트"""
        # Mock 데이터 설정
        self.mock_collector.get_data.return_value = self.sample_data

        # 테스트 실행
        eval_set = self.evaluator.create_evaluation_set(
            test_size=3,
            balance_categories=True,
            balance_difficulties=True
        )

        # 검증
        self.assertEqual(len(eval_set), 3)
        self.mock_collector.get_data.assert_called()

    def test_create_evaluation_set_simple(self):
        """단순 평가 데이터셋 생성 테스트"""
        # Mock 데이터 설정
        self.mock_collector.get_data.return_value = self.sample_data

        # 테스트 실행
        eval_set = self.evaluator.create_evaluation_set(
            test_size=2,
            balance_categories=False
        )

        # 검증
        self.assertLessEqual(len(eval_set), 2)

    def test_evaluate_model_success(self):
        """모델 평가 성공 테스트"""

        # 간단한 모델 함수
        def simple_model(prompt):
            if "2 + 2" in prompt:
                return "4"
            elif "참 또는 거짓" in prompt:
                return "참"
            else:
                return "모름"

        # 평가 실행
        results = self.evaluator.evaluate_model(
            model_name="test_model",
            evaluation_set=self.sample_data[:2],  # 첫 두 문제만 사용
            model_function=simple_model,
            save_results=False
        )

        # 결과 검증
        self.assertIn('accuracy', results)
        self.assertIn('total_questions', results)
        self.assertIn('correct_answers', results)
        self.assertEqual(results['total_questions'], 2)
        self.assertEqual(results['correct_answers'], 2)
        self.assertEqual(results['accuracy'], 1.0)

    def test_evaluate_model_with_errors(self):
        """모델 평가 중 오류 발생 테스트"""

        # 오류를 발생시키는 모델 함수
        def error_model(prompt):
            if "2 + 2" in prompt:
                raise Exception("Model error")
            return "답변"

        # 평가 실행
        results = self.evaluator.evaluate_model(
            model_name="error_model",
            evaluation_set=self.sample_data,
            model_function=error_model,
            save_results=False
        )

        # 결과 검증 (오류가 발생해도 시스템이 계속 작동해야 함)
        self.assertIn('accuracy', results)
        self.assertGreaterEqual(results['total_questions'], 0)

    def test_check_answer_exact_match(self):
        """정확한 일치 답변 확인 테스트"""
        result = self.evaluator._check_answer("4", "4", "math")
        self.assertTrue(result)

        result = self.evaluator._check_answer("참", "참", "logic")
        self.assertTrue(result)

        result = self.evaluator._check_answer("4", "5", "math")
        self.assertFalse(result)

    def test_check_answer_case_insensitive(self):
        """대소문자 무시 답변 확인 테스트"""
        result = self.evaluator._check_answer("TRUE", "true", "logic")
        self.assertTrue(result)

        result = self.evaluator._check_answer("H2O", "h2o", "science")
        self.assertTrue(result)

    def test_check_math_answer(self):
        """수학 답변 확인 테스트"""
        # 정확한 숫자 일치
        result = self.evaluator._check_math_answer("3.14", "3.14")
        self.assertTrue(result)

        # 오차 범위 내 일치
        result = self.evaluator._check_math_answer("3.141", "3.14159")
        self.assertTrue(result)

        # 다른 숫자
        result = self.evaluator._check_math_answer("2", "3")
        self.assertFalse(result)

        # 텍스트 답변
        result = self.evaluator._check_math_answer("답: 5", "5")
        self.assertTrue(result)

    def test_check_logic_answer(self):
        """논리 답변 확인 테스트"""
        # 예/아니오 패턴
        result = self.evaluator._check_logic_answer("예", "참")
        self.assertTrue(result)

        result = self.evaluator._check_logic_answer("아니오", "거짓")
        self.assertTrue(result)

        result = self.evaluator._check_logic_answer("yes", "true")
        self.assertTrue(result)

        # 반대 답변
        result = self.evaluator._check_logic_answer("예", "아니오")
        self.assertFalse(result)

    def test_create_prompt(self):
        """프롬프트 생성 테스트"""
        # 객관식 문제
        data_point = ReasoningDataPoint(
            id="test",
            category="math",
            difficulty="easy",
            question="2 + 2 = ?",
            correct_answer="4",
            options=["2", "3", "4", "5"]
        )

        prompt = self.evaluator._create_prompt(data_point)
        self.assertIn("2 + 2 = ?", prompt)
        self.assertIn("A)", prompt)
        self.assertIn("B)", prompt)
        self.assertIn("선택지:", prompt)

        # 주관식 문제
        data_point_subjective = ReasoningDataPoint(
            id="test",
            category="math",
            difficulty="easy",
            question="2 + 2 = ?",
            correct_answer="4"
        )

        prompt = self.evaluator._create_prompt(data_point_subjective)
        self.assertIn("2 + 2 = ?", prompt)
        self.assertNotIn("선택지:", prompt)

    def test_calculate_statistics(self):
        """통계 계산 테스트"""
        # 샘플 평가 결과
        results = [
            EvaluationResult(
                id="eval_1",
                data_point_id="test_1",
                model_name="test_model",
                predicted_answer="4",
                is_correct=True,
                execution_time=0.5,
                metadata={"category": "math", "difficulty": "easy"}
            ),
            EvaluationResult(
                id="eval_2",
                data_point_id="test_2",
                model_name="test_model",
                predicted_answer="참",
                is_correct=True,
                execution_time=0.3,
                metadata={"category": "logic", "difficulty": "medium"}
            ),
            EvaluationResult(
                id="eval_3",
                data_point_id="test_3",
                model_name="test_model",
                predicted_answer="H2O",
                is_correct=False,
                execution_time=0.8,
                metadata={"category": "science", "difficulty": "hard"}
            )
        ]

        # 통계 계산
        stats = self.evaluator._calculate_statistics(results, total_time=5.0)

        # 검증
        self.assertEqual(stats['total_questions'], 3)
        self.assertEqual(stats['correct_answers'], 2)
        self.assertAlmostEqual(stats['accuracy'], 2 / 3)
        self.assertIn('category_accuracy', stats)
        self.assertIn('difficulty_accuracy', stats)
        self.assertEqual(stats['total_evaluation_time'], 5.0)

    def test_save_evaluation_results(self):
        """평가 결과 저장 테스트"""
        # Mock 설정
        self.mock_db_manager.execute_batch_dml.return_value = 2

        # 샘플 결과
        results = [
            EvaluationResult(
                id="eval_1",
                data_point_id="test_1",
                model_name="test_model",
                predicted_answer="4",
                is_correct=True,
                execution_time=0.5
            ),
            EvaluationResult(
                id="eval_2",
                data_point_id="test_2",
                model_name="test_model",
                predicted_answer="참",
                is_correct=False,
                execution_time=0.3
            )
        ]

        # 저장 실행
        success = self.evaluator._save_evaluation_results(results)

        # 검증
        self.assertTrue(success)
        self.mock_db_manager.execute_batch_dml.assert_called_once()

    def test_get_model_performance(self):
        """모델 성능 조회 테스트"""
        # Mock 데이터 설정
        self.mock_db_manager.execute_query.return_value = [
            (100, 85, 1.25, datetime.now(), datetime.now())
        ]

        # 성능 조회
        performance = self.evaluator.get_model_performance("test_model")

        # 검증
        self.assertIn('model_name', performance)
        self.assertIn('total_evaluations', performance)
        self.assertIn('accuracy', performance)
        self.assertEqual(performance['total_evaluations'], 100)
        self.assertEqual(performance['correct_answers'], 85)
        self.assertAlmostEqual(performance['accuracy'], 0.85)

    def test_get_model_performance_no_data(self):
        """데이터가 없는 모델 성능 조회 테스트"""
        # Mock 데이터 설정 (결과 없음)
        self.mock_db_manager.execute_query.return_value = [(0, 0, 0, None, None)]

        # 성능 조회
        performance = self.evaluator.get_model_performance("nonexistent_model")

        # 검증
        self.assertIn('message', performance)

    def test_compare_models(self):
        """모델 비교 테스트"""

        # Mock 설정
        def mock_get_performance(model_name):
            if model_name == "model_1":
                return {'accuracy': 0.85, 'total_evaluations': 100}
            elif model_name == "model_2":
                return {'accuracy': 0.90, 'total_evaluations': 100}
            else:
                return {'message': '평가 기록이 없습니다.'}

        # Mock 함수 패치
        with patch.object(self.evaluator, 'get_model_performance', side_effect=mock_get_performance):
            comparison = self.evaluator.compare_models(["model_1", "model_2", "model_3"])

        # 검증
        self.assertIn('model_1', comparison)
        self.assertIn('model_2', comparison)
        self.assertIn('model_3', comparison)
        self.assertEqual(comparison['model_1']['accuracy'], 0.85)
        self.assertEqual(comparison['model_2']['accuracy'], 0.90)

    def test_save_evaluation_format(self):
        """평가용 포맷 저장 테스트"""
        # Mock 설정
        self.mock_collector.get_data.return_value = self.sample_data

        # 파일 쓰기 모킹
        with patch('builtins.open', create=True) as mock_open:
            with patch('json.dump') as mock_json_dump:
                # 테스트 실행
                count = self.evaluator.save_evaluation_format("test_eval.json", test_size=3)

                # 검증
                self.assertGreater(count, 0)
                mock_open.assert_called_once()
                mock_json_dump.assert_called_once()


class TestMetrics(unittest.TestCase):
    """지표 계산 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.predictions = ["4", "파리", "예", "3.14", "H2O"]
        self.ground_truth = ["4", "파리", "예", "3.14159", "H2O"]
        self.metadata = [
            {"category": "math", "difficulty": "easy"},
            {"category": "common_sense", "difficulty": "easy"},
            {"category": "logic", "difficulty": "medium"},
            {"category": "math", "difficulty": "hard"},
            {"category": "science", "difficulty": "medium"}
        ]

    def test_accuracy_metric(self):
        """정확도 지표 테스트"""
        metric = AccuracyMetric()
        result = metric.calculate(self.predictions, self.ground_truth)

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "accuracy")
        self.assertAlmostEqual(result.value, 0.8)  # 4/5 정답
        self.assertIn('correct_count', result.details)
        self.assertEqual(result.details['correct_count'], 4)

    def test_accuracy_metric_case_insensitive(self):
        """대소문자 무시 정확도 테스트"""
        predictions = ["TRUE", "False", "YES"]
        ground_truth = ["true", "false", "yes"]

        metric = AccuracyMetric(case_sensitive=False)
        result = metric.calculate(predictions, ground_truth)

        self.assertEqual(result.value, 1.0)

    def test_category_accuracy_metric(self):
        """카테고리별 정확도 테스트"""
        metric = CategoryAccuracyMetric()
        result = metric.calculate(self.predictions, self.ground_truth, self.metadata)

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "category_accuracy")
        self.assertIn('category_accuracies', result.details)

        # 각 카테고리별 정확도 확인
        cat_acc = result.details['category_accuracies']
        self.assertIn('math', cat_acc)
        self.assertIn('common_sense', cat_acc)
        self.assertIn('logic', cat_acc)
        self.assertIn('science', cat_acc)

    def test_difficulty_accuracy_metric(self):
        """난이도별 정확도 테스트"""
        metric = DifficultyAccuracyMetric()
        result = metric.calculate(self.predictions, self.ground_truth, self.metadata)

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "difficulty_accuracy")
        self.assertIn('difficulty_accuracies', result.details)

        # 각 난이도별 정확도 확인
        diff_acc = result.details['difficulty_accuracies']
        self.assertIn('easy', diff_acc)
        self.assertIn('medium', diff_acc)
        self.assertIn('hard', diff_acc)

    def test_math_accuracy_metric(self):
        """수학 정확도 테스트"""
        math_predictions = ["3.14", "2", "5.0", "1/2"]
        math_ground_truth = ["3.14159", "2.0", "5", "0.5"]

        metric = MathAccuracyMetric(tolerance=0.01)
        result = metric.calculate(math_predictions, math_ground_truth)

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "math_accuracy")
        self.assertEqual(result.value, 1.0)  # 모든 답이 허용 오차 내

    def test_bleu_metric(self):
        """BLEU 점수 테스트"""
        predictions = ["the cat sat on the mat", "hello world"]
        ground_truth = ["the cat is on the mat", "hello world"]

        metric = BleuMetric(n_gram=2)
        result = metric.calculate(predictions, ground_truth)

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "bleu")
        self.assertGreaterEqual(result.value, 0.0)
        self.assertLessEqual(result.value, 1.0)

    def test_performance_metric(self):
        """성능 지표 테스트"""
        predictions = ["answer1", "answer2", "answer3"]
        ground_truth = ["truth1", "truth2", "truth3"]
        metadata = [
            {"execution_time": 0.5, "memory_usage": 100},
            {"execution_time": 0.3, "memory_usage": 150},
            {"execution_time": 0.8, "memory_usage": 120}
        ]

        metric = PerformanceMetric()
        result = metric.calculate(predictions, ground_truth, metadata)

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "performance")
        self.assertIn('avg_execution_time', result.details)
        self.assertIn('total_execution_time', result.details)
        self.assertAlmostEqual(result.details['avg_execution_time'], 0.533, places=2)

    def test_metrics_calculator(self):
        """지표 계산기 테스트"""
        calculator = MetricsCalculator()

        # 등록된 지표 확인
        metric_names = calculator.get_metric_names()
        self.assertIn('accuracy', metric_names)
        self.assertIn('category_accuracy', metric_names)

        # 모든 지표 계산
        results = calculator.calculate_all(self.predictions, self.ground_truth, self.metadata)

        self.assertIsInstance(results, dict)
        self.assertIn('accuracy', results)
        self.assertIn('category_accuracy', results)

        # 각 결과가 MetricResult 인스턴스인지 확인
        for result in results.values():
            self.assertIsInstance(result, MetricResult)

    def test_custom_metric(self):
        """커스텀 지표 테스트"""
        calculator = MetricsCalculator()

        # 커스텀 지표 함수
        def length_difference_metric(predictions, ground_truth, metadata):
            total_diff = sum(abs(len(p) - len(t)) for p, t in zip(predictions, ground_truth))
            return total_diff / len(predictions)

        # 커스텀 지표 등록
        calculator.register_custom_metric("length_diff", length_difference_metric)

        # 계산
        result = calculator.calculate_single("length_diff", self.predictions, self.ground_truth)

        self.assertIsInstance(result, MetricResult)
        self.assertEqual(result.name, "length_diff")
        self.assertGreaterEqual(result.value, 0.0)

    def test_metrics_report_generation(self):
        """지표 리포트 생성 테스트"""
        calculator = MetricsCalculator()
        results = calculator.calculate_all(self.predictions, self.ground_truth, self.metadata)

        # 요약 생성
        summary = MetricsReport.generate_summary(results)

        self.assertIn('timestamp', summary)
        self.assertIn('metrics_count', summary)
        self.assertIn('primary_metrics', summary)
        self.assertIn('detailed_results', summary)

        # 디스플레이 포맷 생성
        display_text = MetricsReport.format_for_display(results)

        self.assertIsInstance(display_text, str)
        self.assertIn("평가 결과 요약", display_text)
        self.assertIn("정확도", display_text)

    def test_metrics_export(self):
        """지표 내보내기 테스트"""
        calculator = MetricsCalculator()
        results = calculator.calculate_all(self.predictions, self.ground_truth, self.metadata)

        # JSON 내보내기 테스트
        with patch('builtins.open', create=True):
            with patch('json.dump') as mock_dump:
                success = MetricsReport.export_to_json(results, "test_metrics.json")
                self.assertTrue(success)
                mock_dump.assert_called_once()

        # CSV 내보내기 테스트
        with patch('builtins.open', create=True):
            with patch('csv.writer') as mock_writer:
                mock_writer_instance = Mock()
                mock_writer.return_value = mock_writer_instance

                success = MetricsReport.export_to_csv(results, "test_metrics.csv")
                self.assertTrue(success)
                mock_writer_instance.writerow.assert_called()

    def test_metric_validation(self):
        """지표 입력 검증 테스트"""
        metric = AccuracyMetric()

        # 길이가 다른 입력
        with self.assertRaises(ValueError):
            metric.calculate(["a", "b"], ["a"])

        # 빈 입력
        with self.assertRaises(ValueError):
            metric.calculate([], [])

    def test_statistical_functions(self):
        """통계 함수 테스트"""
        from evaluation.metrics import calculate_confidence_interval, calculate_statistical_significance

        # 신뢰구간 계산 테스트
        values = [0.8, 0.85, 0.9, 0.75, 0.88]
        try:
            interval = calculate_confidence_interval(values)
            self.assertIsInstance(interval, tuple)
            self.assertEqual(len(interval), 2)
            self.assertLessEqual(interval[0], interval[1])
        except ImportError:
            # scipy가 없는 경우 패스
            pass

        # 통계적 유의성 검정 테스트
        group1 = [0.8, 0.85, 0.9, 0.75, 0.88]
        group2 = [0.7, 0.75, 0.8, 0.65, 0.78]

        try:
            sig_test = calculate_statistical_significance(group1, group2)
            self.assertIn('p_value', sig_test)
            self.assertIn('significant', sig_test)
            self.assertIsInstance(sig_test['significant'], bool)
        except ImportError:
            # scipy가 없는 경우 패스
            pass


class TestEvaluationIntegration(unittest.TestCase):
    """평가 시스템 통합 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.mock_db_manager = Mock()
        self.mock_collector = Mock()
        self.evaluator = ReasoningEvaluator(self.mock_db_manager, self.mock_collector)

        # 복합 테스트 데이터
        self.complex_data = [
            ReasoningDataPoint(
                id="complex_1",
                category="math",
                difficulty="easy",
                question="15 + 27 = ?",
                correct_answer="42",
                explanation="15와 27을 더하면 42입니다.",
                source="test"
            ),
            ReasoningDataPoint(
                id="complex_2",
                category="logic",
                difficulty="medium",
                question="모든 새는 날 수 있다. 참새는 새다. 따라서 참새는 날 수 있다. 이 논리가 타당한가?",
                correct_answer="예",
                explanation="올바른 삼단논법입니다.",
                source="test"
            ),
            ReasoningDataPoint(
                id="complex_3",
                category="science",
                difficulty="hard",
                question="빛의 속도는 초당 몇 km인가?",
                correct_answer="300000",
                explanation="진공에서 빛의 속도는 약 30만 km/s입니다.",
                source="test"
            ),
            ReasoningDataPoint(
                id="complex_4",
                category="common_sense",
                difficulty="easy",
                question="비가 올 때 사용하는 것은?",
                correct_answer="우산",
                options=["우산", "선글라스", "모자", "장갑"],
                source="test"
            )
        ]

    def test_full_evaluation_pipeline(self):
        """전체 평가 파이프라인 테스트"""

        # 완전한 모델 함수
        def comprehensive_model(prompt):
            if "15 + 27" in prompt:
                return "42"
            elif "삼단논법" in prompt or "논리가 타당" in prompt:
                return "예"
            elif "빛의 속도" in prompt:
                return "300000 km/s"
            elif "비가 올 때" in prompt:
                return "우산"
            else:
                return "모르겠습니다"

        # Mock 설정
        self.mock_collector.get_data.return_value = self.complex_data

        # 평가 데이터셋 생성
        eval_set = self.evaluator.create_evaluation_set(test_size=4)
        self.assertEqual(len(eval_set), 4)

        # 모델 평가 실행
        results = self.evaluator.evaluate_model(
            model_name="comprehensive_model",
            evaluation_set=eval_set,
            model_function=comprehensive_model,
            save_results=False
        )

        # 결과 검증
        self.assertIn('accuracy', results)
        self.assertIn('category_accuracy', results)
        self.assertIn('difficulty_accuracy', results)
        self.assertEqual(results['total_questions'], 4)
        self.assertGreaterEqual(results['accuracy'], 0.75)  # 최소 75% 정확도 기대

        # 카테고리별 성능 확인
        cat_acc = results['category_accuracy']
        self.assertIn('math', cat_acc)
        self.assertIn('logic', cat_acc)
        self.assertIn('science', cat_acc)
        self.assertIn('common_sense', cat_acc)

    def test_evaluation_with_timeout(self):
        """타임아웃을 포함한 평가 테스트"""

        # 느린 모델 함수
        def slow_model(prompt):
            time.sleep(0.1)  # 0.1초 지연
            return "답변"

        # 평가 실행
        start_time = time.time()
        results = self.evaluator.evaluate_model(
            model_name="slow_model",
            evaluation_set=self.complex_data[:2],  # 2문제만 사용
            model_function=slow_model,
            save_results=False
        )
        end_time = time.time()

        # 실행 시간 확인
        total_time = end_time - start_time
        self.assertGreater(total_time, 0.2)  # 최소 0.2초 (2문제 × 0.1초)

        # 평균 실행 시간 확인
        self.assertGreater(results['average_execution_time'], 0.1)

    def test_evaluation_error_handling(self):
        """평가 중 오류 처리 테스트"""

        # 간헐적으로 오류를 발생시키는 모델
        def unreliable_model(prompt):
            if "빛의 속도" in prompt:
                raise Exception("Network timeout")
            return "답변"

        # 평가 실행 (오류가 발생해도 계속 진행되어야 함)
        results = self.evaluator.evaluate_model(
            model_name="unreliable_model",
            evaluation_set=self.complex_data,
            model_function=unreliable_model,
            save_results=False
        )

        # 결과 검증
        self.assertIn('accuracy', results)
        self.assertLess(results['total_questions'], len(self.complex_data))  # 일부 문제는 실패

    def test_multilingual_evaluation(self):
        """다국어 평가 테스트"""
        multilingual_data = [
            ReasoningDataPoint(
                id="ko_1",
                category="math",
                difficulty="easy",
                question="삼 더하기 다섯은?",
                correct_answer="팔",
                source="korean"
            ),
            ReasoningDataPoint(
                id="en_1",
                category="math",
                difficulty="easy",
                question="Three plus five equals?",
                correct_answer="eight",
                source="english"
            )
        ]

        def multilingual_model(prompt):
            if "삼 더하기 다섯" in prompt:
                return "팔"
            elif "Three plus five" in prompt:
                return "eight"
            return "unknown"

        # 평가 실행
        results = self.evaluator.evaluate_model(
            model_name="multilingual_model",
            evaluation_set=multilingual_data,
            model_function=multilingual_model,
            save_results=False
        )

        # 결과 검증
        self.assertEqual(results['accuracy'], 1.0)

    def test_batch_evaluation(self):
        """배치 평가 테스트"""
        # 여러 모델 동시 평가 시뮬레이션
        models = {
            "simple_model": lambda p: "답변",
            "smart_model": lambda p: "42" if "15 + 27" in p else "답변",
            "random_model": lambda p: "랜덤"
        }

        results_comparison = {}

        for model_name, model_func in models.items():
            results = self.evaluator.evaluate_model(
                model_name=model_name,
                evaluation_set=self.complex_data[:2],
                model_function=model_func,
                save_results=False
            )
            results_comparison[model_name] = results

        # 결과 비교
        self.assertEqual(len(results_comparison), 3)
        self.assertIn('simple_model', results_comparison)
        self.assertIn('smart_model', results_comparison)
        self.assertIn('random_model', results_comparison)

        # smart_model이 더 좋은 성능을 보여야 함
        smart_acc = results_comparison['smart_model']['accuracy']
        simple_acc = results_comparison['simple_model']['accuracy']
        self.assertGreaterEqual(smart_acc, simple_acc)

    def test_evaluation_with_custom_metrics(self):
        """커스텀 지표를 포함한 평가 테스트"""
        from evaluation.metrics import MetricsCalculator

        # 커스텀 지표 함수
        def korean_specific_accuracy(predictions, ground_truth, metadata):
            """한국어 특화 정확도"""
            correct = 0
            for pred, truth, meta in zip(predictions, ground_truth, metadata):
                # 한국어 문제만 고려
                if meta and meta.get('source') == 'korean':
                    if pred.replace(" ", "") == truth.replace(" ", ""):
                        correct += 1
            korean_count = sum(1 for meta in metadata if meta and meta.get('source') == 'korean')
            return correct / korean_count if korean_count > 0 else 0.0

        # 지표 계산기 설정
        calculator = MetricsCalculator()
        calculator.register_custom_metric("korean_accuracy", korean_specific_accuracy)

        # 테스트 데이터와 결과
        predictions = ["팔", "eight", "답변", "우산"]
        ground_truth = ["팔", "eight", "모름", "우산"]
        metadata = [
            {"source": "korean", "category": "math"},
            {"source": "english", "category": "math"},
            {"source": "korean", "category": "logic"},
            {"source": "korean", "category": "common_sense"}
        ]

        # 모든 지표 계산
        results = calculator.calculate_all(predictions, ground_truth, metadata)

        # 커스텀 지표 확인
        self.assertIn('korean_accuracy', results)
        korean_result = results['korean_accuracy']
        self.assertGreaterEqual(korean_result.value, 0.0)
        self.assertLessEqual(korean_result.value, 1.0)

    def test_performance_benchmarking(self):
        """성능 벤치마킹 테스트"""

        # 다양한 성능을 가진 모델들
        def fast_model(prompt):
            return "빠른답변"

        def slow_model(prompt):
            time.sleep(0.05)  # 50ms 지연
            return "느린답변"

        # 성능 측정
        fast_results = self.evaluator.evaluate_model(
            model_name="fast_model",
            evaluation_set=self.complex_data[:3],
            model_function=fast_model,
            save_results=False
        )

        slow_results = self.evaluator.evaluate_model(
            model_name="slow_model",
            evaluation_set=self.complex_data[:3],
            model_function=slow_model,
            save_results=False
        )

        # 성능 비교
        fast_time = fast_results['average_execution_time']
        slow_time = slow_results['average_execution_time']

        self.assertLess(fast_time, slow_time)
        self.assertGreater(slow_time, 0.04)  # 최소 40ms 이상

    def test_evaluation_result_persistence(self):
        """평가 결과 영속성 테스트"""
        # Mock DB 설정
        self.mock_db_manager.execute_batch_dml.return_value = 4

        def test_model(prompt):
            return "테스트답변"

        # 평가 실행 (결과 저장 포함)
        results = self.evaluator.evaluate_model(
            model_name="persistence_test_model",
            evaluation_set=self.complex_data,
            model_function=test_model,
            save_results=True
        )

        # DB 저장 확인
        self.mock_db_manager.execute_batch_dml.assert_called_once()

        # 호출된 SQL 파라미터 확인
        call_args = self.mock_db_manager.execute_batch_dml.call_args
        self.assertIsNotNone(call_args)

        sql, params_list = call_args[0]
        self.assertIn("INSERT INTO", sql)
        self.assertEqual(len(params_list), 4)  # 4개 문제에 대한 파라미터


class TestEvaluationEdgeCases(unittest.TestCase):
    """평가 시스템 엣지 케이스 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.mock_db_manager = Mock()
        self.mock_collector = Mock()
        self.evaluator = ReasoningEvaluator(self.mock_db_manager, self.mock_collector)

    def test_empty_evaluation_set(self):
        """빈 평가 데이터셋 테스트"""

        def dummy_model(prompt):
            return "답변"

        # 빈 데이터셋으로 평가
        results = self.evaluator.evaluate_model(
            model_name="test_model",
            evaluation_set=[],
            model_function=dummy_model,
            save_results=False
        )

        # 결과 확인
        self.assertEqual(results['total_questions'], 0)
        self.assertEqual(results['accuracy'], 0.0)

    def test_unicode_handling(self):
        """유니코드 처리 테스트"""
        unicode_data = [
            ReasoningDataPoint(
                id="unicode_1",
                category="language",
                difficulty="medium",
                question="이 문장에서 한글이 몇 글자인가? '안녕하세요 🌟'",
                correct_answer="5",
                source="unicode_test"
            ),
            ReasoningDataPoint(
                id="unicode_2",
                category="math",
                difficulty="easy",
                question="π의 근사값은?",
                correct_answer="3.14",
                source="unicode_test"
            )
        ]

        def unicode_model(prompt):
            if "한글이 몇 글자" in prompt:
                return "5"
            elif "π의 근사값" in prompt:
                return "3.14"
            return "모름"

        # 평가 실행
        results = self.evaluator.evaluate_model(
            model_name="unicode_model",
            evaluation_set=unicode_data,
            model_function=unicode_model,
            save_results=False
        )

        # 결과 확인
        self.assertEqual(results['accuracy'], 1.0)

    def test_very_long_inputs(self):
        """매우 긴 입력 처리 테스트"""
        long_question = "이것은 매우 긴 질문입니다. " * 100  # 긴 텍스트 생성

        long_data = [
            ReasoningDataPoint(
                id="long_1",
                category="reading_comprehension",
                difficulty="hard",
                question=long_question + " 이 글의 주제는 무엇인가?",
                correct_answer="반복되는 문장에 대한 이해",
                source="long_text_test"
            )
        ]

        def patient_model(prompt):
            if len(prompt) > 1000:
                return "긴 텍스트 처리 완료"
            return "짧은 답변"

        # 평가 실행
        results = self.evaluator.evaluate_model(
            model_name="patient_model",
            evaluation_set=long_data,
            model_function=patient_model,
            save_results=False
        )

        # 결과 확인 (오류 없이 처리되어야 함)
        self.assertEqual(results['total_questions'], 1)

    def test_numerical_precision(self):
        """숫자 정밀도 테스트"""
        precision_data = [
            ReasoningDataPoint(
                id="precision_1",
                category="math",
                difficulty="hard",
                question="π를 소수점 6자리까지 구하면?",
                correct_answer="3.141593",
                source="precision_test"
            )
        ]

        def precise_model(prompt):
            return "3.141592"  # 약간 다른 값

        # 수학 전용 지표로 평가
        from evaluation.metrics import MathAccuracyMetric

        math_metric = MathAccuracyMetric(tolerance=0.000001)  # 매우 낮은 허용 오차

        predictions = ["3.141592"]
        ground_truth = ["3.141593"]

        result = math_metric.calculate(predictions, ground_truth)

        # 허용 오차를 벗어나므로 틀려야 함
        self.assertEqual(result.value, 0.0)

    def test_concurrent_evaluation(self):
        """동시 평가 테스트"""
        import threading
        import time

        results_list = []
        errors_list = []

        def evaluation_worker(worker_id):
            try:
                def worker_model(prompt):
                    time.sleep(0.01)  # 짧은 지연
                    return f"worker_{worker_id}_response"

                test_data = [
                    ReasoningDataPoint(
                        id=f"concurrent_{worker_id}",
                        category="test",
                        difficulty="easy",
                        question="테스트 질문",
                        correct_answer="테스트 답변",
                        source="concurrent_test"
                    )
                ]

                results = self.evaluator.evaluate_model(
                    model_name=f"worker_model_{worker_id}",
                    evaluation_set=test_data,
                    model_function=worker_model,
                    save_results=False
                )

                results_list.append(results)

            except Exception as e:
                errors_list.append(str(e))

        # 여러 스레드로 동시 평가
        threads = []
        for i in range(3):
            thread = threading.Thread(target=evaluation_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()

        # 결과 확인
        self.assertEqual(len(errors_list), 0, f"동시 평가 중 오류 발생: {errors_list}")
        self.assertEqual(len(results_list), 3)

        # 각 결과가 유효한지 확인
        for results in results_list:
            self.assertIn('accuracy', results)
            self.assertEqual(results['total_questions'], 1)


if __name__ == '__main__':
    # 테스트 실행 설정
    unittest.main(verbosity=2, buffer=True)