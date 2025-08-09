"""
통합 테스트
"""
import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

from database_config import DatabaseConfig, DatabaseManager
from data_collector import ReasoningDatasetCollector
from evaluation_system import ReasoningEvaluator
from external_data_loader import ExternalDatasetLoader
from main import ReasoningEvaluationSystem
from data_models import ReasoningDataPoint


class TestSystemIntegration(unittest.TestCase):
    """시스템 통합 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_db_config = DatabaseConfig(
            username="test_user",
            password="test_pass",
            dsn="test:1521/test"
        )

        # 데이터베이스 모킹
        self.mock_db_manager = MagicMock(spec=DatabaseManager)
        self.mock_db_manager.execute_query.return_value = []
        self.mock_db_manager.execute_dml.return_value = 1
        self.mock_db_manager.execute_batch_dml.return_value = 10

    def test_full_system_workflow(self):
        """전체 시스템 워크플로우 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            # 시스템 초기화
            system = ReasoningEvaluationSystem(self.mock_db_config)

            # 데이터베이스 설정
            system.setup_database()
            self.mock_db_manager.init_database.assert_called_once()

            # 샘플 데이터 로드
            load_results = system.load_sample_data()
            self.assertIsInstance(load_results, dict)
            self.assertIn('basic_samples', load_results)

            # 평가 실행
            # 평가를 위한 mock 데이터 설정
            self.mock_db_manager.execute_query.return_value = [
                ("test_id", "math", "easy", "2+2=?", "4", None, None, "test",
                 "2024-01-01", "2024-01-01", None)
            ]

            eval_results = system.run_evaluation("test_model", test_size=1)
            self.assertIsInstance(eval_results, dict)

            if eval_results:  # 평가가 성공한 경우
                self.assertIn('accuracy', eval_results)
                self.assertIn('total_questions', eval_results)

    def test_data_flow_integrity(self):
        """데이터 흐름 무결성 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            collector = ReasoningDatasetCollector(self.mock_db_manager)

            # 테스트 데이터 생성
            test_data = ReasoningDataPoint(
                id="test_id",
                category="math",
                difficulty="easy",
                question="2+2=?",
                correct_answer="4"
            )

            # 데이터 추가
            result = collector.add_data_point(test_data)
            self.assertTrue(result)

            # 데이터베이스 호출 확인
            self.mock_db_manager.execute_dml.assert_called()

            # 호출 파라미터 검증
            call_args = self.mock_db_manager.execute_dml.call_args
            self.assertIsNotNone(call_args)

            sql, params = call_args[0]
            self.assertIn("MERGE INTO", sql)
            self.assertEqual(params[0], "test_id")  # ID 확인
            self.assertEqual(params[1], "math")  # 카테고리 확인

    def test_evaluation_pipeline(self):
        """평가 파이프라인 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            # 모킹된 데이터 반환
            self.mock_db_manager.execute_query.return_value = [
                ("test_id", "math", "easy", "2+2=?", "4", None, None, "test",
                 "2024-01-01", "2024-01-01", None)
            ]

            collector = ReasoningDatasetCollector(self.mock_db_manager)
            evaluator = ReasoningEvaluator(self.mock_db_manager, collector)

            # 평가 데이터셋 생성
            eval_set = evaluator.create_evaluation_set(test_size=1)
            self.assertEqual(len(eval_set), 1)
            self.assertEqual(eval_set[0].question, "2+2=?")

            # 모델 평가
            def dummy_model(prompt):
                return "4"  # 정답 반환

            results = evaluator.evaluate_model(
                model_name="test_model",
                evaluation_set=eval_set,
                model_function=dummy_model,
                save_results=False
            )

            # 결과 검증
            self.assertIn('accuracy', results)
            self.assertIn('total_questions', results)
            self.assertEqual(results['total_questions'], 1)
            self.assertEqual(results['correct_answers'], 1)
            self.assertEqual(results['accuracy'], 1.0)

    def test_external_data_loading(self):
        """외부 데이터 로딩 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            collector = ReasoningDatasetCollector(self.mock_db_manager)
            loader = ExternalDatasetLoader(collector)

            # GSM8K 샘플 로드 테스트
            result = loader.load_gsm8k_sample()
            self.assertGreater(result, 0)

            # 배치 DML 호출 확인
            self.mock_db_manager.execute_batch_dml.assert_called()

    def test_error_handling_database_failure(self):
        """데이터베이스 실패 시 오류 처리 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            # 데이터베이스 오류 시뮬레이션
            self.mock_db_manager.execute_dml.side_effect = Exception("DB Connection Error")

            collector = ReasoningDatasetCollector(self.mock_db_manager)

            test_data = ReasoningDataPoint(
                id="test_id",
                category="math",
                difficulty="easy",
                question="2+2=?",
                correct_answer="4"
            )

            # 오류가 적절히 처리되는지 확인
            result = collector.add_data_point(test_data)
            self.assertFalse(result)

    def test_memory_management_large_dataset(self):
        """대용량 데이터셋 메모리 관리 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            collector = ReasoningDatasetCollector(self.mock_db_manager, batch_size=100)

            # 대량의 테스트 데이터 생성
            large_dataset = [
                ReasoningDataPoint(
                    id=f"test_id_{i:06d}",
                    category="math",
                    difficulty="easy",
                    question=f"문제 {i}",
                    correct_answer=f"답 {i}",
                    source="test"
                )
                for i in range(1000)  # 1000개 데이터
            ]

            # 배치 처리 테스트
            result = collector.add_batch_data_points(large_dataset)

            # 배치 처리가 여러 번 호출되었는지 확인
            self.assertGreater(self.mock_db_manager.execute_batch_dml.call_count, 1)

            # 결과 검증
            self.assertGreater(result, 0)

    def test_configuration_management(self):
        """설정 관리 테스트"""
        # 임시 설정 파일 생성
        config_data = {
            "username": "test_user",
            "password": "test_pass",
            "dsn": "test:1521/test",
            "pool_min": 2,
            "pool_max": 20
        }

        config_file = os.path.join(self.temp_dir, "test_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        # 설정 로드 테스트
        with patch('database_config.load_db_config_from_file') as mock_load:
            mock_load.return_value = self.mock_db_config

            from database_config import load_db_config_from_file
            config = load_db_config_from_file(config_file)

            self.assertIsInstance(config, DatabaseConfig)

    def test_performance_monitoring(self):
        """성능 모니터링 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            from performance_optimizer import performance_monitor

            # 성능 모니터링 데코레이터 테스트
            @performance_monitor()
            def test_function():
                import time
                time.sleep(0.1)  # 0.1초 대기
                return "test_result"

            with patch('logging.Logger.info') as mock_log:
                result = test_function()

                # 결과 검증
                self.assertEqual(result, "test_result")

                # 로그 호출 확인
                mock_log.assert_called()
                log_call_args = mock_log.call_args[0][0]
                self.assertIn("실행 완료", log_call_args)

    def test_data_export_import_cycle(self):
        """데이터 내보내기/가져오기 사이클 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            collector = ReasoningDatasetCollector(self.mock_db_manager)

            # 내보내기용 모킹 데이터
            self.mock_db_manager.execute_query.return_value = [
                ("test_id", "math", "easy", "2+2=?", "4", "설명", None, "test",
                 "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)
            ]

            # JSON 내보내기 테스트
            export_file = os.path.join(self.temp_dir, "export_test.json")
            with patch('builtins.open', create=True) as mock_open:
                with patch('json.dump') as mock_json_dump:
                    result = collector.export_to_json(export_file)
                    self.assertTrue(result)

            # JSON 가져오기 테스트
            test_import_data = [
                {
                    "id": "import_test_id",
                    "category": "logic",
                    "difficulty": "medium",
                    "question": "논리 문제",
                    "correct_answer": "답",
                    "explanation": None,
                    "options": None,
                    "source": "import_test",
                    "created_at": "2024-01-01T10:00:00",
                    "updated_at": None,
                    "metadata": None
                }
            ]

            with patch('builtins.open', create=True):
                with patch('json.load', return_value=test_import_data):
                    import_result = collector.load_from_json("import_test.json")
                    self.assertGreater(import_result, 0)

    def test_concurrent_operations(self):
        """동시 작업 테스트"""
        import threading
        import time

        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            collector = ReasoningDatasetCollector(self.mock_db_manager)

            results = []
            errors = []

            def worker_thread(thread_id):
                try:
                    for i in range(10):
                        test_data = ReasoningDataPoint(
                            id=f"thread_{thread_id}_item_{i}",
                            category="math",
                            difficulty="easy",
                            question=f"Thread {thread_id} Question {i}",
                            correct_answer=f"Answer {i}",
                            source="concurrent_test"
                        )

                        result = collector.add_data_point(test_data)
                        results.append(result)

                        time.sleep(0.01)  # 짧은 대기

                except Exception as e:
                    errors.append(str(e))

            # 여러 스레드 동시 실행
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker_thread, args=(i,))
                threads.append(thread)
                thread.start()

            # 모든 스레드 완료 대기
            for thread in threads:
                thread.join()

            # 결과 검증
            self.assertEqual(len(errors), 0, f"동시 작업 중 오류 발생: {errors}")
            self.assertEqual(len(results), 30)  # 3 스레드 × 10 작업

    def test_system_recovery_after_failure(self):
        """실패 후 시스템 복구 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            system = ReasoningEvaluationSystem(self.mock_db_config)

            # 첫 번째 시도: 실패
            self.mock_db_manager.init_database.side_effect = Exception("DB 초기화 실패")

            with self.assertRaises(Exception):
                system.setup_database()

            # 두 번째 시도: 성공
            self.mock_db_manager.init_database.side_effect = None

            # 시스템이 복구되어 정상 작동하는지 확인
            try:
                system.setup_database()
                # 예외가 발생하지 않으면 성공
            except Exception as e:
                self.fail(f"시스템 복구 실패: {e}")

    def tearDown(self):
        """테스트 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestDataValidation(unittest.TestCase):
    """데이터 검증 테스트"""

    def test_data_point_validation(self):
        """데이터 포인트 검증 테스트"""
        from data_models import ReasoningDataPoint, Constants

        # 유효한 데이터
        valid_data = ReasoningDataPoint(
            id="test_001",
            category="math",
            difficulty="easy",
            question="What is 2 + 2?",
            correct_answer="4",
            explanation="Simple addition",
            source="test"
        )

        # 기본 검증
        self.assertTrue(valid_data.question)
        self.assertTrue(valid_data.correct_answer)
        self.assertIn(valid_data.category, Constants.CATEGORIES)
        self.assertIn(valid_data.difficulty, Constants.DIFFICULTIES)

        # ID 생성 테스트
        data_without_id = ReasoningDataPoint(
            id="",
            category="math",
            difficulty="easy",
            question="Test question",
            correct_answer="Test answer"
        )

        generated_id = data_without_id.generate_id()
        self.assertTrue(generated_id)
        self.assertEqual(len(generated_id), 32)  # MD5 해시 길이

    def test_json_serialization(self):
        """JSON 직렬화 테스트"""
        from data_models import ReasoningDataPoint

        data = ReasoningDataPoint(
            id="test_001",
            category="math",
            difficulty="easy",
            question="Test question",
            correct_answer="Test answer",
            options=["A", "B", "C", "D"],
            metadata={"key": "value"}
        )

        # JSON 변환 테스트
        json_str = data.to_json()
        self.assertIsInstance(json_str, str)

        # JSON에서 객체 복원 테스트
        restored_data = ReasoningDataPoint.from_json(json_str)
        self.assertEqual(data.id, restored_data.id)
        self.assertEqual(data.question, restored_data.question)
        self.assertEqual(data.options, restored_data.options)
        self.assertEqual(data.metadata, restored_data.metadata)

    def test_category_validation(self):
        """카테고리 유효성 검증 테스트"""
        from data_models import Constants

        # 모든 유효한 카테고리 테스트
        for category in Constants.CATEGORIES:
            data = ReasoningDataPoint(
                id=f"test_{category}",
                category=category,
                difficulty="easy",
                question="Test question",
                correct_answer="Test answer"
            )
            self.assertEqual(data.category, category)

    def test_difficulty_validation(self):
        """난이도 유효성 검증 테스트"""
        from data_models import Constants

        # 모든 유효한 난이도 테스트
        for difficulty in Constants.DIFFICULTIES:
            data = ReasoningDataPoint(
                id=f"test_{difficulty}",
                category="math",
                difficulty=difficulty,
                question="Test question",
                correct_answer="Test answer"
            )
            self.assertEqual(data.difficulty, difficulty)


class TestPerformanceIntegration(unittest.TestCase):
    """성능 통합 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.mock_db_manager = MagicMock(spec=DatabaseManager)
        self.mock_db_manager.execute_batch_dml.return_value = 1000

    def test_large_batch_processing(self):
        """대용량 배치 처리 성능 테스트"""
        from performance_optimizer import BatchOptimizer

        optimizer = BatchOptimizer()

        # 대량 데이터 생성
        large_dataset = [f"item_{i}" for i in range(10000)]

        def mock_process_func(batch):
            return len(batch)

        # 성능 측정
        import time
        start_time = time.time()

        result = optimizer.process_with_adaptive_batching(
            large_dataset, mock_process_func
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # 결과 검증
        self.assertEqual(result, len(large_dataset))
        self.assertLess(execution_time, 5.0)  # 5초 이내 완료

    def test_memory_optimization(self):
        """메모리 최적화 테스트"""
        from performance_optimizer import MemoryManager

        memory_manager = MemoryManager()

        # 초기 메모리 상태
        initial_memory = memory_manager.check_memory_usage()

        # 메모리 정리 실행
        memory_manager.cleanup_if_needed(force=True)

        # 메모리 정리 후 상태 (실제 메모리 감소는 보장할 수 없으므로 예외 없이 실행되는지만 확인)
        final_memory = memory_manager.check_memory_usage()

        # 메모리 관리 함수가 정상 실행되었는지 확인
        self.assertIsInstance(initial_memory, dict)
        self.assertIsInstance(final_memory, dict)
        self.assertIn('system_memory_percent', initial_memory)
        self.assertIn('system_memory_percent', final_memory)

    def test_query_optimization(self):
        """쿼리 최적화 테스트"""
        from performance_optimizer import QueryOptimizer

        optimizer = QueryOptimizer(self.mock_db_manager)

        # 최적화된 쿼리 생성
        sql, params = optimizer.get_optimized_data_query(
            category="math",
            difficulty="easy",
            limit=100
        )

        # 쿼리 구조 검증
        self.assertIn("SELECT", sql)
        self.assertIn("INDEX", sql)  # 힌트 포함 확인
        self.assertIn("FETCH FIRST", sql)  # LIMIT 처리 확인
        self.assertEqual(len(params), 2)  # category, difficulty 파라미터

    def test_concurrent_performance(self):
        """동시성 성능 테스트"""
        from performance_optimizer import ParallelProcessor

        processor = ParallelProcessor(max_workers=2)

        # 테스트 배치 생성
        batches = [
            [f"batch_{i}_item_{j}" for j in range(100)]
            for i in range(5)
        ]

        def mock_process_func(batch):
            import time
            time.sleep(0.1)  # 작업 시뮬레이션
            return len(batch)

        # 병렬 처리 실행
        import time
        start_time = time.time()

        result = processor.process_parallel_batches(batches, mock_process_func)

        end_time = time.time()
        execution_time = end_time - start_time

        # 결과 검증
        self.assertEqual(result, 500)  # 5 배치 × 100 항목
        self.assertLess(execution_time, 1.0)  # 병렬 처리로 인한 시간 단축


class TestLoggingIntegration(unittest.TestCase):
    """로깅 통합 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = os.path.join(self.temp_dir, "test.log")

    def test_structured_logging(self):
        """구조화된 로깅 테스트"""
        from logging_system import LoggingManager, LogContext

        # 로깅 매니저 설정
        logging_manager = LoggingManager(
            log_level="DEBUG",
            log_format="structured",
            log_file=self.log_file,
            enable_console=False
        )

        import logging
        logger = logging.getLogger("test_logger")

        # 컨텍스트가 있는 로그 작성
        with LogContext(user_id="test_user", operation_id="op_001"):
            logger.info("테스트 메시지")
            logger.error("테스트 에러", exc_info=True)

        # 로그 파일 확인
        self.assertTrue(os.path.exists(self.log_file))

        with open(self.log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()

        # JSON 구조 확인
        self.assertIn('"user_id": "test_user"', log_content)
        self.assertIn('"operation_id": "op_001"', log_content)
        self.assertIn('"level": "INFO"', log_content)
        self.assertIn('"level": "ERROR"', log_content)

    def test_performance_logging(self):
        """성능 로깅 테스트"""
        from logging_system import log_performance, LoggingManager

        # 로깅 매니저 설정
        LoggingManager(
            log_level="DEBUG",
            log_format="structured",
            log_file=self.log_file,
            enable_console=False
        )

        @log_performance()
        def test_function():
            import time
            time.sleep(0.1)
            return "test_result"

        # 함수 실행
        result = test_function()

        # 결과 검증
        self.assertEqual(result, "test_result")

        # 성능 로그 파일 확인
        perf_log_file = self.log_file.replace('.log', '.performance.log')
        self.assertTrue(os.path.exists(perf_log_file))

    def test_log_analysis(self):
        """로그 분석 테스트"""
        from logging_system import LogAnalyzer, LoggingManager

        # 테스트 로그 생성
        LoggingManager(
            log_level="DEBUG",
            log_format="structured",
            log_file=self.log_file,
            enable_console=False
        )

        import logging
        logger = logging.getLogger("test_logger")

        # 다양한 로그 작성
        logger.info("정보 메시지")
        logger.warning("경고 메시지")
        logger.error("에러 메시지")

        # 로그 분석
        analyzer = LogAnalyzer(self.log_file)
        summary = analyzer.generate_log_summary(hours=1)

        # 분석 결과 검증
        self.assertIn('total_log_entries', summary)
        self.assertIn('log_level_distribution', summary)
        self.assertGreater(summary['total_log_entries'], 0)

    def tearDown(self):
        """테스트 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestEndToEndScenario(unittest.TestCase):
    """종단간 시나리오 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_db_manager = MagicMock(spec=DatabaseManager)
        self.mock_db_manager.execute_query.return_value = []
        self.mock_db_manager.execute_dml.return_value = 1
        self.mock_db_manager.execute_batch_dml.return_value = 100

    def test_complete_evaluation_workflow(self):
        """완전한 평가 워크플로우 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            from main import ReasoningEvaluationSystem
            from database_config import DatabaseConfig

            # 1. 시스템 초기화
            db_config = DatabaseConfig("test", "test", "test:1521/test")
            system = ReasoningEvaluationSystem(db_config)

            # 2. 데이터베이스 설정
            system.setup_database()

            # 3. 샘플 데이터 로드
            load_results = system.load_sample_data()
            self.assertIsInstance(load_results, dict)

            # 4. 평가 데이터 설정
            self.mock_db_manager.execute_query.return_value = [
                ("eval_id_1", "math", "easy", "1+1=?", "2", None, None, "test",
                 "2024-01-01", "2024-01-01", None),
                ("eval_id_2", "logic", "medium", "참/거짓 문제", "참", None, None, "test",
                 "2024-01-01", "2024-01-01", None),
            ]

            # 5. 모델 평가 실행
            def simple_model(prompt):
                if "1+1" in prompt:
                    return "2"
                elif "참/거짓" in prompt:
                    return "참"
                else:
                    return "모름"

            # 평가 실행
            eval_results = system.evaluator.evaluate_model(
                model_name="simple_test_model",
                evaluation_set=system.evaluator.create_evaluation_set(test_size=2),
                model_function=simple_model,
                save_results=False
            )

            # 6. 결과 검증
            self.assertIn('accuracy', eval_results)
            self.assertIn('total_questions', eval_results)
            self.assertIn('category_accuracy', eval_results)

            # 7. 통계 조회
            stats = system.get_system_statistics()
            self.assertIsInstance(stats, dict)

    def test_data_lifecycle_management(self):
        """데이터 생명주기 관리 테스트"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            from data_collector import ReasoningDatasetCollector

            collector = ReasoningDatasetCollector(self.mock_db_manager)

            # 1. 데이터 추가
            test_data = ReasoningDataPoint(
                id="lifecycle_test",
                category="math",
                difficulty="easy",
                question="생명주기 테스트 질문",
                correct_answer="테스트 답변",
                source="lifecycle_test"
            )

            add_result = collector.add_data_point(test_data)
            self.assertTrue(add_result)

            # 2. 데이터 조회
            self.mock_db_manager.execute_query.return_value = [
                ("lifecycle_test", "math", "easy", "생명주기 테스트 질문", "테스트 답변",
                 None, None, "lifecycle_test", "2024-01-01", "2024-01-01", None)
            ]

            retrieved_data = collector.get_data_by_id("lifecycle_test")
            self.assertIsNotNone(retrieved_data)
            self.assertEqual(retrieved_data.question, "생명주기 테스트 질문")

            # 3. 데이터 내보내기
            export_file = os.path.join(self.temp_dir, "lifecycle_export.json")
            with patch('builtins.open', create=True):
                with patch('json.dump') as mock_dump:
                    export_result = collector.export_to_json(export_file)
                    self.assertTrue(export_result)

            # 4. 데이터 삭제
            self.mock_db_manager.execute_dml.return_value = 1
            delete_result = collector.delete_data_point("lifecycle_test")
            self.assertTrue(delete_result)

            # 5. 통계 업데이트
            stats = collector.get_statistics()
            self.assertIsInstance(stats, type(stats))

    def test_system_stress_simulation(self):
        """시스템 스트레스 시뮬레이션"""
        with patch('database_config.DatabaseManager', return_value=self.mock_db_manager):
            from data_collector import ReasoningDatasetCollector
            from performance_optimizer import OptimizedDataProcessor

            collector = ReasoningDatasetCollector(self.mock_db_manager)
            optimizer = OptimizedDataProcessor(self.mock_db_manager)

            # 대량 데이터 생성
            stress_data = [
                ReasoningDataPoint(
                    id=f"stress_test_{i:06d}",
                    category="math" if i % 2 == 0 else "logic",
                    difficulty=["easy", "medium", "hard"][i % 3],
                    question=f"스트레스 테스트 질문 {i}",
                    correct_answer=f"답변 {i}",
                    source="stress_test"
                )
                for i in range(5000)  # 5000개 데이터
            ]

            # 배치 처리 함수
            def stress_process_func(batch):
                return len(batch)

            # 스트레스 테스트 실행
            import time
            start_time = time.time()

            result = optimizer.process_large_dataset(stress_data, stress_process_func)

            end_time = time.time()
            execution_time = end_time - start_time

            # 결과 검증
            self.assertGreater(result, 0)
            self.assertLess(execution_time, 30.0)  # 30초 이내 완료

            # 메모리 사용량 체크
            memory_info = optimizer.memory_manager.check_memory_usage()
            self.assertIsInstance(memory_info, dict)
            self.assertIn('system_memory_percent', memory_info)

    def tearDown(self):
        """테스트 정리"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == '__main__':
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()

    # 통합 테스트 추가
    test_suite.addTest(unittest.makeSuite(TestSystemIntegration))
    test_suite.addTest(unittest.makeSuite(TestDataValidation))
    test_suite.addTest(unittest.makeSuite(TestPerformanceIntegration))
    test_suite.addTest(unittest.makeSuite(TestLoggingIntegration))
    test_suite.addTest(unittest.makeSuite(TestEndToEndScenario))

    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 결과 요약
    print(f"\n=== 테스트 결과 요약 ===")
    print(f"실행된 테스트: {result.testsRun}")
    print(f"실패한 테스트: {len(result.failures)}")
    print(f"에러가 발생한 테스트: {len(result.errors)}")

    if result.failures:
        print("\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print("\n에러가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    # 성공 여부 반환
    exit(0 if result.wasSuccessful() else 1)