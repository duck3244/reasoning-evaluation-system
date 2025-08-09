"""
데이터 컬렉터 단위 테스트
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
from datetime import datetime

from data_collector import ReasoningDatasetCollector
from data_models import ReasoningDataPoint, Constants


class TestReasoningDatasetCollector(unittest.TestCase):
    """ReasoningDatasetCollector 단위 테스트"""

    def setUp(self):
        """테스트 설정"""
        self.mock_db_manager = Mock()
        self.collector = ReasoningDatasetCollector(self.mock_db_manager)

        # 기본 테스트 데이터
        self.test_data_point = ReasoningDataPoint(
            id="test_id_001",
            category="math",
            difficulty="easy",
            question="2 + 2 = ?",
            correct_answer="4",
            explanation="2와 2를 더하면 4입니다.",
            source="test"
        )

    def test_add_data_point_success(self):
        """데이터 포인트 추가 성공 테스트"""
        # Mock 설정
        self.mock_db_manager.execute_dml.return_value = 1

        # 테스트 실행
        result = self.collector.add_data_point(self.test_data_point)

        # 검증
        self.assertTrue(result)
        self.mock_db_manager.execute_dml.assert_called_once()

        # 호출된 파라미터 검증
        call_args = self.mock_db_manager.execute_dml.call_args
        self.assertIsNotNone(call_args)
        self.assertIn("MERGE INTO", call_args[0][0])  # SQL 확인

    def test_add_data_point_invalid_data(self):
        """잘못된 데이터 포인트 추가 테스트"""
        # 잘못된 데이터 (빈 질문)
        invalid_data = ReasoningDataPoint(
            id="invalid_id",
            category="math",
            difficulty="easy",
            question="",  # 빈 질문
            correct_answer="4",
            source="test"
        )

        # 테스트 실행
        result = self.collector.add_data_point(invalid_data)

        # 검증
        self.assertFalse(result)
        self.mock_db_manager.execute_dml.assert_not_called()

    def test_add_data_point_database_error(self):
        """데이터베이스 오류 시 테스트"""
        # Mock 설정 (DB 오류 발생)
        self.mock_db_manager.execute_dml.side_effect = Exception("DB Error")

        # 테스트 실행
        result = self.collector.add_data_point(self.test_data_point)

        # 검증
        self.assertFalse(result)
        self.mock_db_manager.execute_dml.assert_called_once()

    def test_add_batch_data_points_success(self):
        """배치 데이터 추가 성공 테스트"""
        # 테스트 데이터 생성
        batch_data = [
            ReasoningDataPoint(
                id=f"test_id_{i:03d}",
                category="math",
                difficulty="easy",
                question=f"문제 {i}",
                correct_answer=f"답 {i}",
                source="test"
            )
            for i in range(5)
        ]

        # Mock 설정
        self.mock_db_manager.execute_batch_dml.return_value = len(batch_data)

        # 테스트 실행
        result = self.collector.add_batch_data_points(batch_data)

        # 검증
        self.assertEqual(result, len(batch_data))
        self.mock_db_manager.execute_batch_dml.assert_called_once()

    def test_add_batch_empty_list(self):
        """빈 리스트 배치 추가 테스트"""
        result = self.collector.add_batch_data_points([])

        # 검증
        self.assertEqual(result, 0)
        self.mock_db_manager.execute_batch_dml.assert_not_called()

    def test_get_data_with_filters(self):
        """필터를 사용한 데이터 조회 테스트"""
        # Mock 데이터 설정
        mock_rows = [
            ("test_id", "math", "easy", "2+2=?", "4", None, None, "test",
             "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)
        ]
        self.mock_db_manager.execute_query.return_value = mock_rows

        # 테스트 실행
        result = self.collector.get_data(category="math", difficulty="easy", limit=10)

        # 검증
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].category, "math")
        self.assertEqual(result[0].difficulty, "easy")
        self.mock_db_manager.execute_query.assert_called_once()

    def test_get_data_empty_result(self):
        """빈 결과 조회 테스트"""
        # Mock 설정 (빈 결과)
        self.mock_db_manager.execute_query.return_value = []

        # 테스트 실행
        result = self.collector.get_data(category="nonexistent")

        # 검증
        self.assertEqual(len(result), 0)
        self.assertIsInstance(result, list)

    def test_get_data_by_id_success(self):
        """ID로 데이터 조회 성공 테스트"""
        # Mock 데이터 설정
        mock_rows = [
            ("test_id", "math", "easy", "2+2=?", "4", "설명", None, "test",
             "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)
        ]
        self.mock_db_manager.execute_query.return_value = mock_rows

        # 테스트 실행
        result = self.collector.get_data_by_id("test_id")

        # 검증
        self.assertIsNotNone(result)
        self.assertEqual(result.id, "test_id")
        self.assertEqual(result.question, "2+2=?")

    def test_get_data_by_id_not_found(self):
        """ID로 데이터 조회 실패 테스트"""
        # Mock 설정 (결과 없음)
        self.mock_db_manager.execute_query.return_value = []

        # 테스트 실행
        result = self.collector.get_data_by_id("nonexistent_id")

        # 검증
        self.assertIsNone(result)

    def test_delete_data_point_success(self):
        """데이터 포인트 삭제 성공 테스트"""
        # Mock 설정
        self.mock_db_manager.execute_dml.return_value = 1

        # 테스트 실행
        result = self.collector.delete_data_point("test_id")

        # 검증
        self.assertTrue(result)
        self.mock_db_manager.execute_dml.assert_called_once()

    def test_delete_data_point_not_found(self):
        """존재하지 않는 데이터 삭제 테스트"""
        # Mock 설정 (삭제된 행 없음)
        self.mock_db_manager.execute_dml.return_value = 0

        # 테스트 실행
        result = self.collector.delete_data_point("nonexistent_id")

        # 검증
        self.assertFalse(result)

    def test_get_statistics(self):
        """통계 정보 조회 테스트"""
        # Mock 통계 데이터
        self.mock_db_manager.execute_query.side_effect = [
            [(100,)],  # 전체 개수
            [("math", 30), ("logic", 25), ("science", 45)],  # 카테고리별
            [("easy", 40), ("medium", 35), ("hard", 25)],  # 난이도별
            [("test", 50), ("manual", 50)]  # 소스별
        ]

        # 테스트 실행
        result = self.collector.get_statistics()

        # 검증
        self.assertEqual(result.total_count, 100)
        self.assertEqual(result.category_counts["math"], 30)
        self.assertEqual(result.difficulty_counts["easy"], 40)
        self.assertEqual(result.source_counts["test"], 50)

    def test_validate_data_point_valid(self):
        """유효한 데이터 포인트 검증 테스트"""
        valid_data = ReasoningDataPoint(
            id="test",
            category="math",
            difficulty="easy",
            question="Valid question",
            correct_answer="Valid answer",
            source="test"
        )

        result = self.collector._validate_data_point(valid_data)
        self.assertTrue(result)

    def test_validate_data_point_invalid_empty_question(self):
        """빈 질문 데이터 검증 테스트"""
        invalid_data = ReasoningDataPoint(
            id="test",
            category="math",
            difficulty="easy",
            question="",  # 빈 질문
            correct_answer="answer",
            source="test"
        )

        result = self.collector._validate_data_point(invalid_data)
        self.assertFalse(result)

    def test_validate_data_point_invalid_empty_answer(self):
        """빈 답변 데이터 검증 테스트"""
        invalid_data = ReasoningDataPoint(
            id="test",
            category="math",
            difficulty="easy",
            question="question",
            correct_answer="",  # 빈 답변
            source="test"
        )

        result = self.collector._validate_data_point(invalid_data)
        self.assertFalse(result)

    def test_validate_data_point_invalid_category(self):
        """잘못된 카테고리 데이터 검증 테스트"""
        data = ReasoningDataPoint(
            id="test",
            category="invalid_category",
            difficulty="easy",
            question="question",
            correct_answer="answer",
            source="test"
        )

        result = self.collector._validate_data_point(data)
        self.assertTrue(result)  # 카테고리는 'unknown'으로 수정되어 통과
        self.assertEqual(data.category, "unknown")

    def test_validate_data_point_invalid_difficulty(self):
        """잘못된 난이도 데이터 검증 테스트"""
        data = ReasoningDataPoint(
            id="test",
            category="math",
            difficulty="invalid_difficulty",
            question="question",
            correct_answer="answer",
            source="test"
        )

        result = self.collector._validate_data_point(data)
        self.assertTrue(result)  # 난이도는 'medium'으로 수정되어 통과
        self.assertEqual(data.difficulty, "medium")

    def test_validate_data_point_too_long_question(self):
        """너무 긴 질문 데이터 검증 테스트"""
        long_question = "x" * 10001  # 10,000자 초과
        data = ReasoningDataPoint(
            id="test",
            category="math",
            difficulty="easy",
            question=long_question,
            correct_answer="answer",
            source="test"
        )

        result = self.collector._validate_data_point(data)
        self.assertFalse(result)

    def test_validate_data_point_too_long_answer(self):
        """너무 긴 답변 데이터 검증 테스트"""
        long_answer = "x" * 5001  # 5,000자 초과
        data = ReasoningDataPoint(
            id="test",
            category="math",
            difficulty="easy",
            question="question",
            correct_answer=long_answer,
            source="test"
        )

        result = self.collector._validate_data_point(data)
        self.assertFalse(result)

    def test_export_to_json(self):
        """JSON 내보내기 테스트"""
        # Mock 데이터
        mock_rows = [
            ("test_id", "math", "easy", "2+2=?", "4", None, None, "test",
             "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)
        ]
        self.mock_db_manager.execute_query.return_value = mock_rows

        # 파일 경로 모킹
        with patch('builtins.open', create=True) as mock_open:
            with patch('json.dump') as mock_json_dump:
                # 테스트 실행
                result = self.collector.export_to_json("test.json", category="math")

                # 검증
                self.assertTrue(result)
                mock_open.assert_called_once()
                mock_json_dump.assert_called_once()

    def test_export_to_csv(self):
        """CSV 내보내기 테스트"""
        # Mock 데이터
        mock_rows = [
            ("test_id", "math", "easy", "2+2=?", "4", "설명", None, "test",
             "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)
        ]
        self.mock_db_manager.execute_query.return_value = mock_rows

        # 파일 경로 모킹
        with patch('builtins.open', create=True):
            with patch('csv.DictWriter') as mock_writer:
                mock_writer_instance = Mock()
                mock_writer.return_value = mock_writer_instance

                # 테스트 실행
                result = self.collector.export_to_csv("test.csv", category="math")

                # 검증
                self.assertTrue(result)
                mock_writer_instance.writeheader.assert_called_once()
                mock_writer_instance.writerow.assert_called_once()

    def test_load_from_json_success(self):
        """JSON 파일 로드 성공 테스트"""
        test_data = [
            {
                "id": "test_id",
                "category": "math",
                "difficulty": "easy",
                "question": "2+2=?",
                "correct_answer": "4",
                "explanation": None,
                "options": None,
                "source": "test",
                "created_at": "2024-01-01T10:00:00",
                "updated_at": None,
                "metadata": None
            }
        ]

        # Mock 설정
        self.mock_db_manager.execute_batch_dml.return_value = 1

        with patch('builtins.open', create=True):
            with patch('json.load', return_value=test_data):
                # 테스트 실행
                result = self.collector.load_from_json("test.json")

                # 검증
                self.assertEqual(result, 1)
                self.mock_db_manager.execute_batch_dml.assert_called_once()

    def test_load_from_json_invalid_format(self):
        """잘못된 형식의 JSON 파일 로드 테스트"""
        # JSON이 리스트가 아닌 경우
        invalid_data = {"invalid": "format"}

        with patch('builtins.open', create=True):
            with patch('json.load', return_value=invalid_data):
                # 테스트 실행
                result = self.collector.load_from_json("invalid.json")

                # 검증
                self.assertEqual(result, 0)
                self.mock_db_manager.execute_batch_dml.assert_not_called()

    def test_load_from_json_file_not_found(self):
        """존재하지 않는 JSON 파일 로드 테스트"""
        # 테스트 실행
        result = self.collector.load_from_json("nonexistent.json")

        # 검증
        self.assertEqual(result, 0)

    def test_load_from_csv_success(self):
        """CSV 파일 로드 성공 테스트"""
        # Mock 설정
        self.mock_db_manager.execute_batch_dml.return_value = 1

        # CSV 데이터 모킹
        csv_data = [
            {
                'id': 'test_id',
                'category': 'math',
                'difficulty': 'easy',
                'question': '2+2=?',
                'correct_answer': '4',
                'explanation': '',
                'options': '',
                'source': 'test',
                'created_at': '2024-01-01T10:00:00',
                'updated_at': '',
                'metadata': ''
            }
        ]

        with patch('builtins.open', create=True):
            with patch('csv.DictReader', return_value=csv_data):
                # 테스트 실행
                result = self.collector.load_from_csv("test.csv")

                # 검증
                self.assertEqual(result, 1)
                self.mock_db_manager.execute_batch_dml.assert_called_once()

    def test_operation_stats(self):
        """작업 통계 테스트"""
        # 초기 상태
        stats = self.collector.get_operation_stats()
        self.assertEqual(stats['total_operations'], 0)

        # 성공 작업 시뮬레이션
        self.mock_db_manager.execute_dml.return_value = 1
        self.collector.add_data_point(self.test_data_point)

        stats = self.collector.get_operation_stats()
        self.assertEqual(stats['total_operations'], 1)
        self.assertEqual(stats['successful_operations'], 1)
        self.assertEqual(stats['success_rate_percent'], 100.0)

        # 실패 작업 시뮬레이션
        self.mock_db_manager.execute_dml.side_effect = Exception("DB Error")
        invalid_data = ReasoningDataPoint(
            id="fail_test",
            category="math",
            difficulty="easy",
            question="Test",
            correct_answer="Test",
            source="test"
        )
        self.collector.add_data_point(invalid_data)

        stats = self.collector.get_operation_stats()
        self.assertEqual(stats['total_operations'], 2)
        self.assertEqual(stats['successful_operations'], 1)
        self.assertEqual(stats['failed_operations'], 1)
        self.assertEqual(stats['success_rate_percent'], 50.0)

    def test_reset_operation_stats(self):
        """작업 통계 리셋 테스트"""
        # 작업 수행
        self.mock_db_manager.execute_dml.return_value = 1
        self.collector.add_data_point(self.test_data_point)

        # 통계 확인
        stats = self.collector.get_operation_stats()
        self.assertGreater(stats['total_operations'], 0)

        # 리셋
        self.collector.reset_operation_stats()

        # 리셋 후 확인
        stats = self.collector.get_operation_stats()
        self.assertEqual(stats['total_operations'], 0)
        self.assertEqual(stats['successful_operations'], 0)
        self.assertEqual(stats['failed_operations'], 0)

    def test_get_data_iterator(self):
        """데이터 이터레이터 테스트"""
        # Mock 데이터 설정 (두 배치)
        batch1 = [
            ("test_id_1", "math", "easy", "1+1=?", "2", None, None, "test",
             "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)
        ]
        batch2 = []  # 빈 배치로 종료

        self.mock_db_manager.execute_query.side_effect = [batch1, batch2]

        # 테스트 실행
        batches = list(self.collector.get_data_iterator(batch_size=1))

        # 검증
        self.assertEqual(len(batches), 1)
        self.assertEqual(len(batches[0]), 1)
        self.assertEqual(batches[0][0].question, "1+1=?")

    def test_save_statistics(self):
        """통계 저장 테스트"""
        from data_models import DatasetStatistics

        # Mock 설정
        self.mock_db_manager.execute_dml.return_value = 1

        # 테스트 데이터
        stats = DatasetStatistics(
            total_count=100,
            category_counts={"math": 50, "logic": 50},
            difficulty_counts={"easy": 30, "medium": 40, "hard": 30},
            source_counts={"test": 100},
            created_at="2024-01-01T10:00:00"
        )

        # 테스트 실행
        result = self.collector.save_statistics(stats)

        # 검증
        self.assertTrue(result)
        self.mock_db_manager.execute_dml.assert_called_once()

    def test_cleanup_old_data(self):
        """오래된 데이터 정리 테스트"""
        # Mock 설정
        self.mock_db_manager.execute_dml.return_value = 5

        # 테스트 실행
        result = self.collector.cleanup_old_data(days_old=30)

        # 검증
        self.assertEqual(result, 5)
        self.mock_db_manager.execute_dml.assert_called_once()

    def test_get_data_quality_report(self):
        """데이터 품질 리포트 테스트"""
        # Mock 통계 데이터
        self.mock_db_manager.execute_query.side_effect = [
            [(100,)],  # 전체 개수
            [("math", 60), ("logic", 40)],  # 카테고리별
            [("easy", 30), ("medium", 40), ("hard", 30)],  # 난이도별
            [("test", 100)],  # 소스별
            # 샘플 데이터용
            [("test_id", "math", "easy", "Short question", "Short answer", "explanation", None, "test",
              "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)]
        ]

        # 테스트 실행
        report = self.collector.get_data_quality_report()

        # 검증
        self.assertIn('total_records', report)
        self.assertIn('data_quality_issues', report)
        self.assertIn('category_distribution', report)
        self.assertIn('average_question_length', report)
        self.assertEqual(report['total_records'], 100)

    def test_backup_data(self):
        """데이터 백업 테스트"""
        # Mock 데이터
        mock_rows = [
            ("test_id", "math", "easy", "2+2=?", "4", None, None, "test",
             "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)
        ]
        self.mock_db_manager.execute_query.return_value = mock_rows

        with patch('builtins.open', create=True):
            with patch('json.dump') as mock_json_dump:
                # 테스트 실행 (압축 없음)
                result = self.collector.backup_data("backup.json", compress=False)

                # 검증
                self.assertTrue(result)
                mock_json_dump.assert_called_once()

    def test_backup_data_compressed(self):
        """압축 데이터 백업 테스트"""
        # Mock 데이터
        mock_rows = [
            ("test_id", "math", "easy", "2+2=?", "4", None, None, "test",
             "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)
        ]
        self.mock_db_manager.execute_query.return_value = mock_rows

        with patch('gzip.open', create=True):
            with patch('json.dump') as mock_json_dump:
                # 테스트 실행 (압축)
                result = self.collector.backup_data("backup.json", compress=True)

                # 검증
                self.assertTrue(result)
                mock_json_dump.assert_called_once()

    def test_restore_from_backup(self):
        """백업 복원 테스트"""
        # 백업 데이터
        backup_data = [
            {
                "id": "test_id",
                "category": "math",
                "difficulty": "easy",
                "question": "2+2=?",
                "correct_answer": "4",
                "explanation": None,
                "options": None,
                "source": "test",
                "created_at": "2024-01-01T10:00:00",
                "updated_at": None,
                "metadata": None
            }
        ]

        # Mock 설정
        self.mock_db_manager.execute_batch_dml.return_value = 1

        with patch('builtins.open', create=True):
            with patch('json.load', return_value=backup_data):
                # 테스트 실행
                result = self.collector.restore_from_backup("backup.json")

                # 검증
                self.assertEqual(result, 1)
                self.mock_db_manager.execute_batch_dml.assert_called_once()

    def test_optimize_storage(self):
        """스토리지 최적화 테스트"""
        # Mock 설정
        self.mock_db_manager.execute_dml.return_value = 3  # 3개 중복 제거

        # 테스트 실행
        result = self.collector.optimize_storage()

        # 검증
        self.assertIn('deleted_duplicates', result)
        self.assertEqual(result['deleted_duplicates'], 3)
        self.mock_db_manager.execute_dml.assert_called_once()

    def test_export_to_json_streaming(self):
        """스트리밍 JSON 내보내기 테스트"""
        # Mock 데이터
        mock_rows = [
            ("test_id", "math", "easy", "2+2=?", "4", None, None, "test",
             "2024-01-01T10:00:00", "2024-01-01T10:00:00", None)
        ]
        self.mock_db_manager.execute_query.side_effect = [mock_rows, []]  # 첫 배치 후 빈 배치

        with patch('builtins.open', create=True) as mock_open:
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            with patch('json.dump') as mock_json_dump:
                # 테스트 실행
                result = self.collector.export_to_json_streaming("test.json")

                # 검증
                self.assertTrue(result)
                mock_file.write.assert_called()  # JSON 구조 작성 확인
                mock_json_dump.assert_called()  # 데이터 dump 확인


class TestBatchProcessor(unittest.TestCase):
    """BatchProcessor 단위 테스트"""

    def setUp(self):
        """테스트 설정"""
        from data_collector import BatchProcessor
        self.batch_processor = BatchProcessor(batch_size=3)

    def test_process_in_batches_success(self):
        """배치 처리 성공 테스트"""
        # 테스트 데이터
        test_data = list(range(10))  # 0~9

        # 처리 함수 (각 배치의 크기 반환)
        def process_func(batch):
            return len(batch)

        # 테스트 실행
        result = self.batch_processor.process_in_batches(test_data, process_func)

        # 검증 (총 10개 처리됨)
        self.assertEqual(result, 10)

    def test_process_in_batches_with_error(self):
        """배치 처리 중 오류 발생 테스트"""
        test_data = list(range(10))

        def process_func(batch):
            if len(batch) == 3:  # 첫 번째 배치에서 오류 발생
                raise Exception("Test error")
            return len(batch)

        # 테스트 실행 (오류가 발생해도 계속 진행)
        result = self.batch_processor.process_in_batches(test_data, process_func)

        # 검증 (첫 배치 3개 실패, 나머지 7개 성공)
        self.assertEqual(result, 7)

    def test_process_empty_data(self):
        """빈 데이터 배치 처리 테스트"""

        def process_func(batch):
            return len(batch)

        result = self.batch_processor.process_in_batches([], process_func)
        self.assertEqual(result, 0)


if __name__ == '__main__':
    # 테스트 실행 설정
    unittest.main(verbosity=2)