"""
추론 성능 평가용 데이터 수집기 (개선된 예외 처리 및 성능 최적화)
"""
import json
import csv
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
import logging
import gc
import time

from data_models import ReasoningDataPoint, DatasetStatistics, Constants
from database_config import DatabaseManager, DatabaseConfig

logger = logging.getLogger(__name__)


class BatchProcessor:
    """배치 처리 유틸리티"""

    def __init__(self, batch_size: int = 1000):
        self.batch_size = batch_size

    def process_in_batches(self, data: List[Any], process_func: callable) -> int:
        """데이터를 배치 단위로 처리"""
        total_processed = 0

        for i in range(0, len(data), self.batch_size):
            batch = data[i:i + self.batch_size]
            try:
                processed = process_func(batch)
                total_processed += processed

                # 주기적으로 메모리 정리
                if i % (self.batch_size * 10) == 0:
                    gc.collect()

            except Exception as e:
                logger.error(f"배치 처리 실패 (배치 {i//self.batch_size + 1}): {e}")
                # 배치 실패 시에도 계속 진행
                continue

        return total_processed


class ReasoningDatasetCollector:
    """추론 성능 평가용 데이터셋 수집 및 저장 (개선됨)"""

    def __init__(self, db_manager: DatabaseManager, batch_size: int = 1000):
        self.db_manager = db_manager
        self.batch_processor = BatchProcessor(batch_size)
        self._operation_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0
        }

    def add_data_point(self, data_point: ReasoningDataPoint) -> bool:
        """단일 데이터 포인트 추가 (개선된 예외 처리)"""
        self._operation_stats['total_operations'] += 1

        try:
            # 데이터 검증
            if not self._validate_data_point(data_point):
                logger.error(f"데이터 포인트 검증 실패: {data_point.id}")
                self._operation_stats['failed_operations'] += 1
                return False

            # ID가 없으면 생성
            if not data_point.id:
                data_point.id = data_point.generate_id()

            # 타임스탬프 설정
            if not data_point.created_at:
                data_point.created_at = datetime.now().isoformat()
            data_point.updated_at = datetime.now().isoformat()

            sql = f"""
            MERGE INTO {Constants.TABLE_REASONING_DATA} rd
            USING (SELECT :1 as id FROM dual) src
            ON (rd.ID = src.id)
            WHEN MATCHED THEN
                UPDATE SET 
                    CATEGORY = :2,
                    DIFFICULTY = :3,
                    QUESTION = :4,
                    CORRECT_ANSWER = :5,
                    EXPLANATION = :6,
                    OPTIONS = :7,
                    SOURCE = :8,
                    UPDATED_AT = CURRENT_TIMESTAMP,
                    METADATA = :9
            WHEN NOT MATCHED THEN
                INSERT (ID, CATEGORY, DIFFICULTY, QUESTION, CORRECT_ANSWER, 
                       EXPLANATION, OPTIONS, SOURCE, CREATED_AT, UPDATED_AT, METADATA)
                VALUES (:1, :2, :3, :4, :5, :6, :7, :8, 
                       TO_TIMESTAMP(:10, 'YYYY-MM-DD"T"HH24:MI:SS'), 
                       CURRENT_TIMESTAMP, :9)
            """

            params = [
                data_point.id,
                data_point.category,
                data_point.difficulty,
                data_point.question,
                data_point.correct_answer,
                data_point.explanation,
                json.dumps(data_point.options, ensure_ascii=False) if data_point.options else None,
                data_point.source,
                json.dumps(data_point.metadata, ensure_ascii=False) if data_point.metadata else None,
                data_point.created_at
            ]

            self.db_manager.execute_dml(sql, params)
            logger.debug(f"데이터 포인트 추가/업데이트 완료: {data_point.id}")
            self._operation_stats['successful_operations'] += 1
            return True

        except Exception as e:
            logger.error(f"데이터 추가 오류 (ID: {data_point.id if data_point.id else 'Unknown'}): {e}")
            self._operation_stats['failed_operations'] += 1
            return False

    def add_batch_data_points(self, data_points: List[ReasoningDataPoint]) -> int:
        """여러 데이터 포인트 배치 추가 (메모리 최적화)"""
        if not data_points:
            logger.warning("추가할 데이터 포인트가 없습니다.")
            return 0

        def process_batch(batch: List[ReasoningDataPoint]) -> int:
            return self._process_batch_safely(batch)

        total_added = self.batch_processor.process_in_batches(data_points, process_batch)
        logger.info(f"배치 데이터 추가 완료: {total_added}/{len(data_points)}개")
        return total_added

    def _process_batch_safely(self, batch: List[ReasoningDataPoint]) -> int:
        """안전한 배치 처리"""
        try:
            # 데이터 검증 및 전처리
            valid_batch = []
            for data_point in batch:
                if self._validate_data_point(data_point):
                    if not data_point.id:
                        data_point.id = data_point.generate_id()
                    if not data_point.created_at:
                        data_point.created_at = datetime.now().isoformat()
                    valid_batch.append(data_point)

            if not valid_batch:
                logger.warning("배치에 유효한 데이터 포인트가 없습니다.")
                return 0

            sql = f"""
            INSERT INTO {Constants.TABLE_REASONING_DATA} 
            (ID, CATEGORY, DIFFICULTY, QUESTION, CORRECT_ANSWER, 
             EXPLANATION, OPTIONS, SOURCE, CREATED_AT, UPDATED_AT, METADATA)
            VALUES (:1, :2, :3, :4, :5, :6, :7, :8, 
                   TO_TIMESTAMP(:9, 'YYYY-MM-DD"T"HH24:MI:SS'), 
                   CURRENT_TIMESTAMP, :10)
            """

            params_list = []
            for data_point in valid_batch:
                params = [
                    data_point.id,
                    data_point.category,
                    data_point.difficulty,
                    data_point.question,
                    data_point.correct_answer,
                    data_point.explanation,
                    json.dumps(data_point.options, ensure_ascii=False) if data_point.options else None,
                    data_point.source,
                    data_point.created_at,
                    json.dumps(data_point.metadata, ensure_ascii=False) if data_point.metadata else None
                ]
                params_list.append(params)

            rowcount = self.db_manager.execute_batch_dml(sql, params_list)
            logger.debug(f"배치 처리 완료: {rowcount}개")
            return rowcount

        except Exception as e:
            logger.error(f"배치 처리 오류: {e}")
            return 0

    def _validate_data_point(self, data_point: ReasoningDataPoint) -> bool:
        """데이터 포인트 유효성 검증"""
        if not data_point.question or not data_point.question.strip():
            logger.warning("질문이 비어있습니다.")
            return False

        if not data_point.correct_answer or not data_point.correct_answer.strip():
            logger.warning("정답이 비어있습니다.")
            return False

        if data_point.category not in Constants.CATEGORIES:
            logger.warning(f"유효하지 않은 카테고리: {data_point.category}")
            data_point.category = 'unknown'

        if data_point.difficulty not in Constants.DIFFICULTIES:
            logger.warning(f"유효하지 않은 난이도: {data_point.difficulty}")
            data_point.difficulty = 'medium'

        # 길이 제한 검사
        if len(data_point.question) > 10000:
            logger.warning("질문이 너무 깁니다 (10,000자 초과)")
            return False

        if len(data_point.correct_answer) > 5000:
            logger.warning("정답이 너무 깁니다 (5,000자 초과)")
            return False

        return True

    def get_data_iterator(self,
                         category: Optional[str] = None,
                         difficulty: Optional[str] = None,
                         source: Optional[str] = None,
                         batch_size: int = 1000) -> Iterator[List[ReasoningDataPoint]]:
        """대용량 데이터를 배치 단위로 조회하는 이터레이터"""
        offset = 0

        while True:
            try:
                batch = self.get_data(
                    category=category,
                    difficulty=difficulty,
                    source=source,
                    limit=batch_size,
                    offset=offset
                )

                if not batch:
                    break

                yield batch

                if len(batch) < batch_size:
                    break

                offset += batch_size

            except Exception as e:
                logger.error(f"데이터 이터레이션 오류 (offset: {offset}): {e}")
                break

    def get_data(self,
                 category: Optional[str] = None,
                 difficulty: Optional[str] = None,
                 source: Optional[str] = None,
                 limit: Optional[int] = None,
                 offset: Optional[int] = None) -> List[ReasoningDataPoint]:
        """조건에 따라 데이터 조회 (개선된 예외 처리)"""
        try:
            where_conditions = []
            params = []

            if category:
                where_conditions.append("CATEGORY = :category")
                params.append(category)

            if difficulty:
                where_conditions.append("DIFFICULTY = :difficulty")
                params.append(difficulty)

            if source:
                where_conditions.append("SOURCE = :source")
                params.append(source)

            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"

            sql = f"""
            SELECT ID, CATEGORY, DIFFICULTY, QUESTION, CORRECT_ANSWER,
                   EXPLANATION, OPTIONS, SOURCE, 
                   TO_CHAR(CREATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as CREATED_AT,
                   TO_CHAR(UPDATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as UPDATED_AT,
                   METADATA
            FROM {Constants.TABLE_REASONING_DATA}
            WHERE {where_clause}
            ORDER BY CREATED_AT DESC
            """

            if limit:
                if offset:
                    sql += f" OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
                else:
                    sql += f" FETCH FIRST {limit} ROWS ONLY"

            rows = self.db_manager.execute_query(sql, params)

            data_points = []
            for row in rows:
                try:
                    data_point = ReasoningDataPoint(
                        id=row[0],
                        category=row[1],
                        difficulty=row[2],
                        question=row[3],
                        correct_answer=row[4],
                        explanation=row[5],
                        options=json.loads(row[6]) if row[6] else None,
                        source=row[7],
                        created_at=row[8],
                        updated_at=row[9],
                        metadata=json.loads(row[10]) if row[10] else None
                    )
                    data_points.append(data_point)
                except Exception as e:
                    logger.warning(f"데이터 포인트 파싱 오류 (ID: {row[0] if row else 'Unknown'}): {e}")
                    continue

            return data_points

        except Exception as e:
            logger.error(f"데이터 조회 오류: {e}")
            return []

    def get_data_by_id(self, data_id: str) -> Optional[ReasoningDataPoint]:
        """ID로 데이터 조회 (개선된 예외 처리)"""
        try:
            sql = f"""
            SELECT ID, CATEGORY, DIFFICULTY, QUESTION, CORRECT_ANSWER,
                   EXPLANATION, OPTIONS, SOURCE, 
                   TO_CHAR(CREATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as CREATED_AT,
                   TO_CHAR(UPDATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as UPDATED_AT,
                   METADATA
            FROM {Constants.TABLE_REASONING_DATA}
            WHERE ID = :1
            """

            rows = self.db_manager.execute_query(sql, [data_id])
            if rows:
                row = rows[0]
                return ReasoningDataPoint(
                    id=row[0],
                    category=row[1],
                    difficulty=row[2],
                    question=row[3],
                    correct_answer=row[4],
                    explanation=row[5],
                    options=json.loads(row[6]) if row[6] else None,
                    source=row[7],
                    created_at=row[8],
                    updated_at=row[9],
                    metadata=json.loads(row[10]) if row[10] else None
                )
            return None
        except Exception as e:
            logger.error(f"ID로 데이터 조회 오류: {e}")
            return None

    def delete_data_point(self, data_id: str) -> bool:
        """데이터 포인트 삭제 (개선된 예외 처리)"""
        try:
            sql = f"DELETE FROM {Constants.TABLE_REASONING_DATA} WHERE ID = :1"
            rowcount = self.db_manager.execute_dml(sql, [data_id])

            if rowcount > 0:
                logger.info(f"데이터 삭제 완료: {data_id}")
                return True
            else:
                logger.warning(f"삭제할 데이터를 찾을 수 없음: {data_id}")
                return False

        except Exception as e:
            logger.error(f"데이터 삭제 오류: {e}")
            return False

    def get_statistics(self) -> DatasetStatistics:
        """데이터셋 통계 정보 (개선된 예외 처리)"""
        try:
            # 전체 개수
            total_sql = f"SELECT COUNT(*) FROM {Constants.TABLE_REASONING_DATA}"
            total_count = self.db_manager.execute_query(total_sql)[0][0]

            # 카테고리별 개수
            category_sql = f"""
            SELECT CATEGORY, COUNT(*) 
            FROM {Constants.TABLE_REASONING_DATA} 
            GROUP BY CATEGORY
            """
            category_rows = self.db_manager.execute_query(category_sql)
            category_counts = dict(category_rows)

            # 난이도별 개수
            difficulty_sql = f"""
            SELECT DIFFICULTY, COUNT(*) 
            FROM {Constants.TABLE_REASONING_DATA} 
            GROUP BY DIFFICULTY
            """
            difficulty_rows = self.db_manager.execute_query(difficulty_sql)
            difficulty_counts = dict(difficulty_rows)

            # 소스별 개수
            source_sql = f"""
            SELECT NVL(SOURCE, 'unknown'), COUNT(*) 
            FROM {Constants.TABLE_REASONING_DATA} 
            GROUP BY SOURCE
            """
            source_rows = self.db_manager.execute_query(source_sql)
            source_counts = dict(source_rows)

            return DatasetStatistics(
                total_count=total_count,
                category_counts=category_counts,
                difficulty_counts=difficulty_counts,
                source_counts=source_counts,
                created_at=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"통계 조회 오류: {e}")
            return DatasetStatistics(
                total_count=0,
                category_counts={},
                difficulty_counts={},
                source_counts={},
                created_at=datetime.now().isoformat()
            )

    def export_to_json_streaming(self, file_path: str, category: Optional[str] = None) -> bool:
        """스트리밍 방식의 JSON 내보내기 (메모리 효율적)"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('[\n')

                first_item = True
                total_exported = 0

                for batch in self.get_data_iterator(category=category):
                    for item in batch:
                        if not first_item:
                            f.write(',\n')
                        else:
                            first_item = False

                        json.dump(item.to_dict(), f, ensure_ascii=False, indent=2)
                        total_exported += 1

                f.write('\n]')

            logger.info(f"스트리밍 JSON 내보내기 완료: {file_path} ({total_exported}개)")
            return True

        except Exception as e:
            logger.error(f"스트리밍 JSON 내보내기 오류: {e}")
            return False

    def export_to_json(self, file_path: str, category: Optional[str] = None) -> bool:
        """데이터를 JSON 파일로 내보내기 (개선된 예외 처리)"""
        try:
            data = self.get_data(category=category)

            # ReasoningDataPoint 객체를 딕셔너리로 변환
            export_data = [item.to_dict() for item in data]

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"JSON 내보내기 완료: {file_path} ({len(export_data)}개)")
            return True

        except Exception as e:
            logger.error(f"JSON 내보내기 오류: {e}")
            return False

    def export_to_csv(self, file_path: str, category: Optional[str] = None) -> bool:
        """데이터를 CSV 파일로 내보내기 (개선된 예외 처리)"""
        try:
            data = self.get_data(category=category)

            if not data:
                logger.warning("내보낼 데이터가 없습니다.")
                return False

            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                fieldnames = list(data[0].to_dict().keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                writer.writeheader()
                for item in data:
                    item_dict = item.to_dict()
                    # options와 metadata를 JSON 문자열로 변환
                    if item_dict['options']:
                        item_dict['options'] = json.dumps(item_dict['options'], ensure_ascii=False)
                    if item_dict['metadata']:
                        item_dict['metadata'] = json.dumps(item_dict['metadata'], ensure_ascii=False)
                    writer.writerow(item_dict)

            logger.info(f"CSV 내보내기 완료: {file_path} ({len(data)}개)")
            return True

        except Exception as e:
            logger.error(f"CSV 내보내기 오류: {e}")
            return False

    def load_from_json(self, file_path: str) -> int:
        """JSON 파일에서 데이터 로드 (개선된 예외 처리)"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.error("JSON 파일이 리스트 형태가 아닙니다.")
                return 0

            data_points = []
            for i, item in enumerate(data):
                try:
                    data_point = ReasoningDataPoint.from_dict(item)
                    data_points.append(data_point)
                except Exception as e:
                    logger.warning(f"항목 {i} 파싱 실패: {e}")
                    continue

            if data_points:
                return self.add_batch_data_points(data_points)

            return 0

        except FileNotFoundError:
            logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return 0
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            return 0
        except Exception as e:
            logger.error(f"JSON 파일 로드 오류: {e}")
            return 0

    def load_from_csv(self, file_path: str) -> int:
        """CSV 파일에서 데이터 로드 (개선된 예외 처리)"""
        try:
            data_points = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for i, row in enumerate(reader):
                    try:
                        # options와 metadata는 JSON 문자열로 저장된 경우 파싱
                        if row.get('options') and row['options'].strip():
                            row['options'] = json.loads(row['options'])
                        else:
                            row['options'] = None

                        if row.get('metadata') and row['metadata'].strip():
                            row['metadata'] = json.loads(row['metadata'])
                        else:
                            row['metadata'] = None

                        data_point = ReasoningDataPoint.from_dict(row)
                        data_points.append(data_point)

                    except Exception as e:
                        logger.warning(f"CSV 행 {i+1} 파싱 실패: {e}")
                        continue

            if data_points:
                return self.add_batch_data_points(data_points)

            return 0

        except FileNotFoundError:
            logger.error(f"파일을 찾을 수 없습니다: {file_path}")
            return 0
        except Exception as e:
            logger.error(f"CSV 파일 로드 오류: {e}")
            return 0

    def get_operation_stats(self) -> Dict[str, Any]:
        """작업 통계 반환"""
        total = self._operation_stats['total_operations']
        success_rate = (self._operation_stats['successful_operations'] / total * 100) if total > 0 else 0

        return {
            'total_operations': total,
            'successful_operations': self._operation_stats['successful_operations'],
            'failed_operations': self._operation_stats['failed_operations'],
            'success_rate_percent': round(success_rate, 2)
        }

    def reset_operation_stats(self):
        """작업 통계 리셋"""
        self._operation_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0
        }

    def save_statistics(self, stats: DatasetStatistics) -> bool:
        """통계 정보 저장 (개선된 예외 처리)"""
        try:
            sql = f"""
            INSERT INTO {Constants.TABLE_DATASET_STATS}
            (ID, TOTAL_COUNT, CATEGORY_COUNTS, DIFFICULTY_COUNTS, SOURCE_COUNTS, CREATED_AT)
            VALUES (SEQ_DATASET_STATS.NEXTVAL, :1, :2, :3, :4, CURRENT_TIMESTAMP)
            """

            params = [
                stats.total_count,
                json.dumps(stats.category_counts, ensure_ascii=False),
                json.dumps(stats.difficulty_counts, ensure_ascii=False),
                json.dumps(stats.source_counts, ensure_ascii=False)
            ]

            self.db_manager.execute_dml(sql, params)
            logger.info("통계 정보 저장 완료")
            return True

        except Exception as e:
            logger.error(f"통계 저장 오류: {e}")
            return False

    def cleanup_old_data(self, days_old: int = 30) -> int:
        """오래된 데이터 정리"""
        try:
            sql = f"""
            DELETE FROM {Constants.TABLE_REASONING_DATA}
            WHERE CREATED_AT < SYSDATE - :1
            AND SOURCE LIKE '%temp%'
            """

            rowcount = self.db_manager.execute_dml(sql, [days_old])
            logger.info(f"오래된 임시 데이터 {rowcount}개 정리 완료")
            return rowcount

        except Exception as e:
            logger.error(f"데이터 정리 오류: {e}")
            return 0

    def backup_data(self, backup_file: str, compress: bool = True) -> bool:
        """데이터 백업"""
        try:
            logger.info("데이터 백업 시작...")

            # 전체 데이터 조회
            all_data = []
            for batch in self.get_data_iterator():
                all_data.extend([item.to_dict() for item in batch])

                # 메모리 관리
                if len(all_data) % 10000 == 0:
                    gc.collect()

            # 백업 파일 저장
            if compress:
                import gzip
                with gzip.open(f"{backup_file}.gz", 'wt', encoding='utf-8') as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=1)
                logger.info(f"압축 백업 완료: {backup_file}.gz ({len(all_data)}개)")
            else:
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(all_data, f, ensure_ascii=False, indent=2)
                logger.info(f"백업 완료: {backup_file} ({len(all_data)}개)")

            return True

        except Exception as e:
            logger.error(f"데이터 백업 오류: {e}")
            return False

    def restore_from_backup(self, backup_file: str) -> int:
        """백업에서 데이터 복원"""
        try:
            logger.info("백업에서 데이터 복원 시작...")

            # 백업 파일 읽기
            if backup_file.endswith('.gz'):
                import gzip
                with gzip.open(backup_file, 'rt', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(backup_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

            # 데이터 복원
            data_points = []
            for item in data:
                try:
                    data_point = ReasoningDataPoint.from_dict(item)
                    data_points.append(data_point)
                except Exception as e:
                    logger.warning(f"백업 항목 파싱 실패: {e}")
                    continue

            if data_points:
                restored_count = self.add_batch_data_points(data_points)
                logger.info(f"백업 복원 완료: {restored_count}개")
                return restored_count

            return 0

        except Exception as e:
            logger.error(f"백업 복원 오류: {e}")
            return 0

    def get_data_quality_report(self) -> Dict[str, Any]:
        """데이터 품질 리포트 생성"""
        try:
            report = {
                'total_records': 0,
                'data_quality_issues': [],
                'category_distribution': {},
                'difficulty_distribution': {},
                'average_question_length': 0,
                'average_answer_length': 0,
                'records_without_explanation': 0,
                'records_with_options': 0,
                'duplicate_questions': 0
            }

            # 기본 통계
            stats = self.get_statistics()
            report['total_records'] = stats.total_count
            report['category_distribution'] = stats.category_counts
            report['difficulty_distribution'] = stats.difficulty_counts

            # 상세 품질 분석 (샘플링)
            sample_data = self.get_data(limit=1000)

            if sample_data:
                question_lengths = []
                answer_lengths = []
                without_explanation = 0
                with_options = 0
                questions_seen = set()
                duplicates = 0

                for item in sample_data:
                    question_lengths.append(len(item.question))
                    answer_lengths.append(len(item.correct_answer))

                    if not item.explanation:
                        without_explanation += 1

                    if item.options:
                        with_options += 1

                    # 중복 검사
                    if item.question in questions_seen:
                        duplicates += 1
                    else:
                        questions_seen.add(item.question)

                report['average_question_length'] = sum(question_lengths) / len(question_lengths)
                report['average_answer_length'] = sum(answer_lengths) / len(answer_lengths)
                report['records_without_explanation'] = without_explanation
                report['records_with_options'] = with_options
                report['duplicate_questions'] = duplicates

                # 품질 이슈 식별
                if without_explanation > len(sample_data) * 0.5:
                    report['data_quality_issues'].append("50% 이상의 레코드에 설명이 없습니다")

                if duplicates > 0:
                    report['data_quality_issues'].append(f"{duplicates}개의 중복 질문이 발견되었습니다")

                avg_q_len = report['average_question_length']
                if avg_q_len < 10:
                    report['data_quality_issues'].append("질문 길이가 너무 짧습니다")
                elif avg_q_len > 5000:
                    report['data_quality_issues'].append("질문 길이가 너무 깁니다")

            return report

        except Exception as e:
            logger.error(f"데이터 품질 리포트 생성 오류: {e}")
            return {}

    def optimize_storage(self) -> Dict[str, Any]:
        """스토리지 최적화"""
        try:
            optimization_results = {
                'deleted_duplicates': 0,
                'compressed_data': 0,
                'freed_space_mb': 0
            }

            # 중복 데이터 제거
            duplicate_sql = f"""
            DELETE FROM {Constants.TABLE_REASONING_DATA} a
            WHERE a.ROWID > (
                SELECT MIN(b.ROWID)
                FROM {Constants.TABLE_REASONING_DATA} b
                WHERE a.QUESTION = b.QUESTION 
                AND a.CORRECT_ANSWER = b.CORRECT_ANSWER
            )
            """

            deleted_count = self.db_manager.execute_dml(duplicate_sql)
            optimization_results['deleted_duplicates'] = deleted_count

            logger.info(f"스토리지 최적화 완료: 중복 {deleted_count}개 제거")
            return optimization_results

        except Exception as e:
            logger.error(f"스토리지 최적화 오류: {e}")
            return {}