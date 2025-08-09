"""
추론 성능 평가용 데이터 수집기
"""
import json
import csv
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from data_models import ReasoningDataPoint, DatasetStatistics, Constants
from database_config import DatabaseManager, DatabaseConfig

logger = logging.getLogger(__name__)


class ReasoningDatasetCollector:
    """추론 성능 평가용 데이터셋 수집 및 저장"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def add_data_point(self, data_point: ReasoningDataPoint) -> bool:
        """단일 데이터 포인트 추가"""
        try:
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
                       TO_TIMESTAMP(:10, 'YYYY-MM-DD"T"HH24:MI:SS.FF'), 
                       CURRENT_TIMESTAMP, :9)
            """

            params = [
                data_point.id,
                data_point.category,
                data_point.difficulty,
                data_point.question,
                data_point.correct_answer,
                data_point.explanation,
                json.dumps(data_point.options) if data_point.options else None,
                data_point.source,
                json.dumps(data_point.metadata) if data_point.metadata else None,
                data_point.created_at
            ]

            self.db_manager.execute_dml(sql, params)
            logger.info(f"데이터 포인트 추가/업데이트 완료: {data_point.id}")
            return True

        except Exception as e:
            logger.error(f"데이터 추가 오류: {e}")
            return False

    def add_batch_data_points(self, data_points: List[ReasoningDataPoint]) -> int:
        """여러 데이터 포인트 배치 추가"""
        try:
            sql = f"""
            INSERT INTO {Constants.TABLE_REASONING_DATA} 
            (ID, CATEGORY, DIFFICULTY, QUESTION, CORRECT_ANSWER, 
             EXPLANATION, OPTIONS, SOURCE, CREATED_AT, UPDATED_AT, METADATA)
            VALUES (:1, :2, :3, :4, :5, :6, :7, :8, 
                   TO_TIMESTAMP(:9, 'YYYY-MM-DD"T"HH24:MI:SS.FF'), 
                   CURRENT_TIMESTAMP, :10)
            """

            params_list = []
            for data_point in data_points:
                if not data_point.id:
                    data_point.id = data_point.generate_id()
                if not data_point.created_at:
                    data_point.created_at = datetime.now().isoformat()

                params = [
                    data_point.id,
                    data_point.category,
                    data_point.difficulty,
                    data_point.question,
                    data_point.correct_answer,
                    data_point.explanation,
                    json.dumps(data_point.options) if data_point.options else None,
                    data_point.source,
                    data_point.created_at,
                    json.dumps(data_point.metadata) if data_point.metadata else None
                ]
                params_list.append(params)

            rowcount = self.db_manager.execute_batch_dml(sql, params_list)
            logger.info(f"배치 데이터 추가 완료: {rowcount}개")
            return rowcount

        except Exception as e:
            logger.error(f"배치 데이터 추가 오류: {e}")
            return 0

    def get_data(self,
                 category: Optional[str] = None,
                 difficulty: Optional[str] = None,
                 source: Optional[str] = None,
                 limit: Optional[int] = None,
                 offset: Optional[int] = None) -> List[ReasoningDataPoint]:
        """조건에 따라 데이터 조회"""
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

            return data_points

        except Exception as e:
            logger.error(f"데이터 조회 오류: {e}")
            return []

    def get_data_by_id(self, data_id: str) -> Optional[ReasoningDataPoint]:
        """ID로 데이터 조회"""
        data_list = self.get_data()
        sql = f"""
        SELECT ID, CATEGORY, DIFFICULTY, QUESTION, CORRECT_ANSWER,
               EXPLANATION, OPTIONS, SOURCE, 
               TO_CHAR(CREATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as CREATED_AT,
               TO_CHAR(UPDATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as UPDATED_AT,
               METADATA
        FROM {Constants.TABLE_REASONING_DATA}
        WHERE ID = :1
        """

        try:
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
        """데이터 포인트 삭제"""
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
        """데이터셋 통계 정보"""
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

    def save_statistics(self, stats: DatasetStatistics) -> bool:
        """통계 정보 저장"""
        try:
            sql = f"""
            INSERT INTO {Constants.TABLE_DATASET_STATS}
            (ID, TOTAL_COUNT, CATEGORY_COUNTS, DIFFICULTY_COUNTS, SOURCE_COUNTS, CREATED_AT)
            VALUES (SEQ_DATASET_STATS.NEXTVAL, :1, :2, :3, :4, CURRENT_TIMESTAMP)
            """

            params = [
                stats.total_count,
                json.dumps(stats.category_counts),
                json.dumps(stats.difficulty_counts),
                json.dumps(stats.source_counts)
            ]

            self.db_manager.execute_dml(sql, params)
            logger.info("통계 정보 저장 완료")
            return True

        except Exception as e:
            logger.error(f"통계 저장 오류: {e}")
            return False

    def load_from_json(self, file_path: str) -> int:
        """JSON 파일에서 데이터 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            data_points = []
            for item in data:
                data_point = ReasoningDataPoint.from_dict(item)
                data_points.append(data_point)

            return self.add_batch_data_points(data_points)

        except Exception as e:
            logger.error(f"JSON 파일 로드 오류: {e}")
            return 0

    def load_from_csv(self, file_path: str) -> int:
        """CSV 파일에서 데이터 로드"""
        try:
            data_points = []
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # options와 metadata는 JSON 문자열로 저장된 경우 파싱
                    if row.get('options'):
                        row['options'] = json.loads(row['options'])
                    if row.get('metadata'):
                        row['metadata'] = json.loads(row['metadata'])

                    data_point = ReasoningDataPoint.from_dict(row)
                    data_points.append(data_point)

            return self.add_batch_data_points(data_points)

        except Exception as e:
            logger.error(f"CSV 파일 로드 오류: {e}")
            return 0

    def export_to_json(self, file_path: str, category: Optional[str] = None) -> bool:
        """데이터를 JSON 파일로 내보내기"""
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
        """데이터를 CSV 파일로 내보내기"""
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
                        item_dict['options'] = json.dumps(item_dict['options'])
                    if item_dict['metadata']:
                        item_dict['metadata'] = json.dumps(item_dict['metadata'])
                    writer.writerow(item_dict)

            logger.info(f"CSV 내보내기 완료: {file_path} ({len(data)}개)")
            return True

        except Exception as e:
            logger.error(f"CSV 내보내기 오류: {e}")
            return False