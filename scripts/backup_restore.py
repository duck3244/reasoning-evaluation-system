#!/usr/bin/env python3
"""
백업 및 복원 스크립트
데이터베이스 데이터를 백업하고 복원하는 기능을 제공합니다.
"""
import sys
import os
import argparse
import logging
import gzip
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database_config import DatabaseConfig, DatabaseManager, load_db_config_from_file
from core.data_collector import ReasoningDatasetCollector
from core.data_models import ReasoningDataPoint, Constants
from monitoring.logging_system import setup_application_logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BackupRestoreManager:
    """백업 및 복원 관리자"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.collector = ReasoningDatasetCollector(db_manager)

    def create_backup(self,
                      backup_path: str,
                      include_tables: List[str] = None,
                      compress: bool = True,
                      include_metadata: bool = True) -> Dict[str, Any]:
        """전체 백업 생성"""
        logger.info(f"백업 생성 시작: {backup_path}")

        if include_tables is None:
            include_tables = [
                Constants.TABLE_REASONING_DATA,
                Constants.TABLE_EVALUATION_RESULTS,
                Constants.TABLE_DATASET_STATS
            ]

        backup_info = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'tables': {},
            'metadata': {}
        }

        try:
            # 백업 디렉토리 생성
            backup_dir = Path(backup_path).parent
            backup_dir.mkdir(parents=True, exist_ok=True)

            # 각 테이블 백업
            for table_name in include_tables:
                logger.info(f"테이블 백업 중: {table_name}")

                table_data = self._backup_table(table_name)
                backup_info['tables'][table_name] = {
                    'record_count': len(table_data),
                    'data': table_data
                }

                logger.info(f"  {table_name}: {len(table_data)}개 레코드")

            # 메타데이터 포함
            if include_metadata:
                backup_info['metadata'] = self._collect_metadata()

            # 백업 파일 저장
            if compress:
                self._save_compressed_backup(backup_path, backup_info)
            else:
                self._save_backup(backup_path, backup_info)

            total_records = sum(info['record_count'] for info in backup_info['tables'].values())
            logger.info(f"✅ 백업 생성 완료: {backup_path} (총 {total_records}개 레코드)")

            return {
                'success': True,
                'backup_file': backup_path,
                'total_records': total_records,
                'tables': list(backup_info['tables'].keys()),
                'compressed': compress,
                'size_mb': self._get_file_size_mb(backup_path)
            }

        except Exception as e:
            logger.error(f"❌ 백업 생성 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _backup_table(self, table_name: str) -> List[Dict[str, Any]]:
        """개별 테이블 백업"""
        if table_name == Constants.TABLE_REASONING_DATA:
            return self._backup_reasoning_data()
        elif table_name == Constants.TABLE_EVALUATION_RESULTS:
            return self._backup_evaluation_results()
        elif table_name == Constants.TABLE_DATASET_STATS:
            return self._backup_dataset_stats()
        else:
            logger.warning(f"알 수 없는 테이블: {table_name}")
            return []

    def _backup_reasoning_data(self) -> List[Dict[str, Any]]:
        """추론 데이터 테이블 백업"""
        try:
            data_points = []

            # 배치 단위로 데이터 조회
            for batch in self.collector.get_data_iterator(batch_size=1000):
                for item in batch:
                    data_points.append(item.to_dict())

            return data_points

        except Exception as e:
            logger.error(f"추론 데이터 백업 실패: {e}")
            return []

    def _backup_evaluation_results(self) -> List[Dict[str, Any]]:
        """평가 결과 테이블 백업"""
        try:
            sql = f"""
            SELECT ID, DATA_POINT_ID, MODEL_NAME, PREDICTED_ANSWER, IS_CORRECT,
                   CONFIDENCE_SCORE, REASONING_STEPS, EXECUTION_TIME,
                   TO_CHAR(CREATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as CREATED_AT,
                   METADATA
            FROM {Constants.TABLE_EVALUATION_RESULTS}
            ORDER BY CREATED_AT
            """

            rows = self.db_manager.execute_query(sql)

            results = []
            for row in rows:
                result = {
                    'id': row[0],
                    'data_point_id': row[1],
                    'model_name': row[2],
                    'predicted_answer': row[3],
                    'is_correct': bool(row[4]),
                    'confidence_score': row[5],
                    'reasoning_steps': json.loads(row[6]) if row[6] else None,
                    'execution_time': row[7],
                    'created_at': row[8],
                    'metadata': json.loads(row[9]) if row[9] else None
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"평가 결과 백업 실패: {e}")
            return []

    def _backup_dataset_stats(self) -> List[Dict[str, Any]]:
        """데이터셋 통계 테이블 백업"""
        try:
            sql = f"""
            SELECT ID, TOTAL_COUNT, CATEGORY_COUNTS, DIFFICULTY_COUNTS, SOURCE_COUNTS,
                   TO_CHAR(CREATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as CREATED_AT
            FROM {Constants.TABLE_DATASET_STATS}
            ORDER BY CREATED_AT
            """

            rows = self.db_manager.execute_query(sql)

            stats = []
            for row in rows:
                stat = {
                    'id': row[0],
                    'total_count': row[1],
                    'category_counts': json.loads(row[2]) if row[2] else {},
                    'difficulty_counts': json.loads(row[3]) if row[3] else {},
                    'source_counts': json.loads(row[4]) if row[4] else {},
                    'created_at': row[5]
                }
                stats.append(stat)

            return stats

        except Exception as e:
            logger.error(f"데이터셋 통계 백업 실패: {e}")
            return []

    def _collect_metadata(self) -> Dict[str, Any]:
        """메타데이터 수집"""
        metadata = {
            'database_version': 'Oracle',
            'backup_created_by': 'LLM Reasoning Evaluation System',
            'system_info': {}
        }

        try:
            # 데이터베이스 버전 정보
            version_sql = "SELECT BANNER FROM V$VERSION WHERE ROWNUM = 1"
            version_result = self.db_manager.execute_query(version_sql)
            if version_result:
                metadata['database_version'] = version_result[0][0]

            # 테이블 정보
            table_info_sql = """
            SELECT TABLE_NAME, NUM_ROWS, LAST_ANALYZED
            FROM USER_TABLES
            WHERE TABLE_NAME IN ('REASONING_DATA', 'EVALUATION_RESULTS', 'DATASET_STATISTICS')
            """
            table_info = self.db_manager.execute_query(table_info_sql)
            metadata['table_info'] = [
                {'table_name': row[0], 'num_rows': row[1], 'last_analyzed': str(row[2]) if row[2] else None}
                for row in table_info
            ]

            # 시스템 통계
            stats = self.collector.get_statistics()
            metadata['system_stats'] = {
                'total_count': stats.total_count,
                'category_counts': stats.category_counts,
                'difficulty_counts': stats.difficulty_counts,
                'source_counts': stats.source_counts
            }

        except Exception as e:
            logger.warning(f"메타데이터 수집 중 오류: {e}")
            metadata['collection_error'] = str(e)

        return metadata

    def _save_backup(self, backup_path: str, backup_data: Dict[str, Any]):
        """백업 데이터 저장 (비압축)"""
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)

    def _save_compressed_backup(self, backup_path: str, backup_data: Dict[str, Any]):
        """백업 데이터 저장 (압축)"""
        if not backup_path.endswith('.gz'):
            backup_path += '.gz'

        with gzip.open(backup_path, 'wt', encoding='utf-8') as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=1)

    def _get_file_size_mb(self, file_path: str) -> float:
        """파일 크기 조회 (MB)"""
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except:
            return 0.0

    def restore_backup(self,
                       backup_path: str,
                       restore_tables: List[str] = None,
                       clear_existing: bool = False,
                       dry_run: bool = False) -> Dict[str, Any]:
        """백업 복원"""
        logger.info(f"백업 복원 {'시뮬레이션' if dry_run else '시작'}: {backup_path}")

        if not os.path.exists(backup_path):
            logger.error(f"백업 파일이 존재하지 않습니다: {backup_path}")
            return {'success': False, 'error': 'Backup file not found'}

        try:
            # 백업 파일 로드
            backup_data = self._load_backup(backup_path)

            if not backup_data:
                return {'success': False, 'error': 'Failed to load backup data'}

            # 백업 정보 확인
            logger.info(f"백업 정보:")
            logger.info(f"  생성 시간: {backup_data.get('timestamp', 'Unknown')}")
            logger.info(f"  버전: {backup_data.get('version', 'Unknown')}")
            logger.info(f"  테이블 수: {len(backup_data.get('tables', {}))}")

            # 복원할 테이블 결정
            available_tables = list(backup_data.get('tables', {}).keys())
            if restore_tables is None:
                restore_tables = available_tables
            else:
                # 요청된 테이블이 백업에 있는지 확인
                missing_tables = [t for t in restore_tables if t not in available_tables]
                if missing_tables:
                    logger.warning(f"백업에 없는 테이블: {missing_tables}")
                    restore_tables = [t for t in restore_tables if t in available_tables]

            if not restore_tables:
                return {'success': False, 'error': 'No tables to restore'}

            restore_results = {
                'success': True,
                'tables_restored': {},
                'total_records': 0
            }

            # 각 테이블 복원
            for table_name in restore_tables:
                table_data = backup_data['tables'][table_name]['data']

                if dry_run:
                    logger.info(f"[시뮬레이션] 테이블 복원: {table_name} ({len(table_data)}개 레코드)")
                    restore_results['tables_restored'][table_name] = len(table_data)
                else:
                    logger.info(f"테이블 복원 중: {table_name} ({len(table_data)}개 레코드)")

                    # 기존 데이터 삭제 (요청된 경우)
                    if clear_existing:
                        self._clear_table(table_name)

                    # 데이터 복원
                    restored_count = self._restore_table_data(table_name, table_data)
                    restore_results['tables_restored'][table_name] = restored_count

                    logger.info(f"  복원 완료: {restored_count}개 레코드")

                restore_results['total_records'] += restore_results['tables_restored'][table_name]

            logger.info(f"✅ 백업 복원 {'시뮬레이션' if dry_run else '완료'}: 총 {restore_results['total_records']}개 레코드")
            return restore_results

        except Exception as e:
            logger.error(f"❌ 백업 복원 실패: {e}")
            return {'success': False, 'error': str(e)}

    def _load_backup(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """백업 파일 로드"""
        try:
            if backup_path.endswith('.gz'):
                with gzip.open(backup_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(backup_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"백업 파일 로드 실패: {e}")
            return None

    def _clear_table(self, table_name: str):
        """테이블 데이터 삭제"""
        logger.warning(f"테이블 데이터 삭제: {table_name}")

        try:
            delete_sql = f"DELETE FROM {table_name}"
            deleted_count = self.db_manager.execute_dml(delete_sql)
            logger.info(f"  삭제된 레코드: {deleted_count}개")
        except Exception as e:
            logger.error(f"테이블 삭제 실패: {e}")
            raise

    def _restore_table_data(self, table_name: str, table_data: List[Dict[str, Any]]) -> int:
        """테이블 데이터 복원"""
        if table_name == Constants.TABLE_REASONING_DATA:
            return self._restore_reasoning_data(table_data)
        elif table_name == Constants.TABLE_EVALUATION_RESULTS:
            return self._restore_evaluation_results(table_data)
        elif table_name == Constants.TABLE_DATASET_STATS:
            return self._restore_dataset_stats(table_data)
        else:
            logger.warning(f"알 수 없는 테이블: {table_name}")
            return 0

    def _restore_reasoning_data(self, data: List[Dict[str, Any]]) -> int:
        """추론 데이터 복원"""
        try:
            data_points = []
            for item in data:
                try:
                    data_point = ReasoningDataPoint.from_dict(item)
                    data_points.append(data_point)
                except Exception as e:
                    logger.warning(f"데이터 포인트 파싱 실패: {e}")
                    continue

            if data_points:
                return self.collector.add_batch_data_points(data_points)
            return 0

        except Exception as e:
            logger.error(f"추론 데이터 복원 실패: {e}")
            return 0

    def _restore_evaluation_results(self, data: List[Dict[str, Any]]) -> int:
        """평가 결과 복원"""
        try:
            if not data:
                return 0

            sql = f"""
            INSERT INTO {Constants.TABLE_EVALUATION_RESULTS}
            (ID, DATA_POINT_ID, MODEL_NAME, PREDICTED_ANSWER, IS_CORRECT,
             CONFIDENCE_SCORE, REASONING_STEPS, EXECUTION_TIME, CREATED_AT, METADATA)
            VALUES (:1, :2, :3, :4, :5, :6, :7, :8, 
                   TO_TIMESTAMP(:9, 'YYYY-MM-DD"T"HH24:MI:SS'), :10)
            """

            params_list = []
            for item in data:
                params = [
                    item.get('id'),
                    item.get('data_point_id'),
                    item.get('model_name'),
                    item.get('predicted_answer'),
                    1 if item.get('is_correct') else 0,
                    item.get('confidence_score'),
                    json.dumps(item.get('reasoning_steps')) if item.get('reasoning_steps') else None,
                    item.get('execution_time'),
                    item.get('created_at'),
                    json.dumps(item.get('metadata')) if item.get('metadata') else None
                ]
                params_list.append(params)

            return self.db_manager.execute_batch_dml(sql, params_list)

        except Exception as e:
            logger.error(f"평가 결과 복원 실패: {e}")
            return 0

    def _restore_dataset_stats(self, data: List[Dict[str, Any]]) -> int:
        """데이터셋 통계 복원"""
        try:
            if not data:
                return 0

            sql = f"""
            INSERT INTO {Constants.TABLE_DATASET_STATS}
            (ID, TOTAL_COUNT, CATEGORY_COUNTS, DIFFICULTY_COUNTS, SOURCE_COUNTS, CREATED_AT)
            VALUES (:1, :2, :3, :4, :5, TO_TIMESTAMP(:6, 'YYYY-MM-DD"T"HH24:MI:SS'))
            """

            params_list = []
            for item in data:
                params = [
                    item.get('id'),
                    item.get('total_count'),
                    json.dumps(item.get('category_counts', {})),
                    json.dumps(item.get('difficulty_counts', {})),
                    json.dumps(item.get('source_counts', {})),
                    item.get('created_at')
                ]
                params_list.append(params)

            return self.db_manager.execute_batch_dml(sql, params_list)

        except Exception as e:
            logger.error(f"데이터셋 통계 복원 실패: {e}")
            return 0

    def list_backups(self, backup_dir: str) -> List[Dict[str, Any]]:
        """백업 목록 조회"""
        backup_list = []

        try:
            backup_path = Path(backup_dir)
            if not backup_path.exists():
                logger.warning(f"백업 디렉토리가 존재하지 않습니다: {backup_dir}")
                return backup_list

            # 백업 파일 검색
            patterns = ['*.json', '*.json.gz']
            backup_files = []

            for pattern in patterns:
                backup_files.extend(backup_path.glob(pattern))

            for backup_file in sorted(backup_files):
                try:
                    # 백업 파일 정보 로드
                    backup_data = self._load_backup(str(backup_file))
                    if backup_data:
                        file_info = {
                            'filename': backup_file.name,
                            'path': str(backup_file),
                            'size_mb': self._get_file_size_mb(str(backup_file)),
                            'created': backup_data.get('timestamp', 'Unknown'),
                            'version': backup_data.get('version', 'Unknown'),
                            'tables': list(backup_data.get('tables', {}).keys()),
                            'total_records': sum(
                                info.get('record_count', 0)
                                for info in backup_data.get('tables', {}).values()
                            ),
                            'compressed': backup_file.name.endswith('.gz')
                        }
                        backup_list.append(file_info)

                except Exception as e:
                    logger.warning(f"백업 파일 정보 로드 실패 ({backup_file}): {e}")
                    continue

        except Exception as e:
            logger.error(f"백업 목록 조회 실패: {e}")

        return backup_list

    def verify_backup(self, backup_path: str) -> Dict[str, Any]:
        """백업 파일 검증"""
        logger.info(f"백업 파일 검증: {backup_path}")

        verification_result = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }

        try:
            # 파일 존재 확인
            if not os.path.exists(backup_path):
                verification_result['errors'].append("백업 파일이 존재하지 않습니다")
                return verification_result

            # 백업 데이터 로드
            backup_data = self._load_backup(backup_path)
            if not backup_data:
                verification_result['errors'].append("백업 파일을 읽을 수 없습니다")
                return verification_result

            # 필수 필드 확인
            required_fields = ['timestamp', 'version', 'tables']
            for field in required_fields:
                if field not in backup_data:
                    verification_result['errors'].append(f"필수 필드 누락: {field}")

            # 테이블 데이터 검증
            tables = backup_data.get('tables', {})
            if not tables:
                verification_result['errors'].append("백업에 테이블 데이터가 없습니다")
            else:
                total_records = 0
                for table_name, table_info in tables.items():
                    if 'data' not in table_info:
                        verification_result['errors'].append(f"테이블 {table_name}에 데이터가 없습니다")
                    else:
                        record_count = len(table_info['data'])
                        total_records += record_count

                        # 샘플 데이터 검증
                        if record_count > 0:
                            sample_record = table_info['data'][0]
                            if not isinstance(sample_record, dict):
                                verification_result['warnings'].append(
                                    f"테이블 {table_name}의 데이터 형식이 올바르지 않습니다"
                                )

                verification_result['info']['total_records'] = total_records
                verification_result['info']['table_count'] = len(tables)

            # 버전 호환성 확인
            version = backup_data.get('version', '1.0')
            if version != '1.0':
                verification_result['warnings'].append(f"지원하지 않는 백업 버전: {version}")

            # 검증 결과 결정
            verification_result['valid'] = len(verification_result['errors']) == 0

            if verification_result['valid']:
                logger.info("✅ 백업 파일 검증 통과")
            else:
                logger.error("❌ 백업 파일 검증 실패")
                for error in verification_result['errors']:
                    logger.error(f"  - {error}")

            return verification_result

        except Exception as e:
            verification_result['errors'].append(f"검증 중 오류 발생: {str(e)}")
            logger.error(f"백업 파일 검증 실패: {e}")
            return verification_result

    def create_incremental_backup(self,
                                  backup_path: str,
                                  last_backup_date: str,
                                  compress: bool = True) -> Dict[str, Any]:
        """증분 백업 생성"""
        logger.info(f"증분 백업 생성: {backup_path} (기준일: {last_backup_date})")

        try:
            # 기준일 이후 데이터 조회
            incremental_data = self._get_incremental_data(last_backup_date)

            backup_info = {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'backup_type': 'incremental',
                'base_date': last_backup_date,
                'tables': incremental_data
            }

            # 백업 파일 저장
            if compress:
                self._save_compressed_backup(backup_path, backup_info)
            else:
                self._save_backup(backup_path, backup_info)

            total_records = sum(len(data['data']) for data in incremental_data.values())

            logger.info(f"✅ 증분 백업 생성 완료: {total_records}개 레코드")

            return {
                'success': True,
                'backup_file': backup_path,
                'total_records': total_records,
                'base_date': last_backup_date,
                'compressed': compress
            }

        except Exception as e:
            logger.error(f"❌ 증분 백업 생성 실패: {e}")
            return {'success': False, 'error': str(e)}

    def _get_incremental_data(self, base_date: str) -> Dict[str, Dict[str, Any]]:
        """증분 데이터 조회"""
        incremental_data = {}

        try:
            # 추론 데이터 (생성일 기준)
            reasoning_sql = f"""
            SELECT ID, CATEGORY, DIFFICULTY, QUESTION, CORRECT_ANSWER,
                   EXPLANATION, OPTIONS, SOURCE, 
                   TO_CHAR(CREATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as CREATED_AT,
                   TO_CHAR(UPDATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as UPDATED_AT,
                   METADATA
            FROM {Constants.TABLE_REASONING_DATA}
            WHERE CREATED_AT > TO_TIMESTAMP(:1, 'YYYY-MM-DD"T"HH24:MI:SS')
            ORDER BY CREATED_AT
            """

            reasoning_rows = self.db_manager.execute_query(reasoning_sql, [base_date])
            reasoning_data = []

            for row in reasoning_rows:
                item = {
                    'id': row[0],
                    'category': row[1],
                    'difficulty': row[2],
                    'question': row[3],
                    'correct_answer': row[4],
                    'explanation': row[5],
                    'options': json.loads(row[6]) if row[6] else None,
                    'source': row[7],
                    'created_at': row[8],
                    'updated_at': row[9],
                    'metadata': json.loads(row[10]) if row[10] else None
                }
                reasoning_data.append(item)

            incremental_data[Constants.TABLE_REASONING_DATA] = {
                'record_count': len(reasoning_data),
                'data': reasoning_data
            }

            # 평가 결과 (생성일 기준)
            eval_sql = f"""
            SELECT ID, DATA_POINT_ID, MODEL_NAME, PREDICTED_ANSWER, IS_CORRECT,
                   CONFIDENCE_SCORE, REASONING_STEPS, EXECUTION_TIME,
                   TO_CHAR(CREATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as CREATED_AT,
                   METADATA
            FROM {Constants.TABLE_EVALUATION_RESULTS}
            WHERE CREATED_AT > TO_TIMESTAMP(:1, 'YYYY-MM-DD"T"HH24:MI:SS')
            ORDER BY CREATED_AT
            """

            eval_rows = self.db_manager.execute_query(eval_sql, [base_date])
            eval_data = []

            for row in eval_rows:
                item = {
                    'id': row[0],
                    'data_point_id': row[1],
                    'model_name': row[2],
                    'predicted_answer': row[3],
                    'is_correct': bool(row[4]),
                    'confidence_score': row[5],
                    'reasoning_steps': json.loads(row[6]) if row[6] else None,
                    'execution_time': row[7],
                    'created_at': row[8],
                    'metadata': json.loads(row[9]) if row[9] else None
                }
                eval_data.append(item)

            incremental_data[Constants.TABLE_EVALUATION_RESULTS] = {
                'record_count': len(eval_data),
                'data': eval_data
            }

        except Exception as e:
            logger.error(f"증분 데이터 조회 실패: {e}")

        return incremental_data


def backup_main():
    """백업 메인 함수"""
    parser = argparse.ArgumentParser(description="LLM 추론 평가 시스템 데이터 백업")

    parser.add_argument(
        "--config", "-c",
        default="config/db_config.json",
        help="데이터베이스 설정 파일 경로"
    )

    parser.add_argument(
        "--output", "-o",
        required=True,
        help="백업 파일 경로"
    )

    parser.add_argument(
        "--tables", "-t",
        nargs="*",
        choices=["reasoning_data", "evaluation_results", "dataset_stats", "all"],
        default=["all"],
        help="백업할 테이블 선택"
    )

    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="압축하지 않음"
    )

    parser.add_argument(
        "--incremental",
        help="증분 백업 생성 (기준 날짜: YYYY-MM-DDTHH:MM:SS)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력"
    )

    args = parser.parse_args()

    # 로깅 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_application_logging({
        'log_level': 'DEBUG' if args.verbose else 'INFO',
        'log_format': 'simple',
        'enable_console': True
    })

    try:
        # 데이터베이스 연결
        db_config = load_db_config_from_file(args.config)
        if not db_config.test_connection():
            logger.error("데이터베이스 연결 실패")
            return 1

        db_manager = DatabaseManager(db_config)
        backup_manager = BackupRestoreManager(db_manager)

        # 백업할 테이블 결정
        if "all" in args.tables:
            include_tables = [
                Constants.TABLE_REASONING_DATA,
                Constants.TABLE_EVALUATION_RESULTS,
                Constants.TABLE_DATASET_STATS
            ]
        else:
            table_mapping = {
                "reasoning_data": Constants.TABLE_REASONING_DATA,
                "evaluation_results": Constants.TABLE_EVALUATION_RESULTS,
                "dataset_stats": Constants.TABLE_DATASET_STATS
            }
            include_tables = [table_mapping[t] for t in args.tables if t in table_mapping]

        # 백업 실행
        if args.incremental:
            result = backup_manager.create_incremental_backup(
                args.output,
                args.incremental,
                compress=not args.no_compress
            )
        else:
            result = backup_manager.create_backup(
                args.output,
                include_tables=include_tables,
                compress=not args.no_compress
            )

        if result['success']:
            print(f"✅ 백업 성공: {result['backup_file']}")
            print(f"   총 레코드: {result['total_records']:,}개")
            if 'size_mb' in result:
                print(f"   파일 크기: {result['size_mb']:.2f} MB")
            return 0
        else:
            print(f"❌ 백업 실패: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"백업 프로세스 오류: {e}")
        return 1
    finally:
        try:
            if 'db_config' in locals():
                db_config.close_pool()
        except:
            pass


def restore_main():
    """복원 메인 함수"""
    parser = argparse.ArgumentParser(description="LLM 추론 평가 시스템 데이터 복원")

    parser.add_argument(
        "--config", "-c",
        default="config/db_config.json",
        help="데이터베이스 설정 파일 경로"
    )

    parser.add_argument(
        "--backup", "-b",
        required=True,
        help="복원할 백업 파일 경로"
    )

    parser.add_argument(
        "--tables", "-t",
        nargs="*",
        help="복원할 테이블 선택 (기본: 모든 테이블)"
    )

    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="기존 데이터 삭제 후 복원"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 복원 없이 시뮬레이션만 수행"
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="백업 파일 검증만 수행"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력"
    )

    args = parser.parse_args()

    # 로깅 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    setup_application_logging({
        'log_level': 'DEBUG' if args.verbose else 'INFO',
        'log_format': 'simple',
        'enable_console': True
    })

    try:
        # 데이터베이스 연결
        db_config = load_db_config_from_file(args.config)
        if not db_config.test_connection():
            logger.error("데이터베이스 연결 실패")
            return 1

        db_manager = DatabaseManager(db_config)
        backup_manager = BackupRestoreManager(db_manager)

        # 백업 파일 검증
        verification = backup_manager.verify_backup(args.backup)

        print(f"백업 파일 검증 결과: {'✅ 통과' if verification['valid'] else '❌ 실패'}")

        if verification['errors']:
            print("오류:")
            for error in verification['errors']:
                print(f"  - {error}")

        if verification['warnings']:
            print("경고:")
            for warning in verification['warnings']:
                print(f"  - {warning}")

        if verification['info']:
            print("백업 정보:")
            for key, value in verification['info'].items():
                print(f"  {key}: {value}")

        if args.verify_only:
            return 0 if verification['valid'] else 1

        if not verification['valid']:
            print("백업 파일 검증 실패로 복원을 중단합니다.")
            return 1

        # 복원 실행
        result = backup_manager.restore_backup(
            args.backup,
            restore_tables=args.tables,
            clear_existing=args.clear_existing,
            dry_run=args.dry_run
        )

        if result['success']:
            action = "시뮬레이션" if args.dry_run else "복원"
            print(f"✅ {action} 성공")
            print(f"   총 레코드: {result['total_records']:,}개")
            print("   테이블별 레코드:")
            for table, count in result['tables_restored'].items():
                print(f"     {table}: {count:,}개")
            return 0
        else:
            print(f"❌ 복원 실패: {result.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"복원 프로세스 오류: {e}")
        return 1
    finally:
        try:
            if 'db_config' in locals():
                db_config.close_pool()
        except:
            pass


def management_main():
    """백업 관리 메인 함수"""
    parser = argparse.ArgumentParser(description="LLM 추론 평가 시스템 백업 관리")

    parser.add_argument(
        "--list", "-l",
        help="백업 디렉토리의 백업 목록 조회"
    )

    parser.add_argument(
        "--verify",
        help="백업 파일 검증"
    )

    parser.add_argument(
        "--config", "-c",
        default="config/db_config.json",
        help="데이터베이스 설정 파일 경로"
    )

    args = parser.parse_args()

    setup_application_logging({
        'log_level': 'INFO',
        'log_format': 'simple',
        'enable_console': True
    })

    try:
        if args.list:
            # 백업 목록 조회
            db_config = load_db_config_from_file(args.config)
            db_manager = DatabaseManager(db_config)
            backup_manager = BackupRestoreManager(db_manager)

            backups = backup_manager.list_backups(args.list)

            if not backups:
                print(f"백업 디렉토리에 백업 파일이 없습니다: {args.list}")
                return 0

            print(f"백업 목록 ({args.list}):")
            print("=" * 80)
            print(f"{'파일명':<30} {'크기(MB)':<10} {'레코드':<10} {'생성일시':<20} {'압축'}")
            print("-" * 80)

            for backup in backups:
                compressed = "✓" if backup['compressed'] else "✗"
                print(f"{backup['filename']:<30} "
                      f"{backup['size_mb']:<10.2f} "
                      f"{backup['total_records']:<10,} "
                      f"{backup['created'][:19]:<20} "
                      f"{compressed}")

            print("=" * 80)
            print(f"총 {len(backups)}개 백업 파일")

        elif args.verify:
            # 백업 파일 검증
            db_config = load_db_config_from_file(args.config)
            db_manager = DatabaseManager(db_config)
            backup_manager = BackupRestoreManager(db_manager)

            verification = backup_manager.verify_backup(args.verify)

            print(f"백업 파일 검증: {args.verify}")
            print(f"결과: {'✅ 통과' if verification['valid'] else '❌ 실패'}")

            if verification['errors']:
                print("\n오류:")
                for error in verification['errors']:
                    print(f"  - {error}")

            if verification['warnings']:
                print("\n경고:")
                for warning in verification['warnings']:
                    print(f"  - {warning}")

            if verification['info']:
                print("\n백업 정보:")
                for key, value in verification['info'].items():
                    print(f"  {key}: {value}")

            return 0 if verification['valid'] else 1

        else:
            parser.print_help()
            return 0

    except Exception as e:
        logger.error(f"오류: {e}")
        return 1
    finally:
        try:
            if 'db_config' in locals():
                db_config.close_pool()
        except:
            pass


def main():
    """메인 함수"""
    import sys

    # 스크립트 이름이나 인수에 따라 적절한 함수 실행
    if len(sys.argv) > 1:
        if sys.argv[1] == "backup":
            # backup 서브커맨드 제거 후 backup_main 실행
            sys.argv.pop(1)
            return backup_main()
        elif sys.argv[1] == "restore":
            # restore 서브커맨드 제거 후 restore_main 실행
            sys.argv.pop(1)
            return restore_main()
        elif sys.argv[1] in ["list", "verify"]:
            # 관리 명령어는 management_main으로
            return management_main()

    # 기본적으로 도움말 표시
    print("LLM 추론 평가 시스템 백업/복원 도구")
    print()
    print("사용법:")
    print("  백업:     python backup_restore.py backup --output backup.json.gz")
    print("  복원:     python backup_restore.py restore --backup backup.json.gz")
    print("  목록:     python backup_restore.py list --list backups/")
    print("  검증:     python backup_restore.py verify --verify backup.json.gz")
    print()
    print("상세 옵션:")
    print("  python backup_restore.py backup --help")
    print("  python backup_restore.py restore --help")

    return 0


if __name__ == "__main__":
    exit(main())