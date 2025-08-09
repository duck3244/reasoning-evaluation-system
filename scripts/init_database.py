#!/usr/bin/env python3
"""
데이터베이스 초기화 스크립트
Oracle 데이터베이스에 필요한 테이블, 인덱스, 시퀀스를 생성합니다.
"""
import sys
import os
import argparse
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database_config import DatabaseConfig, DatabaseManager, load_db_config_from_file
from core.data_models import Constants
from monitoring.logging_system import setup_application_logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """데이터베이스 초기화 클래스"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def create_tables(self, drop_existing: bool = False):
        """테이블 생성"""
        logger.info("테이블 생성 시작...")

        try:
            with self.db_manager.db_config.get_connection() as conn:
                cursor = conn.cursor()

                if drop_existing:
                    self._drop_existing_tables(cursor)

                # 테이블 생성
                self._create_reasoning_data_table(cursor)
                self._create_evaluation_results_table(cursor)
                self._create_dataset_statistics_table(cursor)
                self._create_system_logs_table(cursor)

                conn.commit()
                logger.info("모든 테이블 생성 완료")

        except Exception as e:
            logger.error(f"테이블 생성 실패: {e}")
            raise

    def _drop_existing_tables(self, cursor):
        """기존 테이블 삭제"""
        logger.warning("기존 테이블 삭제 중...")

        tables_to_drop = [
            Constants.TABLE_EVALUATION_RESULTS,
            Constants.TABLE_DATASET_STATS,
            Constants.TABLE_REASONING_DATA,
            "SYSTEM_LOGS"
        ]

        for table in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE {table} CASCADE CONSTRAINTS")
                logger.info(f"테이블 삭제: {table}")
            except Exception as e:
                logger.debug(f"테이블 {table} 삭제 실패 (존재하지 않을 수 있음): {e}")

    def _create_reasoning_data_table(self, cursor):
        """추론 데이터 테이블 생성"""
        logger.info("추론 데이터 테이블 생성 중...")

        create_sql = f"""
        CREATE TABLE {Constants.TABLE_REASONING_DATA} (
            ID VARCHAR2(32) NOT NULL,
            CATEGORY VARCHAR2(50) NOT NULL,
            DIFFICULTY VARCHAR2(20) NOT NULL,
            QUESTION CLOB NOT NULL,
            CORRECT_ANSWER CLOB NOT NULL,
            EXPLANATION CLOB,
            OPTIONS CLOB,
            SOURCE VARCHAR2(100),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            METADATA CLOB,
            CONSTRAINT PK_REASONING_DATA PRIMARY KEY (ID),
            CONSTRAINT CHK_CATEGORY CHECK (CATEGORY IN (
                'math', 'logic', 'common_sense', 'reading_comprehension', 
                'science', 'history', 'language', 'unknown'
            )),
            CONSTRAINT CHK_DIFFICULTY CHECK (DIFFICULTY IN ('easy', 'medium', 'hard'))
        )
        """

        try:
            cursor.execute(create_sql)
            logger.info(f"테이블 생성 완료: {Constants.TABLE_REASONING_DATA}")
        except Exception as e:
            if "name is already used" in str(e).lower():
                logger.info(f"테이블이 이미 존재함: {Constants.TABLE_REASONING_DATA}")
            else:
                raise

        # 코멘트 추가
        self._add_table_comments(cursor, Constants.TABLE_REASONING_DATA)

    def _create_evaluation_results_table(self, cursor):
        """평가 결과 테이블 생성"""
        logger.info("평가 결과 테이블 생성 중...")

        create_sql = f"""
        CREATE TABLE {Constants.TABLE_EVALUATION_RESULTS} (
            ID VARCHAR2(64) NOT NULL,
            DATA_POINT_ID VARCHAR2(32) NOT NULL,
            MODEL_NAME VARCHAR2(100) NOT NULL,
            PREDICTED_ANSWER CLOB NOT NULL,
            IS_CORRECT NUMBER(1) NOT NULL,
            CONFIDENCE_SCORE NUMBER(5,4),
            REASONING_STEPS CLOB,
            EXECUTION_TIME NUMBER(10,3),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            METADATA CLOB,
            CONSTRAINT PK_EVALUATION_RESULTS PRIMARY KEY (ID),
            CONSTRAINT FK_EVAL_DATA_POINT FOREIGN KEY (DATA_POINT_ID) 
                REFERENCES {Constants.TABLE_REASONING_DATA}(ID) ON DELETE CASCADE,
            CONSTRAINT CHK_IS_CORRECT CHECK (IS_CORRECT IN (0, 1)),
            CONSTRAINT CHK_CONFIDENCE_SCORE CHECK (CONFIDENCE_SCORE BETWEEN 0 AND 1)
        )
        """

        try:
            cursor.execute(create_sql)
            logger.info(f"테이블 생성 완료: {Constants.TABLE_EVALUATION_RESULTS}")
        except Exception as e:
            if "name is already used" in str(e).lower():
                logger.info(f"테이블이 이미 존재함: {Constants.TABLE_EVALUATION_RESULTS}")
            else:
                raise

    def _create_dataset_statistics_table(self, cursor):
        """데이터셋 통계 테이블 생성"""
        logger.info("데이터셋 통계 테이블 생성 중...")

        create_sql = f"""
        CREATE TABLE {Constants.TABLE_DATASET_STATS} (
            ID NUMBER NOT NULL,
            TOTAL_COUNT NUMBER NOT NULL,
            CATEGORY_COUNTS CLOB,
            DIFFICULTY_COUNTS CLOB,
            SOURCE_COUNTS CLOB,
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
            CONSTRAINT PK_DATASET_STATS PRIMARY KEY (ID)
        )
        """

        try:
            cursor.execute(create_sql)
            logger.info(f"테이블 생성 완료: {Constants.TABLE_DATASET_STATS}")
        except Exception as e:
            if "name is already used" in str(e).lower():
                logger.info(f"테이블이 이미 존재함: {Constants.TABLE_DATASET_STATS}")
            else:
                raise

    def _create_system_logs_table(self, cursor):
        """시스템 로그 테이블 생성 (선택적)"""
        logger.info("시스템 로그 테이블 생성 중...")

        create_sql = """
        CREATE TABLE SYSTEM_LOGS (
            ID NUMBER NOT NULL,
            TIMESTAMP TIMESTAMP NOT NULL,
            LEVEL VARCHAR2(20) NOT NULL,
            LOGGER VARCHAR2(100),
            MESSAGE CLOB,
            MODULE VARCHAR2(100),
            FUNCTION VARCHAR2(100),
            LINE_NUMBER NUMBER,
            EXCEPTION_INFO CLOB,
            CONTEXT_DATA CLOB,
            CONSTRAINT PK_SYSTEM_LOGS PRIMARY KEY (ID)
        )
        """

        try:
            cursor.execute(create_sql)
            logger.info("테이블 생성 완료: SYSTEM_LOGS")
        except Exception as e:
            if "name is already used" in str(e).lower():
                logger.info("테이블이 이미 존재함: SYSTEM_LOGS")
            else:
                raise

    def _add_table_comments(self, cursor, table_name):
        """테이블 코멘트 추가"""
        comments = {
            Constants.TABLE_REASONING_DATA: "LLM 추론 평가용 문제 데이터",
            Constants.TABLE_EVALUATION_RESULTS: "모델 평가 결과",
            Constants.TABLE_DATASET_STATS: "데이터셋 통계 정보",
            "SYSTEM_LOGS": "시스템 로그"
        }

        if table_name in comments:
            try:
                cursor.execute(f"COMMENT ON TABLE {table_name} IS '{comments[table_name]}'")
            except Exception as e:
                logger.debug(f"코멘트 추가 실패: {e}")

    def create_indexes(self):
        """인덱스 생성"""
        logger.info("인덱스 생성 시작...")

        indexes = [
            # 추론 데이터 테이블 인덱스
            f"CREATE INDEX IDX_REASONING_CATEGORY ON {Constants.TABLE_REASONING_DATA}(CATEGORY)",
            f"CREATE INDEX IDX_REASONING_DIFFICULTY ON {Constants.TABLE_REASONING_DATA}(DIFFICULTY)",
            f"CREATE INDEX IDX_REASONING_SOURCE ON {Constants.TABLE_REASONING_DATA}(SOURCE)",
            f"CREATE INDEX IDX_REASONING_CREATED_AT ON {Constants.TABLE_REASONING_DATA}(CREATED_AT)",
            f"CREATE INDEX IDX_REASONING_CAT_DIFF ON {Constants.TABLE_REASONING_DATA}(CATEGORY, DIFFICULTY)",
            f"CREATE INDEX IDX_REASONING_CAT_SOURCE ON {Constants.TABLE_REASONING_DATA}(CATEGORY, SOURCE)",

            # 평가 결과 테이블 인덱스
            f"CREATE INDEX IDX_EVAL_MODEL ON {Constants.TABLE_EVALUATION_RESULTS}(MODEL_NAME)",
            f"CREATE INDEX IDX_EVAL_CORRECT ON {Constants.TABLE_EVALUATION_RESULTS}(IS_CORRECT)",
            f"CREATE INDEX IDX_EVAL_CREATED_AT ON {Constants.TABLE_EVALUATION_RESULTS}(CREATED_AT)",
            f"CREATE INDEX IDX_EVAL_MODEL_CORRECT ON {Constants.TABLE_EVALUATION_RESULTS}(MODEL_NAME, IS_CORRECT)",
            f"CREATE INDEX IDX_EVAL_DATA_POINT ON {Constants.TABLE_EVALUATION_RESULTS}(DATA_POINT_ID)",

            # 시스템 로그 인덱스
            "CREATE INDEX IDX_LOGS_TIMESTAMP ON SYSTEM_LOGS(TIMESTAMP)",
            "CREATE INDEX IDX_LOGS_LEVEL ON SYSTEM_LOGS(LEVEL)",
            "CREATE INDEX IDX_LOGS_MODULE ON SYSTEM_LOGS(MODULE)",
        ]

        created_count = 0
        for index_sql in indexes:
            try:
                self.db_manager.execute_dml(index_sql)
                index_name = index_sql.split()[2]
                logger.info(f"인덱스 생성: {index_name}")
                created_count += 1
            except Exception as e:
                if "name is already used" in str(e).lower():
                    continue
                logger.warning(f"인덱스 생성 실패: {e}")

        logger.info(f"인덱스 생성 완료: {created_count}개")

    def create_sequences(self):
        """시퀀스 생성"""
        logger.info("시퀀스 생성 시작...")

        sequences = [
            f"CREATE SEQUENCE SEQ_{Constants.TABLE_DATASET_STATS} START WITH 1 INCREMENT BY 1",
            "CREATE SEQUENCE SEQ_SYSTEM_LOGS START WITH 1 INCREMENT BY 1"
        ]

        created_count = 0
        for seq_sql in sequences:
            try:
                self.db_manager.execute_dml(seq_sql)
                seq_name = seq_sql.split()[2]
                logger.info(f"시퀀스 생성: {seq_name}")
                created_count += 1
            except Exception as e:
                if "name is already used" in str(e).lower():
                    continue
                logger.warning(f"시퀀스 생성 실패: {e}")

        logger.info(f"시퀀스 생성 완료: {created_count}개")

    def create_views(self):
        """뷰 생성"""
        logger.info("뷰 생성 시작...")

        # 데이터셋 요약 뷰
        view_sql = f"""
        CREATE OR REPLACE VIEW V_DATASET_SUMMARY AS
        SELECT 
            CATEGORY,
            DIFFICULTY,
            COUNT(*) as TOTAL_COUNT,
            COUNT(CASE WHEN SOURCE LIKE '%sample%' THEN 1 END) as SAMPLE_COUNT,
            MIN(CREATED_AT) as EARLIEST_CREATED,
            MAX(CREATED_AT) as LATEST_CREATED
        FROM {Constants.TABLE_REASONING_DATA}
        GROUP BY CATEGORY, DIFFICULTY
        ORDER BY CATEGORY, DIFFICULTY
        """

        try:
            self.db_manager.execute_dml(view_sql)
            logger.info("뷰 생성: V_DATASET_SUMMARY")
        except Exception as e:
            logger.warning(f"뷰 생성 실패: {e}")

        # 모델 성능 요약 뷰
        perf_view_sql = f"""
        CREATE OR REPLACE VIEW V_MODEL_PERFORMANCE AS
        SELECT 
            MODEL_NAME,
            COUNT(*) as TOTAL_EVALUATIONS,
            SUM(IS_CORRECT) as CORRECT_COUNT,
            ROUND(AVG(IS_CORRECT), 4) as ACCURACY,
            ROUND(AVG(EXECUTION_TIME), 3) as AVG_EXECUTION_TIME,
            MIN(CREATED_AT) as FIRST_EVALUATION,
            MAX(CREATED_AT) as LAST_EVALUATION
        FROM {Constants.TABLE_EVALUATION_RESULTS}
        GROUP BY MODEL_NAME
        ORDER BY ACCURACY DESC, TOTAL_EVALUATIONS DESC
        """

        try:
            self.db_manager.execute_dml(perf_view_sql)
            logger.info("뷰 생성: V_MODEL_PERFORMANCE")
        except Exception as e:
            logger.warning(f"모델 성능 뷰 생성 실패: {e}")

    def grant_permissions(self, username: str = None):
        """권한 부여 (선택적)"""
        if not username:
            logger.info("사용자명이 지정되지 않아 권한 부여를 건너뜁니다.")
            return

        logger.info(f"사용자 {username}에게 권한 부여 중...")

        permissions = [
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {Constants.TABLE_REASONING_DATA} TO {username}",
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {Constants.TABLE_EVALUATION_RESULTS} TO {username}",
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON {Constants.TABLE_DATASET_STATS} TO {username}",
            f"GRANT SELECT ON V_DATASET_SUMMARY TO {username}",
            f"GRANT SELECT ON V_MODEL_PERFORMANCE TO {username}",
        ]

        for perm_sql in permissions:
            try:
                self.db_manager.execute_dml(perm_sql)
                logger.info(f"권한 부여: {perm_sql}")
            except Exception as e:
                logger.warning(f"권한 부여 실패: {e}")

    def verify_installation(self):
        """설치 확인"""
        logger.info("데이터베이스 설치 확인 중...")

        # 테이블 존재 확인
        tables_to_check = [
            Constants.TABLE_REASONING_DATA,
            Constants.TABLE_EVALUATION_RESULTS,
            Constants.TABLE_DATASET_STATS
        ]

        try:
            for table in tables_to_check:
                count_sql = f"SELECT COUNT(*) FROM {table}"
                result = self.db_manager.execute_query(count_sql)
                count = result[0][0] if result else 0
                logger.info(f"테이블 {table}: {count}개 레코드")

            logger.info("✅ 데이터베이스 설치 확인 완료")
            return True

        except Exception as e:
            logger.error(f"❌ 데이터베이스 설치 확인 실패: {e}")
            return False

    def get_database_info(self):
        """데이터베이스 정보 조회"""
        logger.info("데이터베이스 정보 조회 중...")

        try:
            # 버전 정보
            version_sql = "SELECT BANNER FROM V$VERSION WHERE ROWNUM = 1"
            version_result = self.db_manager.execute_query(version_sql)
            if version_result:
                logger.info(f"Oracle 버전: {version_result[0][0]}")

            # 테이블스페이스 정보
            tablespace_sql = """
            SELECT TABLESPACE_NAME, ROUND(BYTES/1024/1024, 2) as SIZE_MB 
            FROM USER_SEGMENTS 
            WHERE SEGMENT_TYPE = 'TABLE'
            GROUP BY TABLESPACE_NAME
            """
            ts_result = self.db_manager.execute_query(tablespace_sql)
            if ts_result:
                logger.info("테이블스페이스 사용량:")
                for ts_name, size_mb in ts_result:
                    logger.info(f"  {ts_name}: {size_mb} MB")

        except Exception as e:
            logger.warning(f"데이터베이스 정보 조회 실패: {e}")


def create_sample_config():
    """샘플 설정 파일 생성"""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    config_file = config_dir / "db_config.json"
    if config_file.exists():
        logger.info(f"설정 파일이 이미 존재합니다: {config_file}")
        return str(config_file)

    sample_config = {
        "username": "your_username",
        "password": "your_password",
        "dsn": "localhost:1521/XE",
        "pool_min": 1,
        "pool_max": 10,
        "pool_increment": 1
    }

    with open(config_file, 'w', encoding='utf-8') as f:
        import json
        json.dump(sample_config, f, indent=2, ensure_ascii=False)

    logger.info(f"샘플 설정 파일이 생성되었습니다: {config_file}")
    logger.info("설정 파일을 편집하여 실제 데이터베이스 정보를 입력해주세요.")
    return str(config_file)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LLM 추론 평가 시스템 데이터베이스 초기화")

    parser.add_argument(
        "--config", "-c",
        default="config/db_config.json",
        help="데이터베이스 설정 파일 경로 (기본: config/db_config.json)"
    )

    parser.add_argument(
        "--drop-existing", "-d",
        action="store_true",
        help="기존 테이블을 삭제하고 다시 생성"
    )

    parser.add_argument(
        "--skip-indexes", "-si",
        action="store_true",
        help="인덱스 생성 건너뛰기"
    )

    parser.add_argument(
        "--skip-views", "-sv",
        action="store_true",
        help="뷰 생성 건너뛰기"
    )

    parser.add_argument(
        "--grant-to",
        help="권한을 부여할 사용자명"
    )

    parser.add_argument(
        "--verify-only", "-v",
        action="store_true",
        help="설치 확인만 수행"
    )

    parser.add_argument(
        "--create-config",
        action="store_true",
        help="샘플 설정 파일 생성"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )

    args = parser.parse_args()

    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 로깅 시스템 설정
    setup_application_logging({
        'log_level': 'DEBUG' if args.verbose else 'INFO',
        'log_format': 'simple',
        'enable_console': True
    })

    try:
        # 설정 파일 생성 요청
        if args.create_config:
            create_sample_config()
            return 0

        # 설정 파일 확인
        if not os.path.exists(args.config):
            logger.error(f"설정 파일이 존재하지 않습니다: {args.config}")
            logger.info("다음 명령으로 샘플 설정 파일을 생성할 수 있습니다:")
            logger.info(f"python {__file__} --create-config")
            return 1

        # 데이터베이스 설정 로드
        logger.info(f"설정 파일 로드 중: {args.config}")
        db_config = load_db_config_from_file(args.config)

        # 연결 테스트
        logger.info("데이터베이스 연결 테스트 중...")
        if not db_config.test_connection():
            logger.error("데이터베이스 연결 실패")
            return 1

        logger.info("✅ 데이터베이스 연결 성공")

        # DatabaseManager 및 초기화 클래스 생성
        db_manager = DatabaseManager(db_config)
        initializer = DatabaseInitializer(db_manager)

        # 확인만 수행
        if args.verify_only:
            success = initializer.verify_installation()
            initializer.get_database_info()
            return 0 if success else 1

        # 데이터베이스 초기화 시작
        logger.info("=" * 50)
        logger.info("LLM 추론 평가 시스템 데이터베이스 초기화 시작")
        logger.info("=" * 50)

        # 1. 테이블 생성
        initializer.create_tables(drop_existing=args.drop_existing)

        # 2. 인덱스 생성
        if not args.skip_indexes:
            initializer.create_indexes()

        # 3. 시퀀스 생성
        initializer.create_sequences()

        # 4. 뷰 생성
        if not args.skip_views:
            initializer.create_views()

        # 5. 권한 부여
        if args.grant_to:
            initializer.grant_permissions(args.grant_to)

        # 6. 설치 확인
        success = initializer.verify_installation()

        # 7. 데이터베이스 정보 출력
        initializer.get_database_info()

        if success:
            logger.info("=" * 50)
            logger.info("✅ 데이터베이스 초기화 완료!")
            logger.info("=" * 50)
            logger.info("다음 단계:")
            logger.info("1. 샘플 데이터 로드: python scripts/load_sample_data.py")
            logger.info("2. 시스템 실행: python main.py")
            return 0
        else:
            logger.error("❌ 데이터베이스 초기화 실패")
            return 1

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"예상하지 못한 오류: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # 리소스 정리
        try:
            if 'db_config' in locals():
                db_config.close_pool()
        except:
            pass


if __name__ == "__main__":
    exit(main())