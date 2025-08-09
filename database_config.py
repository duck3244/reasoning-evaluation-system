"""
Oracle 데이터베이스 설정 및 연결 관리
"""
import oracledb
import json
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import logging
from data_models import Constants

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """데이터베이스 설정 클래스"""

    def __init__(self,
                 username: str,
                 password: str,
                 dsn: str,
                 pool_min: int = 1,
                 pool_max: int = 10,
                 pool_increment: int = 1):
        """
        Oracle DB 설정 초기화

        Args:
            username: DB 사용자명
            password: DB 비밀번호
            dsn: DB 연결 문자열 (hostname:port/service_name)
            pool_min: 최소 연결 풀 크기
            pool_max: 최대 연결 풀 크기
            pool_increment: 연결 풀 증가 단위
        """
        self.username = username
        self.password = password
        self.dsn = dsn
        self.pool_min = pool_min
        self.pool_max = pool_max
        self.pool_increment = pool_increment
        self.pool = None

    def create_pool(self):
        """연결 풀 생성"""
        try:
            self.pool = oracledb.create_pool(
                user=self.username,
                password=self.password,
                dsn=self.dsn,
                min=self.pool_min,
                max=self.pool_max,
                increment=self.pool_increment
            )
            logger.info("Oracle DB 연결 풀이 생성되었습니다.")
        except Exception as e:
            logger.error(f"DB 연결 풀 생성 실패: {e}")
            raise

    def close_pool(self):
        """연결 풀 종료"""
        if self.pool:
            self.pool.close()
            self.pool = None
            logger.info("Oracle DB 연결 풀이 종료되었습니다.")

    @contextmanager
    def get_connection(self):
        """연결 컨텍스트 매니저"""
        if not self.pool:
            self.create_pool()

        connection = None
        try:
            connection = self.pool.acquire()
            yield connection
        except Exception as e:
            logger.error(f"DB 연결 오류: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection:
                self.pool.release(connection)


class DatabaseManager:
    """데이터베이스 관리 클래스"""

    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config

    def init_database(self):
        """데이터베이스 초기화 (테이블 생성)"""
        with self.db_config.get_connection() as conn:
            cursor = conn.cursor()

            try:
                # 추론 데이터 테이블 생성
                self._create_reasoning_data_table(cursor)

                # 평가 결과 테이블 생성
                self._create_evaluation_results_table(cursor)

                # 통계 테이블 생성
                self._create_statistics_table(cursor)

                # 인덱스 생성
                self._create_indexes(cursor)

                # 시퀀스 생성
                self._create_sequences(cursor)

                conn.commit()
                logger.info("데이터베이스 초기화가 완료되었습니다.")

            except Exception as e:
                conn.rollback()
                logger.error(f"데이터베이스 초기화 실패: {e}")
                raise

    def _create_reasoning_data_table(self, cursor):
        """추론 데이터 테이블 생성"""
        create_table_sql = f"""
        CREATE TABLE {Constants.TABLE_REASONING_DATA} (
            ID VARCHAR2(32) PRIMARY KEY,
            CATEGORY VARCHAR2(50) NOT NULL,
            DIFFICULTY VARCHAR2(20) NOT NULL,
            QUESTION CLOB NOT NULL,
            CORRECT_ANSWER CLOB NOT NULL,
            EXPLANATION CLOB,
            OPTIONS CLOB,  -- JSON 형태로 저장
            SOURCE VARCHAR2(100),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            METADATA CLOB,  -- JSON 형태로 저장
            CONSTRAINT CHK_CATEGORY CHECK (CATEGORY IN ('math', 'logic', 'common_sense', 'reading_comprehension', 'science', 'history', 'language')),
            CONSTRAINT CHK_DIFFICULTY CHECK (DIFFICULTY IN ('easy', 'medium', 'hard'))
        )
        """

        try:
            cursor.execute(create_table_sql)
            logger.info(f"{Constants.TABLE_REASONING_DATA} 테이블이 생성되었습니다.")
        except oracledb.DatabaseError as e:
            error_code = e.args[0].code
            if error_code == 955:  # 테이블이 이미 존재
                logger.info(f"{Constants.TABLE_REASONING_DATA} 테이블이 이미 존재합니다.")
            else:
                raise

    def _create_evaluation_results_table(self, cursor):
        """평가 결과 테이블 생성"""
        create_table_sql = f"""
        CREATE TABLE {Constants.TABLE_EVALUATION_RESULTS} (
            ID VARCHAR2(32) PRIMARY KEY,
            DATA_POINT_ID VARCHAR2(32) NOT NULL,
            MODEL_NAME VARCHAR2(100) NOT NULL,
            PREDICTED_ANSWER CLOB NOT NULL,
            IS_CORRECT NUMBER(1) CHECK (IS_CORRECT IN (0,1)),
            CONFIDENCE_SCORE NUMBER(5,4),
            REASONING_STEPS CLOB,  -- JSON 형태로 저장
            EXECUTION_TIME NUMBER(10,3),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            METADATA CLOB,  -- JSON 형태로 저장
            CONSTRAINT FK_EVAL_DATA_POINT FOREIGN KEY (DATA_POINT_ID) 
                REFERENCES {Constants.TABLE_REASONING_DATA}(ID)
        )
        """

        try:
            cursor.execute(create_table_sql)
            logger.info(f"{Constants.TABLE_EVALUATION_RESULTS} 테이블이 생성되었습니다.")
        except oracledb.DatabaseError as e:
            error_code = e.args[0].code
            if error_code == 955:  # 테이블이 이미 존재
                logger.info(f"{Constants.TABLE_EVALUATION_RESULTS} 테이블이 이미 존재합니다.")
            else:
                raise

    def _create_statistics_table(self, cursor):
        """통계 테이블 생성"""
        create_table_sql = f"""
        CREATE TABLE {Constants.TABLE_DATASET_STATS} (
            ID NUMBER PRIMARY KEY,
            TOTAL_COUNT NUMBER NOT NULL,
            CATEGORY_COUNTS CLOB,  -- JSON 형태
            DIFFICULTY_COUNTS CLOB,  -- JSON 형태
            SOURCE_COUNTS CLOB,  -- JSON 형태
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        try:
            cursor.execute(create_table_sql)
            logger.info(f"{Constants.TABLE_DATASET_STATS} 테이블이 생성되었습니다.")
        except oracledb.DatabaseError as e:
            error_code = e.args[0].code
            if error_code == 955:  # 테이블이 이미 존재
                logger.info(f"{Constants.TABLE_DATASET_STATS} 테이블이 이미 존재합니다.")
            else:
                raise

    def _create_indexes(self, cursor):
        """인덱스 생성"""
        indexes = [
            f"CREATE INDEX IDX_REASONING_CATEGORY ON {Constants.TABLE_REASONING_DATA}(CATEGORY)",
            f"CREATE INDEX IDX_REASONING_DIFFICULTY ON {Constants.TABLE_REASONING_DATA}(DIFFICULTY)",
            f"CREATE INDEX IDX_REASONING_SOURCE ON {Constants.TABLE_REASONING_DATA}(SOURCE)",
            f"CREATE INDEX IDX_REASONING_CREATED_AT ON {Constants.TABLE_REASONING_DATA}(CREATED_AT)",
            f"CREATE INDEX IDX_EVAL_MODEL ON {Constants.TABLE_EVALUATION_RESULTS}(MODEL_NAME)",
            f"CREATE INDEX IDX_EVAL_CORRECT ON {Constants.TABLE_EVALUATION_RESULTS}(IS_CORRECT)",
            f"CREATE INDEX IDX_EVAL_CREATED_AT ON {Constants.TABLE_EVALUATION_RESULTS}(CREATED_AT)"
        ]

        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.info(f"인덱스 생성: {index_sql.split()[-1]}")
            except oracledb.DatabaseError as e:
                error_code = e.args[0].code
                if error_code == 955:  # 인덱스가 이미 존재
                    continue
                else:
                    logger.warning(f"인덱스 생성 실패: {e}")

    def _create_sequences(self, cursor):
        """시퀀스 생성"""
        sequences = [
            "CREATE SEQUENCE SEQ_DATASET_STATS START WITH 1 INCREMENT BY 1"
        ]

        for seq_sql in sequences:
            try:
                cursor.execute(seq_sql)
                logger.info(f"시퀀스 생성: {seq_sql.split()[2]}")
            except oracledb.DatabaseError as e:
                error_code = e.args[0].code
                if error_code == 955:  # 시퀀스가 이미 존재
                    continue
                else:
                    logger.warning(f"시퀀스 생성 실패: {e}")

    def execute_query(self, sql: str, params: Optional[List] = None) -> List[tuple]:
        """SELECT 쿼리 실행"""
        with self.db_config.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(sql, params)
            else:
                cursor.execute(sql)
            return cursor.fetchall()

    def execute_dml(self, sql: str, params: Optional[List] = None) -> int:
        """INSERT/UPDATE/DELETE 쿼리 실행"""
        with self.db_config.get_connection() as conn:
            cursor = conn.cursor()
            try:
                if params:
                    cursor.execute(sql, params)
                else:
                    cursor.execute(sql)

                rowcount = cursor.rowcount
                conn.commit()
                return rowcount
            except Exception as e:
                conn.rollback()
                raise

    def execute_batch_dml(self, sql: str, params_list: List[List]) -> int:
        """배치 DML 실행"""
        with self.db_config.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.executemany(sql, params_list)
                rowcount = cursor.rowcount
                conn.commit()
                return rowcount
            except Exception as e:
                conn.rollback()
                raise


# 설정 파일에서 DB 정보 로드하는 유틸리티
def load_db_config_from_file(config_file: str = "db_config.json") -> DatabaseConfig:
    """설정 파일에서 DB 설정 로드"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return DatabaseConfig(
            username=config['username'],
            password=config['password'],
            dsn=config['dsn'],
            pool_min=config.get('pool_min', 1),
            pool_max=config.get('pool_max', 10),
            pool_increment=config.get('pool_increment', 1)
        )
    except FileNotFoundError:
        logger.warning(f"설정 파일 {config_file}을 찾을 수 없습니다. 환경변수를 확인하세요.")
        raise
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        raise