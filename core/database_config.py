"""
Oracle 데이터베이스 설정 및 연결 관리 (개선된 예외 처리)
"""
import oracledb
import json
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
import logging
import time
from functools import wraps
from data_models import Constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def retry_on_db_error(max_retries: int = 3, delay: float = 1.0):
    """데이터베이스 오류 시 재시도 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (oracledb.DatabaseError, oracledb.InterfaceError) as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"DB 작업 재시도 {attempt + 1}/{max_retries}: {e}")
                        time.sleep(delay * (2 ** attempt))  # 지수 백오프
                    else:
                        logger.error(f"DB 작업 최종 실패: {e}")
                except Exception as e:
                    # 다른 예외는 재시도하지 않음
                    logger.error(f"예상하지 못한 오류: {e}")
                    raise

            raise last_exception
        return wrapper
    return decorator


class DatabaseConfig:
    """데이터베이스 설정 클래스 (개선된 예외 처리)"""

    def __init__(self,
                 username: str,
                 password: str,
                 dsn: str,
                 pool_min: int = 1,
                 pool_max: int = 10,
                 pool_increment: int = 1,
                 pool_timeout: int = 30):
        self.username = username
        self.password = password
        self.dsn = dsn
        self.pool_min = pool_min
        self.pool_max = pool_max
        self.pool_increment = pool_increment
        self.pool_timeout = pool_timeout
        self.pool = None
        self._pool_creation_attempts = 0
        self._max_pool_creation_attempts = 3

    def create_pool(self):
        """연결 풀 생성 (개선된 예외 처리)"""
        while self._pool_creation_attempts < self._max_pool_creation_attempts:
            try:
                self.pool = oracledb.create_pool(
                    user=self.username,
                    password=self.password,
                    dsn=self.dsn,
                    min=self.pool_min,
                    max=self.pool_max,
                    increment=self.pool_increment,
                    timeout=self.pool_timeout,
                    getmode=oracledb.POOL_GETMODE_WAIT
                )
                logger.info("Oracle DB 연결 풀이 성공적으로 생성되었습니다.")
                self._pool_creation_attempts = 0  # 성공 시 리셋
                return

            except oracledb.DatabaseError as e:
                self._pool_creation_attempts += 1
                error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else 'Unknown'

                logger.error(f"DB 연결 풀 생성 실패 (시도 {self._pool_creation_attempts}/{self._max_pool_creation_attempts}): "
                           f"에러 코드 {error_code}, 메시지: {e}")

                if self._pool_creation_attempts >= self._max_pool_creation_attempts:
                    raise Exception(f"데이터베이스 연결 풀 생성에 {self._max_pool_creation_attempts}번 실패했습니다. "
                                  f"데이터베이스 설정을 확인해주세요.") from e

                # 재시도 전 대기
                time.sleep(2 ** self._pool_creation_attempts)

            except Exception as e:
                self._pool_creation_attempts += 1
                logger.error(f"예상하지 못한 연결 풀 생성 오류: {e}")

                if self._pool_creation_attempts >= self._max_pool_creation_attempts:
                    raise Exception("데이터베이스 연결 풀 생성 중 예상하지 못한 오류가 발생했습니다.") from e

                time.sleep(2)

    def close_pool(self):
        """연결 풀 종료 (안전한 처리)"""
        if self.pool:
            try:
                self.pool.close()
                logger.info("Oracle DB 연결 풀이 안전하게 종료되었습니다.")
            except Exception as e:
                logger.error(f"연결 풀 종료 중 오류: {e}")
            finally:
                self.pool = None

    @contextmanager
    def get_connection(self):
        """안전한 연결 컨텍스트 매니저"""
        if not self.pool:
            try:
                self.create_pool()
            except Exception as e:
                logger.error(f"연결 풀 생성 실패: {e}")
                raise

        connection = None
        try:
            connection = self.pool.acquire()
            connection.autocommit = False  # 명시적 트랜잭션 관리
            yield connection

        except oracledb.DatabaseError as e:
            error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else 'Unknown'
            logger.error(f"DB 연결 오류 (코드: {error_code}): {e}")

            if connection:
                try:
                    connection.rollback()
                    logger.info("트랜잭션 롤백 완료")
                except Exception as rollback_error:
                    logger.error(f"롤백 실패: {rollback_error}")
            raise

        except Exception as e:
            logger.error(f"예상하지 못한 DB 오류: {e}")

            if connection:
                try:
                    connection.rollback()
                except Exception:
                    pass  # 롤백 실패 시에도 연결 반환
            raise

        finally:
            if connection:
                try:
                    self.pool.release(connection)
                except Exception as e:
                    logger.error(f"연결 반환 실패: {e}")

    def test_connection(self) -> bool:
        """연결 테스트"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM DUAL")
                result = cursor.fetchone()
                cursor.close()
                return result is not None
        except Exception as e:
            logger.error(f"연결 테스트 실패: {e}")
            return False


class DatabaseManager:
    """데이터베이스 관리 클래스 (개선된 예외 처리)"""

    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config

    @retry_on_db_error(max_retries=3)
    def execute_query(self, sql: str, params: Optional[List] = None) -> List[tuple]:
        """SELECT 쿼리 실행 (개선된 예외 처리)"""
        if not sql.strip():
            raise ValueError("빈 SQL 쿼리는 실행할 수 없습니다.")

        try:
            with self.db_config.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    if params:
                        cursor.execute(sql, params)
                    else:
                        cursor.execute(sql)

                    result = cursor.fetchall()
                    logger.debug(f"쿼리 실행 완료: {len(result)}개 행 반환")
                    return result

                finally:
                    cursor.close()

        except oracledb.DatabaseError as e:
            error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else 'Unknown'
            logger.error(f"쿼리 실행 실패 (코드: {error_code}): {sql[:100]}...")
            raise
        except Exception as e:
            logger.error(f"예상하지 못한 쿼리 실행 오류: {e}")
            raise

    @retry_on_db_error(max_retries=3)
    def execute_dml(self, sql: str, params: Optional[List] = None) -> int:
        """INSERT/UPDATE/DELETE 쿼리 실행 (개선된 예외 처리)"""
        if not sql.strip():
            raise ValueError("빈 SQL 쿼리는 실행할 수 없습니다.")

        try:
            with self.db_config.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    if params:
                        cursor.execute(sql, params)
                    else:
                        cursor.execute(sql)

                    rowcount = cursor.rowcount
                    conn.commit()
                    logger.debug(f"DML 실행 완료: {rowcount}개 행 영향")
                    return rowcount

                except Exception as e:
                    conn.rollback()
                    raise
                finally:
                    cursor.close()

        except oracledb.DatabaseError as e:
            error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else 'Unknown'
            logger.error(f"DML 실행 실패 (코드: {error_code}): {sql[:100]}...")
            raise
        except Exception as e:
            logger.error(f"예상하지 못한 DML 실행 오류: {e}")
            raise

    @retry_on_db_error(max_retries=3)
    def execute_batch_dml(self, sql: str, params_list: List[List]) -> int:
        """배치 DML 실행 (개선된 예외 처리)"""
        if not sql.strip():
            raise ValueError("빈 SQL 쿼리는 실행할 수 없습니다.")

        if not params_list:
            logger.warning("배치 DML: 파라미터 리스트가 비어있습니다.")
            return 0

        try:
            with self.db_config.get_connection() as conn:
                cursor = conn.cursor()
                try:
                    cursor.executemany(sql, params_list)
                    rowcount = cursor.rowcount
                    conn.commit()
                    logger.info(f"배치 DML 실행 완료: {rowcount}개 행 영향 (배치 크기: {len(params_list)})")
                    return rowcount

                except Exception as e:
                    conn.rollback()
                    raise
                finally:
                    cursor.close()

        except oracledb.DatabaseError as e:
            error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else 'Unknown'
            logger.error(f"배치 DML 실행 실패 (코드: {error_code}): 배치 크기 {len(params_list)}")
            raise
        except Exception as e:
            logger.error(f"예상하지 못한 배치 DML 실행 오류: {e}")
            raise

    def init_database(self):
        """데이터베이스 초기화 (개선된 예외 처리)"""
        try:
            with self.db_config.get_connection() as conn:
                cursor = conn.cursor()

                initialization_steps = [
                    ("추론 데이터 테이블", self._create_reasoning_data_table),
                    ("평가 결과 테이블", self._create_evaluation_results_table),
                    ("통계 테이블", self._create_statistics_table),
                    ("인덱스", self._create_indexes),
                    ("시퀀스", self._create_sequences),
                ]

                for step_name, step_function in initialization_steps:
                    try:
                        step_function(cursor)
                        logger.info(f"{step_name} 생성 완료")
                    except Exception as e:
                        logger.warning(f"{step_name} 생성 중 오류 (계속 진행): {e}")

                conn.commit()
                logger.info("데이터베이스 초기화가 완료되었습니다.")

        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            raise Exception("데이터베이스 초기화에 실패했습니다. 데이터베이스 연결과 권한을 확인해주세요.") from e

    def _create_reasoning_data_table(self, cursor):
        """추론 데이터 테이블 생성 (안전한 처리)"""
        create_table_sql = f"""
        CREATE TABLE {Constants.TABLE_REASONING_DATA} (
            ID VARCHAR2(32) PRIMARY KEY,
            CATEGORY VARCHAR2(50) NOT NULL,
            DIFFICULTY VARCHAR2(20) NOT NULL,
            QUESTION CLOB NOT NULL,
            CORRECT_ANSWER CLOB NOT NULL,
            EXPLANATION CLOB,
            OPTIONS CLOB,
            SOURCE VARCHAR2(100),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UPDATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            METADATA CLOB,
            CONSTRAINT CHK_CATEGORY CHECK (CATEGORY IN ('math', 'logic', 'common_sense', 'reading_comprehension', 'science', 'history', 'language', 'unknown')),
            CONSTRAINT CHK_DIFFICULTY CHECK (DIFFICULTY IN ('easy', 'medium', 'hard'))
        )
        """

        try:
            cursor.execute(create_table_sql)
        except oracledb.DatabaseError as e:
            error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else None
            if error_code == 955:  # 테이블이 이미 존재
                logger.info(f"{Constants.TABLE_REASONING_DATA} 테이블이 이미 존재합니다.")
            else:
                raise

    def _create_evaluation_results_table(self, cursor):
        """평가 결과 테이블 생성 (안전한 처리)"""
        create_table_sql = f"""
        CREATE TABLE {Constants.TABLE_EVALUATION_RESULTS} (
            ID VARCHAR2(32) PRIMARY KEY,
            DATA_POINT_ID VARCHAR2(32) NOT NULL,
            MODEL_NAME VARCHAR2(100) NOT NULL,
            PREDICTED_ANSWER CLOB NOT NULL,
            IS_CORRECT NUMBER(1) CHECK (IS_CORRECT IN (0,1)),
            CONFIDENCE_SCORE NUMBER(5,4),
            REASONING_STEPS CLOB,
            EXECUTION_TIME NUMBER(10,3),
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            METADATA CLOB,
            CONSTRAINT FK_EVAL_DATA_POINT FOREIGN KEY (DATA_POINT_ID) 
                REFERENCES {Constants.TABLE_REASONING_DATA}(ID) ON DELETE CASCADE
        )
        """

        try:
            cursor.execute(create_table_sql)
        except oracledb.DatabaseError as e:
            error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else None
            if error_code == 955:  # 테이블이 이미 존재
                logger.info(f"{Constants.TABLE_EVALUATION_RESULTS} 테이블이 이미 존재합니다.")
            else:
                raise

    def _create_statistics_table(self, cursor):
        """통계 테이블 생성 (안전한 처리)"""
        create_table_sql = f"""
        CREATE TABLE {Constants.TABLE_DATASET_STATS} (
            ID NUMBER PRIMARY KEY,
            TOTAL_COUNT NUMBER NOT NULL,
            CATEGORY_COUNTS CLOB,
            DIFFICULTY_COUNTS CLOB,
            SOURCE_COUNTS CLOB,
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """

        try:
            cursor.execute(create_table_sql)
        except oracledb.DatabaseError as e:
            error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else None
            if error_code == 955:  # 테이블이 이미 존재
                logger.info(f"{Constants.TABLE_DATASET_STATS} 테이블이 이미 존재합니다.")
            else:
                raise

    def _create_indexes(self, cursor):
        """인덱스 생성 (안전한 처리)"""
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
                index_name = index_sql.split()[-1].split('(')[0]
                logger.debug(f"인덱스 생성: {index_name}")
            except oracledb.DatabaseError as e:
                error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else None
                if error_code == 955:  # 인덱스가 이미 존재
                    continue
                else:
                    logger.warning(f"인덱스 생성 실패: {e}")

    def _create_sequences(self, cursor):
        """시퀀스 생성 (안전한 처리)"""
        sequences = [
            "CREATE SEQUENCE SEQ_DATASET_STATS START WITH 1 INCREMENT BY 1"
        ]

        for seq_sql in sequences:
            try:
                cursor.execute(seq_sql)
                seq_name = seq_sql.split()[2]
                logger.debug(f"시퀀스 생성: {seq_name}")
            except oracledb.DatabaseError as e:
                error_code = e.args[0].code if e.args and hasattr(e.args[0], 'code') else None
                if error_code == 955:  # 시퀀스가 이미 존재
                    continue
                else:
                    logger.warning(f"시퀀스 생성 실패: {e}")


def load_db_config_from_file(config_file: str = "db_config.json") -> DatabaseConfig:
    """설정 파일에서 DB 설정 로드 (개선된 예외 처리)"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # 필수 필드 검증
        required_fields = ['username', 'password', 'dsn']
        missing_fields = [field for field in required_fields if field not in config]

        if missing_fields:
            raise ValueError(f"설정 파일에 필수 필드가 누락되었습니다: {missing_fields}")

        return DatabaseConfig(
            username=config['username'],
            password=config['password'],
            dsn=config['dsn'],
            pool_min=config.get('pool_min', 1),
            pool_max=config.get('pool_max', 10),
            pool_increment=config.get('pool_increment', 1)
        )

    except FileNotFoundError:
        logger.error(f"설정 파일 {config_file}을 찾을 수 없습니다.")
        raise FileNotFoundError(f"데이터베이스 설정 파일 '{config_file}'이 존재하지 않습니다. "
                               f"'python main.py config' 명령으로 샘플 설정 파일을 생성하세요.")
    except json.JSONDecodeError as e:
        logger.error(f"설정 파일 JSON 파싱 오류: {e}")
        raise ValueError(f"설정 파일 '{config_file}'의 JSON 형식이 올바르지 않습니다.") from e
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        raise Exception(f"설정 파일 로드 중 예상하지 못한 오류가 발생했습니다.") from e