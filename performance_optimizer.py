"""
성능 최적화 유틸리티
"""
import gc
import time
import psutil
from typing import List, Any, Callable, Dict
from functools import wraps
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

logger = logging.getLogger(__name__)


class MemoryManager:
    """메모리 관리 클래스"""

    def __init__(self, memory_threshold_percent: float = 80.0):
        self.memory_threshold_percent = memory_threshold_percent
        self._last_cleanup = time.time()
        self._cleanup_interval = 60  # 60초마다 체크

    def check_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 체크"""
        memory = psutil.virtual_memory()
        process = psutil.Process()

        return {
            'system_memory_percent': memory.percent,
            'system_available_gb': memory.available / (1024 ** 3),
            'process_memory_mb': process.memory_info().rss / (1024 ** 2),
            'threshold_exceeded': memory.percent > self.memory_threshold_percent
        }

    def cleanup_if_needed(self, force: bool = False):
        """필요시 메모리 정리"""
        current_time = time.time()

        if force or (current_time - self._last_cleanup) > self._cleanup_interval:
            memory_info = self.check_memory_usage()

            if memory_info['threshold_exceeded'] or force:
                logger.info(f"메모리 정리 실행 (사용률: {memory_info['system_memory_percent']:.1f}%)")
                gc.collect()
                self._last_cleanup = current_time

                # 정리 후 메모리 상태
                new_memory_info = self.check_memory_usage()
                logger.info(f"메모리 정리 후: {new_memory_info['system_memory_percent']:.1f}%")


class BatchOptimizer:
    """배치 처리 최적화"""

    def __init__(self,
                 default_batch_size: int = 1000,
                 max_batch_size: int = 5000,
                 min_batch_size: int = 100):
        self.default_batch_size = default_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.memory_manager = MemoryManager()
        self._performance_history = []

    def adaptive_batch_size(self, total_items: int, available_memory_gb: float = None) -> int:
        """적응적 배치 크기 결정"""
        if available_memory_gb is None:
            memory_info = self.memory_manager.check_memory_usage()
            available_memory_gb = memory_info['system_available_gb']

        # 메모리 기반 배치 크기 조정
        if available_memory_gb < 1.0:  # 1GB 미만
            batch_size = self.min_batch_size
        elif available_memory_gb < 2.0:  # 1-2GB
            batch_size = self.default_batch_size // 2
        elif available_memory_gb > 8.0:  # 8GB 이상
            batch_size = min(self.max_batch_size, total_items // 10)
        else:
            batch_size = self.default_batch_size

        # 성능 이력 기반 조정
        if self._performance_history:
            avg_performance = sum(self._performance_history[-5:]) / min(5, len(self._performance_history))
            if avg_performance > 10.0:  # 10초 이상 소요시 배치 크기 감소
                batch_size = max(self.min_batch_size, batch_size // 2)

        return max(self.min_batch_size, min(self.max_batch_size, batch_size))

    def process_with_adaptive_batching(self,
                                       items: List[Any],
                                       process_func: Callable,
                                       progress_callback: Callable = None) -> int:
        """적응적 배치 처리"""
        total_items = len(items)
        total_processed = 0

        batch_size = self.adaptive_batch_size(total_items)
        logger.info(f"적응적 배치 처리 시작: 총 {total_items}개, 배치 크기: {batch_size}")

        for i in range(0, total_items, batch_size):
            batch_start_time = time.time()
            batch = items[i:i + batch_size]

            try:
                processed = process_func(batch)
                total_processed += processed

                batch_time = time.time() - batch_start_time
                self._performance_history.append(batch_time)

                # 성능 이력 관리 (최대 20개)
                if len(self._performance_history) > 20:
                    self._performance_history = self._performance_history[-20:]

                # 진행률 콜백
                if progress_callback:
                    progress = min(100, (i + len(batch)) / total_items * 100)
                    progress_callback(progress, batch_time)

                # 메모리 정리
                if (i // batch_size) % 10 == 0:  # 10배치마다
                    self.memory_manager.cleanup_if_needed()

                # 동적 배치 크기 조정
                if batch_time > 15.0 and batch_size > self.min_batch_size:
                    batch_size = max(self.min_batch_size, batch_size // 2)
                    logger.info(f"배치 크기 감소: {batch_size}")
                elif batch_time < 2.0 and batch_size < self.max_batch_size:
                    batch_size = min(self.max_batch_size, int(batch_size * 1.5))
                    logger.info(f"배치 크기 증가: {batch_size}")

            except Exception as e:
                logger.error(f"배치 처리 실패 (배치 {i // batch_size + 1}): {e}")
                continue

        logger.info(f"적응적 배치 처리 완료: {total_processed}/{total_items}개")
        return total_processed


class ParallelProcessor:
    """병렬 처리 클래스"""

    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or min(4, psutil.cpu_count())
        self.memory_manager = MemoryManager()

    def process_parallel_batches(self,
                                 batches: List[List[Any]],
                                 process_func: Callable,
                                 max_concurrent: int = None) -> int:
        """병렬 배치 처리"""
        if max_concurrent is None:
            max_concurrent = min(self.max_workers, len(batches))

        total_processed = 0
        completed_batches = 0

        logger.info(f"병렬 배치 처리 시작: {len(batches)}개 배치, {max_concurrent}개 동시 실행")

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # 작업 제출
            future_to_batch = {
                executor.submit(self._safe_process_batch, batch, process_func): i
                for i, batch in enumerate(batches)
            }

            # 결과 수집
            for future in as_completed(future_to_batch):
                batch_index = future_to_batch[future]
                try:
                    processed = future.result()
                    total_processed += processed
                    completed_batches += 1

                    logger.debug(f"배치 {batch_index + 1}/{len(batches)} 완료: {processed}개 처리")

                    # 주기적 메모리 정리
                    if completed_batches % 5 == 0:
                        self.memory_manager.cleanup_if_needed()

                except Exception as e:
                    logger.error(f"배치 {batch_index + 1} 처리 실패: {e}")

        logger.info(f"병렬 배치 처리 완료: {total_processed}개 처리 ({completed_batches}/{len(batches)} 배치 성공)")
        return total_processed

    def _safe_process_batch(self, batch: List[Any], process_func: Callable) -> int:
        """안전한 배치 처리 래퍼"""
        try:
            return process_func(batch)
        except Exception as e:
            logger.error(f"배치 처리 중 오류: {e}")
            return 0


def performance_monitor(log_performance: bool = True):
    """성능 모니터링 데코레이터"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024 ** 2)  # MB

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024 ** 2)

                execution_time = end_time - start_time
                memory_diff = end_memory - start_memory

                if log_performance:
                    logger.info(f"함수 '{func.__name__}' 실행 완료: "
                                f"시간 {execution_time:.2f}초, 메모리 변화 {memory_diff:+.1f}MB")

                # 성능 경고
                if execution_time > 30:
                    logger.warning(f"느린 함수 감지: {func.__name__} ({execution_time:.2f}초)")

                if memory_diff > 100:
                    logger.warning(f"높은 메모리 사용: {func.__name__} ({memory_diff:.1f}MB)")

                return result

            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                logger.error(f"함수 '{func.__name__}' 실행 실패 ({execution_time:.2f}초): {e}")
                raise

        return wrapper

    return decorator


class OptimizedDataProcessor:
    """최적화된 데이터 처리 클래스"""

    def __init__(self, db_manager, batch_size: int = 1000):
        self.db_manager = db_manager
        self.batch_optimizer = BatchOptimizer(default_batch_size=batch_size)
        self.parallel_processor = ParallelProcessor()
        self.memory_manager = MemoryManager()

    @performance_monitor()
    def process_large_dataset(self, data_points: List[Any], process_func: Callable) -> int:
        """대용량 데이터셋 최적화 처리"""
        total_items = len(data_points)

        if total_items < 1000:
            # 소량 데이터는 단순 처리
            return process_func(data_points)

        # 메모리 상태 확인
        memory_info = self.memory_manager.check_memory_usage()

        if memory_info['threshold_exceeded']:
            logger.warning("메모리 부족으로 보수적 처리 모드로 전환")
            return self._conservative_processing(data_points, process_func)

        # 일반적인 최적화 처리
        return self._optimized_processing(data_points, process_func)

    def _optimized_processing(self, data_points: List[Any], process_func: Callable) -> int:
        """최적화된 처리"""

        def progress_callback(progress, batch_time):
            if progress % 10 == 0:  # 10%마다 로그
                logger.info(f"처리 진행률: {progress:.1f}% (배치 시간: {batch_time:.2f}초)")

        return self.batch_optimizer.process_with_adaptive_batching(
            data_points, process_func, progress_callback
        )

    def _conservative_processing(self, data_points: List[Any], process_func: Callable) -> int:
        """보수적 처리 (메모리 부족시)"""
        small_batches = []
        batch_size = self.batch_optimizer.min_batch_size

        for i in range(0, len(data_points), batch_size):
            small_batches.append(data_points[i:i + batch_size])

        total_processed = 0
        for i, batch in enumerate(small_batches):
            try:
                processed = process_func(batch)
                total_processed += processed

                # 매 배치마다 메모리 정리
                self.memory_manager.cleanup_if_needed(force=True)

                if (i + 1) % 10 == 0:
                    logger.info(f"보수적 처리 진행: {i + 1}/{len(small_batches)} 배치")

            except Exception as e:
                logger.error(f"보수적 처리 배치 {i + 1} 실패: {e}")
                continue

        return total_processed


class QueryOptimizer:
    """쿼리 최적화 클래스"""

    def __init__(self, db_manager):
        self.db_manager = db_manager
        self._query_cache = {}
        self._cache_timeout = 300  # 5분

    def get_optimized_data_query(self,
                                 category: str = None,
                                 difficulty: str = None,
                                 source: str = None,
                                 limit: int = None,
                                 offset: int = None,
                                 use_hints: bool = True) -> tuple:
        """최적화된 데이터 조회 쿼리 생성"""
        from datetime import datetime, timedelta
        from data_models import Constants

        # 캐시 키 생성
        cache_key = f"{category}_{difficulty}_{source}_{limit}_{offset}"

        # 캐시 확인
        if cache_key in self._query_cache:
            cached_time, cached_query = self._query_cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self._cache_timeout):
                return cached_query

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

        # 힌트 추가 (Oracle 최적화)
        hints = ""
        if use_hints:
            if category:
                hints += "/*+ INDEX(rd IDX_REASONING_CATEGORY) */ "
            elif difficulty:
                hints += "/*+ INDEX(rd IDX_REASONING_DIFFICULTY) */ "
            else:
                hints += "/*+ INDEX(rd IDX_REASONING_CREATED_AT) */ "

        sql = f"""
        SELECT {hints}
               ID, CATEGORY, DIFFICULTY, QUESTION, CORRECT_ANSWER,
               EXPLANATION, OPTIONS, SOURCE, 
               TO_CHAR(CREATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as CREATED_AT,
               TO_CHAR(UPDATED_AT, 'YYYY-MM-DD"T"HH24:MI:SS') as UPDATED_AT,
               METADATA
        FROM {Constants.TABLE_REASONING_DATA} rd
        WHERE {where_clause}
        ORDER BY CREATED_AT DESC
        """

        if limit:
            if offset:
                sql += f" OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY"
            else:
                sql += f" FETCH FIRST {limit} ROWS ONLY"

        # 캐시에 저장
        query_result = (sql, params)
        self._query_cache[cache_key] = (datetime.now(), query_result)

        return query_result

    def get_statistics_query_optimized(self) -> str:
        """최적화된 통계 쿼리"""
        from data_models import Constants
        return f"""
        SELECT /*+ PARALLEL(4) */
               COUNT(*) as total_count,
               COUNT(CASE WHEN CATEGORY = 'math' THEN 1 END) as math_count,
               COUNT(CASE WHEN CATEGORY = 'logic' THEN 1 END) as logic_count,
               COUNT(CASE WHEN CATEGORY = 'common_sense' THEN 1 END) as common_sense_count,
               COUNT(CASE WHEN CATEGORY = 'reading_comprehension' THEN 1 END) as reading_count,
               COUNT(CASE WHEN CATEGORY = 'science' THEN 1 END) as science_count,
               COUNT(CASE WHEN DIFFICULTY = 'easy' THEN 1 END) as easy_count,
               COUNT(CASE WHEN DIFFICULTY = 'medium' THEN 1 END) as medium_count,
               COUNT(CASE WHEN DIFFICULTY = 'hard' THEN 1 END) as hard_count
        FROM {Constants.TABLE_REASONING_DATA}
        """

    def clear_query_cache(self):
        """쿼리 캐시 정리"""
        self._query_cache.clear()
        logger.info("쿼리 캐시 정리 완료")


class IndexOptimizer:
    """인덱스 최적화 클래스"""

    def __init__(self, db_manager):
        self.db_manager = db_manager

    def create_composite_indexes(self):
        """복합 인덱스 생성"""
        from data_models import Constants

        composite_indexes = [
            f"CREATE INDEX IDX_REASONING_CAT_DIFF ON {Constants.TABLE_REASONING_DATA}(CATEGORY, DIFFICULTY)",
            f"CREATE INDEX IDX_REASONING_CAT_SOURCE ON {Constants.TABLE_REASONING_DATA}(CATEGORY, SOURCE)",
            f"CREATE INDEX IDX_REASONING_DIFF_DATE ON {Constants.TABLE_REASONING_DATA}(DIFFICULTY, CREATED_AT)",
            f"CREATE INDEX IDX_EVAL_MODEL_CORRECT ON {Constants.TABLE_EVALUATION_RESULTS}(MODEL_NAME, IS_CORRECT)",
            f"CREATE INDEX IDX_EVAL_MODEL_DATE ON {Constants.TABLE_EVALUATION_RESULTS}(MODEL_NAME, CREATED_AT)",
        ]

        for index_sql in composite_indexes:
            try:
                self.db_manager.execute_dml(index_sql)
                index_name = index_sql.split()[2]
                logger.info(f"복합 인덱스 생성 완료: {index_name}")
            except Exception as e:
                if "name is already used" in str(e).lower() or "already exists" in str(e).lower():
                    continue
                logger.warning(f"복합 인덱스 생성 실패: {e}")

    def analyze_table_statistics(self):
        """테이블 통계 정보 갱신"""
        from data_models import Constants

        tables_to_analyze = [
            Constants.TABLE_REASONING_DATA,
            Constants.TABLE_EVALUATION_RESULTS,
            Constants.TABLE_DATASET_STATS
        ]

        for table in tables_to_analyze:
            try:
                analyze_sql = f"ANALYZE TABLE {table} COMPUTE STATISTICS"
                self.db_manager.execute_dml(analyze_sql)
                logger.info(f"테이블 통계 갱신 완료: {table}")
            except Exception as e:
                logger.warning(f"테이블 통계 갱신 실패 ({table}): {e}")

    def get_index_usage_stats(self) -> List[Dict[str, Any]]:
        """인덱스 사용 통계 조회"""
        from data_models import Constants

        try:
            stats_sql = """
            SELECT INDEX_NAME, TABLE_NAME, NUM_ROWS, DISTINCT_KEYS, 
                   CLUSTERING_FACTOR, LAST_ANALYZED
            FROM USER_INDEXES 
            WHERE TABLE_NAME IN (:1, :2, :3)
            ORDER BY TABLE_NAME, INDEX_NAME
            """

            params = [
                Constants.TABLE_REASONING_DATA,
                Constants.TABLE_EVALUATION_RESULTS,
                Constants.TABLE_DATASET_STATS
            ]

            rows = self.db_manager.execute_query(stats_sql, params)

            return [
                {
                    'index_name': row[0],
                    'table_name': row[1],
                    'num_rows': row[2],
                    'distinct_keys': row[3],
                    'clustering_factor': row[4],
                    'last_analyzed': row[5]
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"인덱스 통계 조회 실패: {e}")
            return []


# 성능 최적화 사용 예시
class PerformanceOptimizedCollector:
    """성능 최적화된 데이터 컬렉터"""

    def __init__(self, db_manager, batch_size: int = 1000):
        self.db_manager = db_manager
        self.optimizer = OptimizedDataProcessor(db_manager, batch_size)
        self.query_optimizer = QueryOptimizer(db_manager)
        self.index_optimizer = IndexOptimizer(db_manager)

    @performance_monitor()
    def add_large_dataset(self, data_points: List[Any]) -> int:
        """대용량 데이터셋 추가 (최적화됨)"""

        def batch_insert_func(batch):
            # 실제 배치 삽입 로직
            return len(batch)  # 임시 반환값

        return self.optimizer.process_large_dataset(data_points, batch_insert_func)

    @performance_monitor()
    def get_optimized_data(self, **filters) -> List[Any]:
        """최적화된 데이터 조회"""
        sql, params = self.query_optimizer.get_optimized_data_query(**filters)
        return self.db_manager.execute_query(sql, params)

    def setup_performance_optimizations(self):
        """성능 최적화 설정"""
        logger.info("성능 최적화 설정 시작...")

        # 복합 인덱스 생성
        self.index_optimizer.create_composite_indexes()

        # 테이블 통계 갱신
        self.index_optimizer.analyze_table_statistics()

        logger.info("성능 최적화 설정 완료")

    def get_performance_report(self) -> Dict[str, Any]:
        """성능 리포트 생성"""
        memory_info = MemoryManager().check_memory_usage()
        index_stats = self.index_optimizer.get_index_usage_stats()

        return {
            'memory_status': memory_info,
            'index_statistics': index_stats,
            'query_cache_size': len(self.query_optimizer._query_cache)
        }


# 전역 성능 유틸리티 함수들
def optimize_system_performance():
    """시스템 성능 최적화"""
    # 가비지 컬렉션 강제 실행
    gc.collect()

    # 시스템 정보 로깅
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent(interval=1)

    logger.info(f"시스템 성능 현황:")
    logger.info(f"  메모리 사용률: {memory.percent:.1f}%")
    logger.info(f"  사용 가능 메모리: {memory.available / (1024 ** 3):.2f}GB")
    logger.info(f"  CPU 사용률: {cpu_percent:.1f}%")


def monitor_system_resources(interval: int = 60):
    """시스템 리소스 모니터링 (백그라운드)"""
    import threading

    def resource_monitor():
        while True:
            try:
                memory = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent(interval=1)

                if memory.percent > 85:
                    logger.warning(f"높은 메모리 사용률 감지: {memory.percent:.1f}%")

                if cpu_percent > 85:
                    logger.warning(f"높은 CPU 사용률 감지: {cpu_percent:.1f}%")

                time.sleep(interval)

            except Exception as e:
                logger.error(f"리소스 모니터링 오류: {e}")
                time.sleep(interval)

    monitor_thread = threading.Thread(target=resource_monitor, daemon=True)
    monitor_thread.start()
    logger.info(f"시스템 리소스 모니터링 시작 (간격: {interval}초)")


# 메모리 효율적인 데이터 처리 유틸리티
class StreamingDataProcessor:
    """스트리밍 데이터 처리기"""

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.memory_manager = MemoryManager()

    def process_large_file(self, file_path: str, process_func: Callable) -> int:
        """대용량 파일 스트리밍 처리"""
        total_processed = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                chunk = []

                for line_num, line in enumerate(file, 1):
                    chunk.append(line.strip())

                    # 청크 크기에 도달하면 처리
                    if len(chunk) >= self.chunk_size:
                        processed = process_func(chunk)
                        total_processed += processed
                        chunk = []

                        # 주기적 메모리 정리
                        if line_num % (self.chunk_size * 10) == 0:
                            self.memory_manager.cleanup_if_needed()

                # 남은 데이터 처리
                if chunk:
                    processed = process_func(chunk)
                    total_processed += processed

        except Exception as e:
            logger.error(f"파일 스트리밍 처리 오류: {e}")

        return total_processed