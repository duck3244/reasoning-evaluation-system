"""
구조화된 로깅 시스템
"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback
from functools import wraps


class StructuredFormatter(logging.Formatter):
    """구조화된 JSON 로그 포매터"""

    def format(self, record):
        # 기본 로그 데이터
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'process_id': record.process
        }

        # 예외 정보 추가
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # 커스텀 컨텍스트 정보 추가
        for attr in ['user_id', 'request_id', 'operation_id', 'model_name', 'data_id']:
            if hasattr(record, attr):
                log_data[attr] = getattr(record, attr)

        # 성능 정보 추가
        if hasattr(record, 'execution_time'):
            log_data['performance'] = {
                'execution_time': record.execution_time,
                'memory_usage': getattr(record, 'memory_usage', None)
            }

        return json.dumps(log_data, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """간단한 텍스트 포매터"""

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class LoggingManager:
    """로깅 관리자"""

    def __init__(self,
                 log_level: str = "INFO",
                 log_format: str = "structured",  # "structured" or "simple"
                 log_file: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True):

        self.log_level = log_level.upper()
        self.log_format = log_format
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console

        self._setup_logging()

    def _setup_logging(self):
        """로깅 시스템 설정"""
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))

        # 기존 핸들러 제거
        root_logger.handlers.clear()

        # 포매터 선택
        if self.log_format == "structured":
            formatter = StructuredFormatter()
            console_formatter = SimpleFormatter()  # 콘솔은 읽기 쉽게
        else:
            formatter = SimpleFormatter()
            console_formatter = formatter

        # 콘솔 핸들러
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)  # 콘솔은 INFO 이상만
            root_logger.addHandler(console_handler)

        # 파일 핸들러
        if self.log_file:
            log_dir = Path(self.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # 에러 전용 파일 핸들러
        if self.log_file:
            error_log_file = str(Path(self.log_file).with_suffix('.error.log'))
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)

        # 성능 로그 핸들러
        if self.log_file:
            perf_log_file = str(Path(self.log_file).with_suffix('.performance.log'))
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            perf_handler.addFilter(lambda record: hasattr(record, 'execution_time'))
            perf_handler.setFormatter(formatter)

            # 성능 로거 설정
            perf_logger = logging.getLogger('performance')
            perf_logger.addHandler(perf_handler)
            perf_logger.propagate = False


class LogContext:
    """로그 컨텍스트 관리자"""

    def __init__(self, **context):
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def with_logging_context(**context):
    """로깅 컨텍스트 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with LogContext(**context):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_performance(logger_name: str = 'performance'):
    """성능 로깅 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            import psutil

            logger = logging.getLogger(logger_name)

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**2)

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024**2)

                execution_time = end_time - start_time
                memory_diff = end_memory - start_memory

                # 성능 로그 기록
                log_record = logger.makeRecord(
                    logger.name, logging.INFO, func.__code__.co_filename,
                    func.__code__.co_firstlineno,
                    f"Function '{func.__name__}' completed successfully",
                    (), None, func.__name__
                )

                # 성능 정보 추가
                log_record.execution_time = execution_time
                log_record.memory_usage = memory_diff
                log_record.function_name = func.__name__

                logger.handle(log_record)

                return result

            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time

                logger.error(
                    f"Function '{func.__name__}' failed after {execution_time:.2f}s",
                    exc_info=True,
                    extra={'execution_time': execution_time, 'function_name': func.__name__}
                )
                raise

        return wrapper
    return decorator


class LogAnalyzer:
    """로그 분석기"""

    def __init__(self, log_file: str):
        self.log_file = log_file

    def analyze_performance_logs(self, hours: int = 24) -> Dict[str, Any]:
        """성능 로그 분석"""
        try:
            from datetime import datetime, timedelta

            cutoff_time = datetime.now() - timedelta(hours=hours)
            performance_data = []

            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # 성능 로그만 필터링
                        if 'performance' in log_entry:
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                            if log_time >= cutoff_time:
                                performance_data.append(log_entry)

                    except (json.JSONDecodeError, ValueError):
                        continue

            if not performance_data:
                return {}

            # 성능 분석
            execution_times = [p['performance']['execution_time'] for p in performance_data]
            memory_usages = [p['performance'].get('memory_usage', 0) for p in performance_data]

            function_stats = {}
            for entry in performance_data:
                func_name = entry.get('function', 'unknown')
                if func_name not in function_stats:
                    function_stats[func_name] = {
                        'count': 0,
                        'total_time': 0,
                        'max_time': 0,
                        'min_time': float('inf')
                    }

                exec_time = entry['performance']['execution_time']
                function_stats[func_name]['count'] += 1
                function_stats[func_name]['total_time'] += exec_time
                function_stats[func_name]['max_time'] = max(function_stats[func_name]['max_time'], exec_time)
                function_stats[func_name]['min_time'] = min(function_stats[func_name]['min_time'], exec_time)

            # 평균 계산
            for stats in function_stats.values():
                stats['avg_time'] = stats['total_time'] / stats['count']
                if stats['min_time'] == float('inf'):
                    stats['min_time'] = 0

            return {
                'total_operations': len(performance_data),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'max_execution_time': max(execution_times),
                'min_execution_time': min(execution_times),
                'avg_memory_usage': sum(memory_usages) / len(memory_usages),
                'function_statistics': function_stats,
                'slow_operations': [
                    {
                        'function': p.get('function', 'unknown'),
                        'execution_time': p['performance']['execution_time'],
                        'timestamp': p['timestamp']
                    }
                    for p in performance_data
                    if p['performance']['execution_time'] > 10  # 10초 이상
                ]
            }

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"성능 로그 분석 오류: {e}")
            return {}

    def analyze_error_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """에러 패턴 분석"""
        try:
            from datetime import datetime, timedelta

            cutoff_time = datetime.now() - timedelta(hours=hours)
            error_data = []

            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # 에러 로그만 필터링
                        if log_entry.get('level') == 'ERROR':
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                            if log_time >= cutoff_time:
                                error_data.append(log_entry)

                    except (json.JSONDecodeError, ValueError):
                        continue

            if not error_data:
                return {'total_errors': 0}

            # 에러 패턴 분석
            error_types = {}
            module_errors = {}
            hourly_distribution = {}

            for error in error_data:
                # 에러 타입별 집계
                error_type = error.get('exception', {}).get('type', 'Unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1

                # 모듈별 집계
                module = error.get('module', 'unknown')
                module_errors[module] = module_errors.get(module, 0) + 1

                # 시간대별 집계
                hour = datetime.fromisoformat(error['timestamp'].replace('Z', '+00:00')).hour
                hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1

            return {
                'total_errors': len(error_data),
                'error_types': error_types,
                'module_errors': module_errors,
                'hourly_distribution': hourly_distribution,
                'recent_errors': error_data[-10:]  # 최근 10개 에러
            }

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"에러 패턴 분석 오류: {e}")
            return {}

    def generate_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """로그 요약 리포트 생성"""
        try:
            from datetime import datetime, timedelta

            cutoff_time = datetime.now() - timedelta(hours=hours)
            log_levels = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
            total_logs = 0

            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                        if log_time >= cutoff_time:
                            level = log_entry.get('level', 'INFO')
                            log_levels[level] = log_levels.get(level, 0) + 1
                            total_logs += 1

                    except (json.JSONDecodeError, ValueError):
                        continue

            # 성능 및 에러 분석 결합
            performance_analysis = self.analyze_performance_logs(hours)
            error_analysis = self.analyze_error_patterns(hours)

            return {
                'summary_period_hours': hours,
                'total_log_entries': total_logs,
                'log_level_distribution': log_levels,
                'error_rate': (log_levels.get('ERROR', 0) + log_levels.get('CRITICAL', 0)) / max(total_logs, 1),
                'performance_summary': {
                    'total_operations': performance_analysis.get('total_operations', 0),
                    'avg_execution_time': performance_analysis.get('avg_execution_time', 0),
                    'slow_operations_count': len(performance_analysis.get('slow_operations', []))
                },
                'error_summary': {
                    'total_errors': error_analysis.get('total_errors', 0),
                    'top_error_types': dict(sorted(
                        error_analysis.get('error_types', {}).items(),
                        key=lambda x: x[1], reverse=True
                    )[:5])
                }
            }

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"로그 요약 생성 오류: {e}")
            return {}


class LogMonitor:
    """실시간 로그 모니터링"""

    def __init__(self, log_file: str, alert_threshold: Dict[str, int] = None):
        self.log_file = log_file
        self.alert_threshold = alert_threshold or {
            'error_rate_per_minute': 5,
            'slow_operation_threshold': 30,
            'memory_usage_threshold': 1000  # MB
        }
        self.alerts = []

    def monitor_realtime(self, callback_func=None):
        """실시간 로그 모니터링"""
        import time
        from collections import deque

        recent_errors = deque(maxlen=100)
        recent_operations = deque(maxlen=100)

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                # 파일 끝으로 이동
                f.seek(0, 2)

                while True:
                    line = f.readline()

                    if not line:
                        time.sleep(1)
                        continue

                    try:
                        log_entry = json.loads(line.strip())

                        # 에러 모니터링
                        if log_entry.get('level') in ['ERROR', 'CRITICAL']:
                            recent_errors.append(log_entry)
                            self._check_error_rate(recent_errors)

                        # 성능 모니터링
                        if 'performance' in log_entry:
                            recent_operations.append(log_entry)
                            self._check_performance(log_entry)

                        # 콜백 함수 호출
                        if callback_func:
                            callback_func(log_entry)

                    except (json.JSONDecodeError, ValueError):
                        continue

        except KeyboardInterrupt:
            print("로그 모니터링 중단")
        except Exception as e:
            print(f"로그 모니터링 오류: {e}")

    def _check_error_rate(self, recent_errors):
        """에러 비율 체크"""
        from datetime import datetime, timedelta

        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        recent_error_count = sum(
            1 for error in recent_errors
            if datetime.fromisoformat(error['timestamp'].replace('Z', '+00:00')) > one_minute_ago
        )

        if recent_error_count > self.alert_threshold['error_rate_per_minute']:
            alert = {
                'type': 'high_error_rate',
                'message': f"높은 에러율 감지: {recent_error_count}개/분",
                'timestamp': now.isoformat(),
                'severity': 'high'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

    def _check_performance(self, log_entry):
        """성능 체크"""
        from datetime import datetime

        performance = log_entry.get('performance', {})
        execution_time = performance.get('execution_time', 0)
        memory_usage = performance.get('memory_usage', 0)

        # 느린 작업 체크
        if execution_time > self.alert_threshold['slow_operation_threshold']:
            alert = {
                'type': 'slow_operation',
                'message': f"느린 작업 감지: {log_entry.get('function', 'unknown')} ({execution_time:.2f}초)",
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

        # 높은 메모리 사용량 체크
        if memory_usage > self.alert_threshold['memory_usage_threshold']:
            alert = {
                'type': 'high_memory_usage',
                'message': f"높은 메모리 사용량: {memory_usage:.1f}MB",
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

    def get_recent_alerts(self, hours: int = 1) -> list:
        """최근 알림 조회"""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]


class DatabaseLogHandler(logging.Handler):
    """데이터베이스 로그 핸들러 (선택사항)"""

    def __init__(self, db_manager, table_name: str = "SYSTEM_LOGS"):
        super().__init__()
        self.db_manager = db_manager
        self.table_name = table_name
        self._ensure_log_table()

    def _ensure_log_table(self):
        """로그 테이블 생성"""
        create_sql = f"""
        CREATE TABLE {self.table_name} (
            ID NUMBER PRIMARY KEY,
            TIMESTAMP TIMESTAMP,
            LEVEL VARCHAR2(20),
            LOGGER VARCHAR2(100),
            MESSAGE CLOB,
            MODULE VARCHAR2(100),
            FUNCTION VARCHAR2(100),
            LINE_NUMBER NUMBER,
            EXCEPTION_INFO CLOB,
            CONTEXT_DATA CLOB
        )
        """

        try:
            self.db_manager.execute_dml(create_sql)
        except Exception:
            # 테이블이 이미 존재하는 경우 무시
            pass

        # 시퀀스 생성
        seq_sql = f"CREATE SEQUENCE SEQ_{self.table_name} START WITH 1 INCREMENT BY 1"
        try:
            self.db_manager.execute_dml(seq_sql)
        except Exception:
            pass

    def emit(self, record):
        """로그 레코드를 데이터베이스에 저장"""
        try:
            # 컨텍스트 데이터 수집
            context_data = {}
            for attr in ['user_id', 'request_id', 'operation_id', 'model_name']:
                if hasattr(record, attr):
                    context_data[attr] = getattr(record, attr)

            insert_sql = f"""
            INSERT INTO {self.table_name} 
            (ID, TIMESTAMP, LEVEL, LOGGER, MESSAGE, MODULE, FUNCTION, 
             LINE_NUMBER, EXCEPTION_INFO, CONTEXT_DATA)
            VALUES (SEQ_{self.table_name}.NEXTVAL, :1, :2, :3, :4, :5, :6, :7, :8, :9)
            """

            params = [
                datetime.fromtimestamp(record.created),
                record.levelname,
                record.name,
                record.getMessage(),
                record.module,
                record.funcName,
                record.lineno,
                self.format(record) if record.exc_info else None,
                json.dumps(context_data) if context_data else None
            ]

            self.db_manager.execute_dml(insert_sql, params)

        except Exception as e:
            # 로그 저장 실패 시 콘솔에 출력
            print(f"Failed to save log to database: {e}")


# 전역 로깅 매니저 초기화
def setup_application_logging(config: Dict[str, Any] = None):
    """애플리케이션 로깅 설정"""
    if config is None:
        config = {
            'log_level': 'INFO',
            'log_format': 'structured',
            'log_file': 'logs/application.log',
            'enable_console': True
        }

    logging_manager = LoggingManager(**config)

    # 특정 로거들의 레벨 조정
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    return logging_manager


# 로깅 유틸리티 함수들
def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """함수 호출 로깅"""
    logger = logging.getLogger(__name__)

    kwargs_str = ', '.join(f"{k}={v}" for k, v in (kwargs or {}).items())
    args_str = ', '.join(str(arg) for arg in args)

    all_args = ', '.join(filter(None, [args_str, kwargs_str]))

    logger.debug(f"함수 호출: {func_name}({all_args})")


def log_data_operation(operation: str, data_type: str, count: int, success: bool = True):
    """데이터 작업 로깅"""
    logger = logging.getLogger('data_operations')

    level = logging.INFO if success else logging.ERROR
    status = "성공" if success else "실패"

    logger.log(level, f"데이터 작업 {status}: {operation} - {data_type} {count}개",
              extra={
                  'operation_type': operation,
                  'data_type': data_type,
                  'record_count': count,
                  'success': success
              })


def log_system_event(event_type: str, description: str, severity: str = 'info'):
    """시스템 이벤트 로깅"""
    logger = logging.getLogger('system_events')

    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    level = level_map.get(severity.lower(), logging.INFO)

    logger.log(level, description, extra={
        'event_type': event_type,
        'severity': severity
    })


# 사용 예시
if __name__ == "__main__":
    # 로깅 시스템 설정
    setup_application_logging({
        'log_level': 'DEBUG',
        'log_format': 'structured',
        'log_file': 'logs/test.log',
        'enable_console': True
    })

    logger = logging.getLogger(__name__)

    # 일반 로그
    logger.info("애플리케이션 시작")

    # 컨텍스트가 있는 로그
    with LogContext(user_id="test_user", operation_id="op_001"):
        logger.info("사용자 작업 시작")

        try:
            # 에러 시뮬레이션
            raise ValueError("테스트 에러")
        except Exception:
            logger.error("작업 중 오류 발생", exc_info=True)

    # 성능 로깅
    @log_performance()
    def test_function():
        import time
        time.sleep(0.1)
        return "완료"

    result = test_function()
    logger.info(f"테스트 함수 결과: {result}")

    # 데이터 작업 로깅
    log_data_operation("INSERT", "reasoning_data", 100, success=True)
    log_data_operation("UPDATE", "evaluation_results", 50, success=False)

    # 시스템 이벤트 로깅
    log_system_event("startup", "시스템 초기화 완료", "info")
    log_system_event("database", "연결 풀 생성 실패", "error")

    # 로그 분석 예시
    try:
        analyzer = LogAnalyzer('logs/test.log')
        summary = analyzer.generate_log_summary(hours=1)
        print("로그 요약:", json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"로그 분석 실패: {e}")

    print("로깅 시스템 테스트 완료") = getattr(record, attr)

        # 성능 정보 추가
        if hasattr(record, 'execution_time'):
            log_data['performance'] = {
                'execution_time': record.execution_time,
                'memory_usage': getattr(record, 'memory_usage', None)
            }

        return json.dumps(log_data, ensure_ascii=False)


class SimpleFormatter(logging.Formatter):
    """간단한 텍스트 포매터"""

    def __init__(self):
        super().__init__(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )


class LoggingManager:
    """로깅 관리자"""

    def __init__(self,
                 log_level: str = "INFO",
                 log_format: str = "structured",  # "structured" or "simple"
                 log_file: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 enable_console: bool = True):

        self.log_level = log_level.upper()
        self.log_format = log_format
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.enable_console = enable_console

        self._setup_logging()

    def _setup_logging(self):
        """로깅 시스템 설정"""
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.log_level))

        # 기존 핸들러 제거
        root_logger.handlers.clear()

        # 포매터 선택
        if self.log_format == "structured":
            formatter = StructuredFormatter()
            console_formatter = SimpleFormatter()  # 콘솔은 읽기 쉽게
        else:
            formatter = SimpleFormatter()
            console_formatter = formatter

        # 콘솔 핸들러
        if self.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(logging.INFO)  # 콘솔은 INFO 이상만
            root_logger.addHandler(console_handler)

        # 파일 핸들러
        if self.log_file:
            log_dir = Path(self.log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        # 에러 전용 파일 핸들러
        if self.log_file:
            error_log_file = str(Path(self.log_file).with_suffix('.error.log'))
            error_handler = logging.handlers.RotatingFileHandler(
                error_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            root_logger.addHandler(error_handler)

        # 성능 로그 핸들러
        if self.log_file:
            perf_log_file = str(Path(self.log_file).with_suffix('.performance.log'))
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            perf_handler.addFilter(lambda record: hasattr(record, 'execution_time'))
            perf_handler.setFormatter(formatter)

            # 성능 로거 설정
            perf_logger = logging.getLogger('performance')
            perf_logger.addHandler(perf_handler)
            perf_logger.propagate = False


class LogContext:
    """로그 컨텍스트 관리자"""

    def __init__(self, **context):
        self.context = context
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record

        logging.setLogRecordFactory(record_factory)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


def with_logging_context(**context):
    """로깅 컨텍스트 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with LogContext(**context):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def log_performance(logger_name: str = 'performance'):
    """성능 로깅 데코레이터"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            import psutil

            logger = logging.getLogger(logger_name)

            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**2)

            try:
                result = func(*args, **kwargs)

                end_time = time.time()
                end_memory = psutil.Process().memory_info().rss / (1024**2)

                execution_time = end_time - start_time
                memory_diff = end_memory - start_memory

                # 성능 로그 기록
                log_record = logger.makeRecord(
                    logger.name, logging.INFO, func.__code__.co_filename,
                    func.__code__.co_firstlineno,
                    f"Function '{func.__name__}' completed successfully",
                    (), None, func.__name__
                )

                # 성능 정보 추가
                log_record.execution_time = execution_time
                log_record.memory_usage = memory_diff
                log_record.function_name = func.__name__

                logger.handle(log_record)

                return result

            except Exception as e:
            logger.error(f"성능 로그 분석 오류: {e}")
            return {}

    def analyze_error_patterns(self, hours: int = 24) -> Dict[str, Any]:
        """에러 패턴 분석"""
        try:
            from datetime import datetime, timedelta

            cutoff_time = datetime.now() - timedelta(hours=hours)
            error_data = []

            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # 에러 로그만 필터링
                        if log_entry.get('level') == 'ERROR':
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                            if log_time >= cutoff_time:
                                error_data.append(log_entry)

                    except (json.JSONDecodeError, ValueError):
                        continue

            if not error_data:
                return {'total_errors': 0}

            # 에러 패턴 분석
            error_types = {}
            module_errors = {}
            hourly_distribution = {}

            for error in error_data:
                # 에러 타입별 집계
                error_type = error.get('exception', {}).get('type', 'Unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1

                # 모듈별 집계
                module = error.get('module', 'unknown')
                module_errors[module] = module_errors.get(module, 0) + 1

                # 시간대별 집계
                hour = datetime.fromisoformat(error['timestamp'].replace('Z', '+00:00')).hour
                hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1

            return {
                'total_errors': len(error_data),
                'error_types': error_types,
                'module_errors': module_errors,
                'hourly_distribution': hourly_distribution,
                'recent_errors': error_data[-10:]  # 최근 10개 에러
            }

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"에러 패턴 분석 오류: {e}")
            return {}

    def generate_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """로그 요약 리포트 생성"""
        try:
            from datetime import datetime, timedelta

            cutoff_time = datetime.now() - timedelta(hours=hours)
            log_levels = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
            total_logs = 0

            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                        if log_time >= cutoff_time:
                            level = log_entry.get('level', 'INFO')
                            log_levels[level] = log_levels.get(level, 0) + 1
                            total_logs += 1

                    except (json.JSONDecodeError, ValueError):
                        continue

            # 성능 및 에러 분석 결합
            performance_analysis = self.analyze_performance_logs(hours)
            error_analysis = self.analyze_error_patterns(hours)

            return {
                'summary_period_hours': hours,
                'total_log_entries': total_logs,
                'log_level_distribution': log_levels,
                'error_rate': (log_levels.get('ERROR', 0) + log_levels.get('CRITICAL', 0)) / max(total_logs, 1),
                'performance_summary': {
                    'total_operations': performance_analysis.get('total_operations', 0),
                    'avg_execution_time': performance_analysis.get('avg_execution_time', 0),
                    'slow_operations_count': len(performance_analysis.get('slow_operations', []))
                },
                'error_summary': {
                    'total_errors': error_analysis.get('total_errors', 0),
                    'top_error_types': dict(sorted(
                        error_analysis.get('error_types', {}).items(),
                        key=lambda x: x[1], reverse=True
                    )[:5])
                }
            }

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"로그 요약 생성 오류: {e}")
            return {}


class LogMonitor:
    """실시간 로그 모니터링"""

    def __init__(self, log_file: str, alert_threshold: Dict[str, int] = None):
        self.log_file = log_file
        self.alert_threshold = alert_threshold or {
            'error_rate_per_minute': 5,
            'slow_operation_threshold': 30,
            'memory_usage_threshold': 1000  # MB
        }
        self.alerts = []

    def monitor_realtime(self, callback_func=None):
        """실시간 로그 모니터링"""
        import time
        from collections import deque

        recent_errors = deque(maxlen=100)
        recent_operations = deque(maxlen=100)

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                # 파일 끝으로 이동
                f.seek(0, 2)

                while True:
                    line = f.readline()

                    if not line:
                        time.sleep(1)
                        continue

                    try:
                        log_entry = json.loads(line.strip())

                        # 에러 모니터링
                        if log_entry.get('level') in ['ERROR', 'CRITICAL']:
                            recent_errors.append(log_entry)
                            self._check_error_rate(recent_errors)

                        # 성능 모니터링
                        if 'performance' in log_entry:
                            recent_operations.append(log_entry)
                            self._check_performance(log_entry)

                        # 콜백 함수 호출
                        if callback_func:
                            callback_func(log_entry)

                    except (json.JSONDecodeError, ValueError):
                        continue

        except KeyboardInterrupt:
            print("로그 모니터링 중단")
        except Exception as e:
            print(f"로그 모니터링 오류: {e}")

    def _check_error_rate(self, recent_errors):
        """에러 비율 체크"""
        from datetime import datetime, timedelta

        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        recent_error_count = sum(
            1 for error in recent_errors
            if datetime.fromisoformat(error['timestamp'].replace('Z', '+00:00')) > one_minute_ago
        )

        if recent_error_count > self.alert_threshold['error_rate_per_minute']:
            alert = {
                'type': 'high_error_rate',
                'message': f"높은 에러율 감지: {recent_error_count}개/분",
                'timestamp': now.isoformat(),
                'severity': 'high'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

    def _check_performance(self, log_entry):
        """성능 체크"""
        from datetime import datetime

        performance = log_entry.get('performance', {})
        execution_time = performance.get('execution_time', 0)
        memory_usage = performance.get('memory_usage', 0)

        # 느린 작업 체크
        if execution_time > self.alert_threshold['slow_operation_threshold']:
            alert = {
                'type': 'slow_operation',
                'message': f"느린 작업 감지: {log_entry.get('function', 'unknown')} ({execution_time:.2f}초)",
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

        # 높은 메모리 사용량 체크
        if memory_usage > self.alert_threshold['memory_usage_threshold']:
            alert = {
                'type': 'high_memory_usage',
                'message': f"높은 메모리 사용량: {memory_usage:.1f}MB",
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

    def get_recent_alerts(self, hours: int = 1) -> list:
        """최근 알림 조회"""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]


class DatabaseLogHandler(logging.Handler):
    """데이터베이스 로그 핸들러 (선택사항)"""

    def __init__(self, db_manager, table_name: str = "SYSTEM_LOGS"):
        super().__init__()
        self.db_manager = db_manager
        self.table_name = table_name
        self._ensure_log_table()

    def _ensure_log_table(self):
        """로그 테이블 생성"""
        create_sql = f"""
        CREATE TABLE {self.table_name} (
            ID NUMBER PRIMARY KEY,
            TIMESTAMP TIMESTAMP,
            LEVEL VARCHAR2(20),
            LOGGER VARCHAR2(100),
            MESSAGE CLOB,
            MODULE VARCHAR2(100),
            FUNCTION VARCHAR2(100),
            LINE_NUMBER NUMBER,
            EXCEPTION_INFO CLOB,
            CONTEXT_DATA CLOB
        )
        """

        try:
            self.db_manager.execute_dml(create_sql)
        except Exception:
            # 테이블이 이미 존재하는 경우 무시
            pass

        # 시퀀스 생성
        seq_sql = f"CREATE SEQUENCE SEQ_{self.table_name} START WITH 1 INCREMENT BY 1"
        try:
            self.db_manager.execute_dml(seq_sql)
        except Exception:
            pass

    def emit(self, record):
        """로그 레코드를 데이터베이스에 저장"""
        try:
            # 컨텍스트 데이터 수집
            context_data = {}
            for attr in ['user_id', 'request_id', 'operation_id', 'model_name']:
                if hasattr(record, attr):
                    context_data[attr] = getattr(record, attr)

            insert_sql = f"""
            INSERT INTO {self.table_name} 
            (ID, TIMESTAMP, LEVEL, LOGGER, MESSAGE, MODULE, FUNCTION, 
             LINE_NUMBER, EXCEPTION_INFO, CONTEXT_DATA)
            VALUES (SEQ_{self.table_name}.NEXTVAL, :1, :2, :3, :4, :5, :6, :7, :8, :9)
            """

            params = [
                datetime.fromtimestamp(record.created),
                record.levelname,
                record.name,
                record.getMessage(),
                record.module,
                record.funcName,
                record.lineno,
                self.format(record) if record.exc_info else None,
                json.dumps(context_data) if context_data else None
            ]

            self.db_manager.execute_dml(insert_sql, params)

        except Exception as e:
            # 로그 저장 실패 시 콘솔에 출력
            print(f"Failed to save log to database: {e}")


# 전역 로깅 매니저 초기화
def setup_application_logging(config: Dict[str, Any] = None):
    """애플리케이션 로깅 설정"""
    if config is None:
        config = {
            'log_level': 'INFO',
            'log_format': 'structured',
            'log_file': 'logs/application.log',
            'enable_console': True
        }

    logging_manager = LoggingManager(**config)

    # 특정 로거들의 레벨 조정
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    return logging_manager


# 로깅 유틸리티 함수들
def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """함수 호출 로깅"""
    logger = logging.getLogger(__name__)

    kwargs_str = ', '.join(f"{k}={v}" for k, v in (kwargs or {}).items())
    args_str = ', '.join(str(arg) for arg in args)

    all_args = ', '.join(filter(None, [args_str, kwargs_str]))

    logger.debug(f"함수 호출: {func_name}({all_args})")


def log_data_operation(operation: str, data_type: str, count: int, success: bool = True):
    """데이터 작업 로깅"""
    logger = logging.getLogger('data_operations')

    level = logging.INFO if success else logging.ERROR
    status = "성공" if success else "실패"

    logger.log(level, f"데이터 작업 {status}: {operation} - {data_type} {count}개",
              extra={
                  'operation_type': operation,
                  'data_type': data_type,
                  'record_count': count,
                  'success': success
              })


def log_system_event(event_type: str, description: str, severity: str = 'info'):
    """시스템 이벤트 로깅"""
    logger = logging.getLogger('system_events')

    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    level = level_map.get(severity.lower(), logging.INFO)

    logger.log(level, description, extra={
        'event_type': event_type,
        'severity': severity
    })


# 사용 예시
if __name__ == "__main__":
    # 로깅 시스템 설정
    setup_application_logging({
        'log_level': 'DEBUG',
        'log_format': 'structured',
        'log_file': 'logs/test.log',
        'enable_console': True
    })

    logger = logging.getLogger(__name__)

    # 일반 로그
    logger.info("애플리케이션 시작")

    # 컨텍스트가 있는 로그
    with LogContext(user_id="test_user", operation_id="op_001"):
        logger.info("사용자 작업 시작")

        try:
            # 에러 시뮬레이션
            raise ValueError("테스트 에러")
        except Exception:
            logger.error("작업 중 오류 발생", exc_info=True)

    # 성능 로깅
    @log_performance()
    def test_function():
        import time
        time.sleep(0.1)
        return "완료"

    result = test_function()
    logger.info(f"테스트 함수 결과: {result}")

    # 데이터 작업 로깅
    log_data_operation("INSERT", "reasoning_data", 100, success=True)
    log_data_operation("UPDATE", "evaluation_results", 50, success=False)

    # 시스템 이벤트 로깅
    log_system_event("startup", "시스템 초기화 완료", "info")
    log_system_event("database", "연결 풀 생성 실패", "error")

    # 로그 분석 예시
    try:
        analyzer = LogAnalyzer('logs/test.log')
        summary = analyzer.generate_log_summary(hours=1)
        print("로그 요약:", json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"로그 분석 실패: {e}")

    print("로깅 시스템 테스트 완료")', encoding='utf-8') as f:
                for log_entry in aggregated_logs:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            print(f"로그 집계 완료: {len(aggregated_logs)}개 항목 -> {output_file}")
            return len(aggregated_logs)

        except Exception as e:
            print(f"집계 로그 저장 오류: {e}")
            return 0

    def generate_daily_report(self, date_str: str = None) -> Dict[str, Any]:
        """일일 리포트 생성"""
        from datetime import datetime, timedelta

        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        else:
            target_date = datetime.now().date()

        start_time = datetime.combine(target_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)

        daily_stats = {
            'date': target_date.isoformat(),
            'total_logs': 0,
            'error_count': 0,
            'warning_count': 0,
            'performance_issues': 0,
            'top_errors': {},
            'peak_hours': {},
            'system_health': 'good'
        }

        hour_counts = {i: 0 for i in range(24)}

        for log_file in self.log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                            if start_time <= log_time < end_time:
                                daily_stats['total_logs'] += 1
                                hour_counts[log_time.hour] += 1

                                level = log_entry.get('level', 'INFO')
                                if level == 'ERROR':
                                    daily_stats['error_count'] += 1
                                    error_type = log_entry.get('exception', {}).get('type', 'Unknown')
                                    daily_stats['top_errors'][error_type] = daily_stats['top_errors'].get(error_type, 0) + 1
                                elif level == 'WARNING':
                                    daily_stats['warning_count'] += 1

                                # 성능 이슈 체크
                                if 'performance' in log_entry:
                                    exec_time = log_entry['performance'].get('execution_time', 0)
                                    if exec_time > 10:  # 10초 이상
                                        daily_stats['performance_issues'] += 1

                        except (json.JSONDecodeError, ValueError):
                            continue

            except Exception as e:
                print(f"일일 리포트 생성 중 오류 ({log_file}): {e}")
                continue

        # 피크 시간 분석
        daily_stats['peak_hours'] = dict(sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3])

        # 시스템 상태 평가
        error_rate = daily_stats['error_count'] / max(daily_stats['total_logs'], 1)
        if error_rate > 0.05:  # 5% 이상
            daily_stats['system_health'] = 'critical'
        elif error_rate > 0.02:  # 2% 이상
            daily_stats['system_health'] = 'warning'

        return daily_stats


# 전역 로깅 매니저 초기화
def setup_application_logging(config: Dict[str, Any] = None):
    """애플리케이션 로깅 설정"""
    if config is None:
        config = {
            'log_level': 'INFO',
            'log_format': 'structured',
            'log_file': 'logs/application.log',
            'enable_console': True
        }

    logging_manager = LoggingManager(**config)

    # 특정 로거들의 레벨 조정
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    return logging_manager


# 로깅 유틸리티 함수들
def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """함수 호출 로깅"""
    logger = logging.getLogger(__name__)

    kwargs_str = ', '.join(f"{k}={v}" for k, v in (kwargs or {}).items())
    args_str = ', '.join(str(arg) for arg in args)

    all_args = ', '.join(filter(None, [args_str, kwargs_str]))

    logger.debug(f"함수 호출: {func_name}({all_args})")


def log_data_operation(operation: str, data_type: str, count: int, success: bool = True):
    """데이터 작업 로깅"""
    logger = logging.getLogger('data_operations')

    level = logging.INFO if success else logging.ERROR
    status = "성공" if success else "실패"

    logger.log(level, f"데이터 작업 {status}: {operation} - {data_type} {count}개",
              extra={
                  'operation_type': operation,
                  'data_type': data_type,
                  'record_count': count,
                  'success': success
              })


def log_system_event(event_type: str, description: str, severity: str = 'info'):
    """시스템 이벤트 로깅"""
    logger = logging.getLogger('system_events')

    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    level = level_map.get(severity.lower(), logging.INFO)

    logger.log(level, description, extra={
        'event_type': event_type,
        'severity': severity
    })


# 백그라운드 로그 관리 스케줄러
class LogScheduler:
    """로그 관리 스케줄러"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.rotation_manager = LogRotationManager(log_dir)
        self.running = False

    def start_background_tasks(self):
        """백그라운드 작업 시작"""
        import threading
        import time
        from datetime import datetime

        self.running = True

        def background_worker():
            while self.running:
                try:
                    # 매일 자정에 로그 정리 작업 실행
                    now = datetime.now()
                    if now.hour == 0 and now.minute == 0:
                        self.rotation_manager.cleanup_old_logs()
                        self.rotation_manager.compress_old_logs()
                        time.sleep(60)  # 1분 대기로 중복 실행 방지

                    time.sleep(30)  # 30초마다 체크

                except Exception as e:
                    print(f"백그라운드 로그 관리 오류: {e}")
                    time.sleep(60)

        worker_thread = threading.Thread(target=background_worker, daemon=True)
        worker_thread.start()
        print("백그라운드 로그 관리 작업 시작")

    def stop_background_tasks(self):
        """백그라운드 작업 중지"""
        self.running = False
        print("백그라운드 로그 관리 작업 중지")


# 사용 예시
if __name__ == "__main__":
    # 로깅 시스템 설정
    setup_application_logging({
        'log_level': 'DEBUG',
        'log_format': 'structured',
        'log_file': 'logs/test.log',
        'enable_console': True
    })

    logger = logging.getLogger(__name__)

    # 일반 로그
    logger.info("애플리케이션 시작")

    # 컨텍스트가 있는 로그
    with LogContext(user_id="test_user", operation_id="op_001"):
        logger.info("사용자 작업 시작")

        try:
            # 에러 시뮬레이션
            raise ValueError("테스트 에러")
        except Exception:
            logger.error("작업 중 오류 발생", exc_info=True)

    # 성능 로깅
    @log_performance()
    def test_function():
        import time
        time.sleep(0.1)
        return "완료"

    result = test_function()
    logger.info(f"테스트 함수 결과: {result}")

    # 데이터 작업 로깅
    log_data_operation("INSERT", "reasoning_data", 100, success=True)
    log_data_operation("UPDATE", "evaluation_results", 50, success=False)

    # 시스템 이벤트 로깅
    log_system_event("startup", "시스템 초기화 완료", "info")
    log_system_event("database", "연결 풀 생성 실패", "error")

    # 로그 분석 예시
    try:
        analyzer = LogAnalyzer('logs/test.log')
        summary = analyzer.generate_log_summary(hours=1)
        print("로그 요약:", json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"로그 분석 실패: {e}")

    # 백그라운드 로그 관리 시작
    scheduler = LogScheduler()
    scheduler.start_background_tasks()

    print("로깅 시스템 테스트 완료") recent_errors):
        """에러 비율 체크"""
        from datetime import datetime, timedelta

        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        recent_error_count = sum(
            1 for error in recent_errors
            if datetime.fromisoformat(error['timestamp'].replace('Z', '+00:00')) > one_minute_ago
        )

        if recent_error_count > self.alert_threshold['error_rate_per_minute']:
            alert = {
                'type': 'high_error_rate',
                'message': f"높은 에러율 감지: {recent_error_count}개/분",
                'timestamp': now.isoformat(),
                'severity': 'high'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

    def _check_performance(self, log_entry):
        """성능 체크"""
        performance = log_entry.get('performance', {})
        execution_time = performance.get('execution_time', 0)
        memory_usage = performance.get('memory_usage', 0)

        # 느린 작업 체크
        if execution_time > self.alert_threshold['slow_operation_threshold']:
            alert = {
                'type': 'slow_operation',
                'message': f"느린 작업 감지: {log_entry.get('function', 'unknown')} ({execution_time:.2f}초)",
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

        # 높은 메모리 사용량 체크
        if memory_usage > self.alert_threshold['memory_usage_threshold']:
            alert = {
                'type': 'high_memory_usage',
                'message': f"높은 메모리 사용량: {memory_usage:.1f}MB",
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

    def get_recent_alerts(self, hours: int = 1) -> list:
        """최근 알림 조회"""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]


class LogRotationManager:
    """로그 회전 관리자"""

    def __init__(self, log_dir: str = "logs", max_age_days: int = 30):
        self.log_dir = Path(log_dir)
        self.max_age_days = max_age_days

    def cleanup_old_logs(self):
        """오래된 로그 파일 정리"""
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=self.max_age_days)
        removed_files = []

        try:
            for log_file in self.log_dir.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    log_file.unlink()
                    removed_files.append(str(log_file))

            if removed_files:
                print(f"오래된 로그 파일 {len(removed_files)}개 정리 완료")
                return removed_files

        except Exception as e:
            print(f"로그 파일 정리 오류: {e}")

        return []

    def compress_old_logs(self):
        """오래된 로그 파일 압축"""
        import gzip
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=7)  # 7일 이전 파일 압축
        compressed_files = []

        try:
            for log_file in self.log_dir.glob("*.log"):
                if log_file.stat().st_mtime < cutoff_date.timestamp():
                    compressed_path = log_file.with_suffix('.log.gz')

                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            f_out.write(f_in.read())

                    log_file.unlink()  # 원본 파일 삭제
                    compressed_files.append(str(compressed_path))

            if compressed_files:
                print(f"로그 파일 {len(compressed_files)}개 압축 완료")
                return compressed_files

        except Exception as e:
            print(f"로그 파일 압축 오류: {e}")

        return []


class LogAggregator:
    """로그 집계자"""

    def __init__(self, log_files: list):
        self.log_files = log_files

    def aggregate_logs(self, output_file: str, hours: int = 24):
        """여러 로그 파일을 집계"""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)
        aggregated_logs = []

        for log_file in self.log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                            if log_time >= cutoff_time:
                                log_entry['source_file'] = log_file
                                aggregated_logs.append(log_entry)

                        except (json.JSONDecodeError, ValueError):
                            continue

            except Exception as e:
                print(f"로그 파일 읽기 오류 ({log_file}): {e}")
                continue

        # 타임스탬프 기준 정렬
        aggregated_logs.sort(key=lambda x: x['timestamp'])

        # 집계된 로그 저장
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for log_entry in aggregated_logs:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            print(f"로그 집계 완료: {len(aggregated_logs)}개 항목 -> {output_file}")
            return len(aggregated_logs)

        except Exception as e:
            print(f"집계 로그 저장 오류: {e}")
            return 0

    def generate_daily_report(self, date_str: str = None) -> Dict[str, Any]:
        """일일 리포트 생성"""
        from datetime import datetime, timedelta

        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        else:
            target_date = datetime.now().date()

        start_time = datetime.combine(target_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)

        daily_stats = {
            'date': target_date.isoformat(),
            'total_logs': 0,
            'error_count': 0,
            'warning_count': 0,
            'performance_issues': 0,
            'top_errors': {},
            'peak_hours': {},
            'system_health': 'good'
        }

        hour_counts = {i: 0 for i in range(24)}

        for log_file in self.log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                            if start_time <= log_time < end_time:
                                daily_stats['total_logs'] += 1
                                hour_counts[log_time.hour] += 1

                                level = log_entry.get('level', 'INFO')
                                if level == 'ERROR':
                                    daily_stats['error_count'] += 1
                                    error_type = log_entry.get('exception', {}).get('type', 'Unknown')
                                    daily_stats['top_errors'][error_type] = daily_stats['top_errors'].get(error_type, 0) + 1
                                elif level == 'WARNING':
                                    daily_stats['warning_count'] += 1

                                # 성능 이슈 체크
                                if 'performance' in log_entry:
                                    exec_time = log_entry['performance'].get('execution_time', 0)
                                    if exec_time > 10:  # 10초 이상
                                        daily_stats['performance_issues'] += 1

                        except (json.JSONDecodeError, ValueError):
                            continue

            except Exception as e:
                print(f"일일 리포트 생성 중 오류 ({log_file}): {e}")
                continue

        # 피크 시간 분석
        daily_stats['peak_hours'] = dict(sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3])

        # 시스템 상태 평가
        error_rate = daily_stats['error_count'] / max(daily_stats['total_logs'], 1)
        if error_rate > 0.05:  # 5% 이상
            daily_stats['system_health'] = 'critical'
        elif error_rate > 0.02:  # 2% 이상
            daily_stats['system_health'] = 'warning'

        return daily_stats', encoding='utf-8') as f:
                for log_entry in aggregated_logs:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')

            print(f"로그 집계 완료: {len(aggregated_logs)}개 항목 -> {output_file}")
            return len(aggregated_logs)

        except Exception as e:
            print(f"집계 로그 저장 오류: {e}")
            return 0

    def generate_daily_report(self, date_str: str = None) -> Dict[str, Any]:
        """일일 리포트 생성"""
        from datetime import datetime, timedelta

        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d')
        else:
            target_date = datetime.now().date()

        start_time = datetime.combine(target_date, datetime.min.time())
        end_time = start_time + timedelta(days=1)

        daily_stats = {
            'date': target_date.isoformat(),
            'total_logs': 0,
            'error_count': 0,
            'warning_count': 0,
            'performance_issues': 0,
            'top_errors': {},
            'peak_hours': {},
            'system_health': 'good'
        }

        hour_counts = {i: 0 for i in range(24)}

        for log_file in self.log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            log_entry = json.loads(line.strip())
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                            if start_time <= log_time < end_time:
                                daily_stats['total_logs'] += 1
                                hour_counts[log_time.hour] += 1

                                level = log_entry.get('level', 'INFO')
                                if level == 'ERROR':
                                    daily_stats['error_count'] += 1
                                    error_type = log_entry.get('exception', {}).get('type', 'Unknown')
                                    daily_stats['top_errors'][error_type] = daily_stats['top_errors'].get(error_type, 0) + 1
                                elif level == 'WARNING':
                                    daily_stats['warning_count'] += 1

                                # 성능 이슈 체크
                                if 'performance' in log_entry:
                                    exec_time = log_entry['performance'].get('execution_time', 0)
                                    if exec_time > 10:  # 10초 이상
                                        daily_stats['performance_issues'] += 1

                        except (json.JSONDecodeError, ValueError):
                            continue

            except Exception as e:
                print(f"일일 리포트 생성 중 오류 ({log_file}): {e}")
                continue

        # 피크 시간 분석
        daily_stats['peak_hours'] = dict(sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:3])

        # 시스템 상태 평가
        error_rate = daily_stats['error_count'] / max(daily_stats['total_logs'], 1)
        if error_rate > 0.05:  # 5% 이상
            daily_stats['system_health'] = 'critical'
        elif error_rate > 0.02:  # 2% 이상
            daily_stats['system_health'] = 'warning'

        return daily_stats


# 전역 로깅 매니저 초기화
def setup_application_logging(config: Dict[str, Any] = None):
    """애플리케이션 로깅 설정"""
    if config is None:
        config = {
            'log_level': 'INFO',
            'log_format': 'structured',
            'log_file': 'logs/application.log',
            'enable_console': True
        }

    logging_manager = LoggingManager(**config)

    # 특정 로거들의 레벨 조정
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    return logging_manager


# 로깅 유틸리티 함수들
def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """함수 호출 로깅"""
    logger = logging.getLogger(__name__)

    kwargs_str = ', '.join(f"{k}={v}" for k, v in (kwargs or {}).items())
    args_str = ', '.join(str(arg) for arg in args)

    all_args = ', '.join(filter(None, [args_str, kwargs_str]))

    logger.debug(f"함수 호출: {func_name}({all_args})")


def log_data_operation(operation: str, data_type: str, count: int, success: bool = True):
    """데이터 작업 로깅"""
    logger = logging.getLogger('data_operations')

    level = logging.INFO if success else logging.ERROR
    status = "성공" if success else "실패"

    logger.log(level, f"데이터 작업 {status}: {operation} - {data_type} {count}개",
              extra={
                  'operation_type': operation,
                  'data_type': data_type,
                  'record_count': count,
                  'success': success
              })


def log_system_event(event_type: str, description: str, severity: str = 'info'):
    """시스템 이벤트 로깅"""
    logger = logging.getLogger('system_events')

    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    level = level_map.get(severity.lower(), logging.INFO)

    logger.log(level, description, extra={
        'event_type': event_type,
        'severity': severity
    })


# 백그라운드 로그 관리 스케줄러
class LogScheduler:
    """로그 관리 스케줄러"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.rotation_manager = LogRotationManager(log_dir)
        self.running = False

    def start_background_tasks(self):
        """백그라운드 작업 시작"""
        import threading
        import time

        self.running = True

        def background_worker():
            while self.running:
                try:
                    # 매일 자정에 로그 정리 작업 실행
                    now = datetime.now()
                    if now.hour == 0 and now.minute == 0:
                        self.rotation_manager.cleanup_old_logs()
                        self.rotation_manager.compress_old_logs()
                        time.sleep(60)  # 1분 대기로 중복 실행 방지

                    time.sleep(30)  # 30초마다 체크

                except Exception as e:
                    print(f"백그라운드 로그 관리 오류: {e}")
                    time.sleep(60)

        worker_thread = threading.Thread(target=background_worker, daemon=True)
        worker_thread.start()
        print("백그라운드 로그 관리 작업 시작")

    def stop_background_tasks(self):
        """백그라운드 작업 중지"""
        self.running = False
        print("백그라운드 로그 관리 작업 중지")


# 사용 예시
if __name__ == "__main__":
    # 로깅 시스템 설정
    setup_application_logging({
        'log_level': 'DEBUG',
        'log_format': 'structured',
        'log_file': 'logs/test.log',
        'enable_console': True
    })

    logger = logging.getLogger(__name__)

    # 일반 로그
    logger.info("애플리케이션 시작")

    # 컨텍스트가 있는 로그
    with LogContext(user_id="test_user", operation_id="op_001"):
        logger.info("사용자 작업 시작")

        try:
            # 에러 시뮬레이션
            raise ValueError("테스트 에러")
        except Exception:
            logger.error("작업 중 오류 발생", exc_info=True)

    # 성능 로깅
    @log_performance()
    def test_function():
        import time
        time.sleep(0.1)
        return "완료"

    result = test_function()
    logger.info(f"테스트 함수 결과: {result}")

    # 데이터 작업 로깅
    log_data_operation("INSERT", "reasoning_data", 100, success=True)
    log_data_operation("UPDATE", "evaluation_results", 50, success=False)

    # 시스템 이벤트 로깅
    log_system_event("startup", "시스템 초기화 완료", "info")
    log_system_event("database", "연결 풀 생성 실패", "error")

    # 로그 분석 예시
    try:
        analyzer = LogAnalyzer('logs/test.log')
        summary = analyzer.generate_log_summary(hours=1)
        print("로그 요약:", json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"로그 분석 실패: {e}")

    # 백그라운드 로그 관리 시작
    scheduler = LogScheduler()
    scheduler.start_background_tasks()

    print("로깅 시스템 테스트 완료") recent_errors):
        """에러 비율 체크"""
        from datetime import datetime, timedelta

        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)

        recent_error_count = sum(
            1 for error in recent_errors
            if datetime.fromisoformat(error['timestamp'].replace('Z', '+00:00')) > one_minute_ago
        )

        if recent_error_count > self.alert_threshold['error_rate_per_minute']:
            alert = {
                'type': 'high_error_rate',
                'message': f"높은 에러율 감지: {recent_error_count}개/분",
                'timestamp': now.isoformat(),
                'severity': 'high'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

    def _check_performance(self, log_entry):
        """성능 체크"""
        performance = log_entry.get('performance', {})
        execution_time = performance.get('execution_time', 0)
        memory_usage = performance.get('memory_usage', 0)

        # 느린 작업 체크
        if execution_time > self.alert_threshold['slow_operation_threshold']:
            alert = {
                'type': 'slow_operation',
                'message': f"느린 작업 감지: {log_entry.get('function', 'unknown')} ({execution_time:.2f}초)",
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

        # 높은 메모리 사용량 체크
        if memory_usage > self.alert_threshold['memory_usage_threshold']:
            alert = {
                'type': 'high_memory_usage',
                'message': f"높은 메모리 사용량: {memory_usage:.1f}MB",
                'timestamp': datetime.now().isoformat(),
                'severity': 'medium'
            }
            self.alerts.append(alert)
            print(f"⚠️ 알림: {alert['message']}")

    def get_recent_alerts(self, hours: int = 1) -> list:
        """최근 알림 조회"""
        from datetime import datetime, timedelta

        cutoff_time = datetime.now() - timedelta(hours=hours)

        return [
            alert for alert in self.alerts
            if datetime.fromisoformat(alert['timestamp']) > cutoff_time
        ]


# 전역 로깅 매니저 초기화
def setup_application_logging(config: Dict[str, Any] = None):
    """애플리케이션 로깅 설정"""
    if config is None:
        config = {
            'log_level': 'INFO',
            'log_format': 'structured',
            'log_file': 'logs/application.log',
            'enable_console': True
        }

    logging_manager = LoggingManager(**config)

    # 특정 로거들의 레벨 조정
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    return logging_manager


# 로깅 유틸리티 함수들
def log_function_call(func_name: str, args: tuple = (), kwargs: dict = None):
    """함수 호출 로깅"""
    logger = logging.getLogger(__name__)

    kwargs_str = ', '.join(f"{k}={v}" for k, v in (kwargs or {}).items())
    args_str = ', '.join(str(arg) for arg in args)

    all_args = ', '.join(filter(None, [args_str, kwargs_str]))

    logger.debug(f"함수 호출: {func_name}({all_args})")


def log_data_operation(operation: str, data_type: str, count: int, success: bool = True):
    """데이터 작업 로깅"""
    logger = logging.getLogger('data_operations')

    level = logging.INFO if success else logging.ERROR
    status = "성공" if success else "실패"

    logger.log(level, f"데이터 작업 {status}: {operation} - {data_type} {count}개",
              extra={
                  'operation_type': operation,
                  'data_type': data_type,
                  'record_count': count,
                  'success': success
              })


def log_system_event(event_type: str, description: str, severity: str = 'info'):
    """시스템 이벤트 로깅"""
    logger = logging.getLogger('system_events')

    level_map = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }

    level = level_map.get(severity.lower(), logging.INFO)

    logger.log(level, description, extra={
        'event_type': event_type,
        'severity': severity
    })


# 사용 예시
if __name__ == "__main__":
    # 로깅 시스템 설정
    setup_application_logging({
        'log_level': 'DEBUG',
        'log_format': 'structured',
        'log_file': 'logs/test.log',
        'enable_console': True
    })

    logger = logging.getLogger(__name__)

    # 일반 로그
    logger.info("애플리케이션 시작")

    # 컨텍스트가 있는 로그
    with LogContext(user_id="test_user", operation_id="op_001"):
        logger.info("사용자 작업 시작")

        try:
            # 에러 시뮬레이션
            raise ValueError("테스트 에러")
        except Exception:
            logger.error("작업 중 오류 발생", exc_info=True)

    # 성능 로깅
    @log_performance()
    def test_function():
        import time
        time.sleep(0.1)
        return "완료"

    result = test_function()
    logger.info(f"테스트 함수 결과: {result}")

    # 데이터 작업 로깅
    log_data_operation("INSERT", "reasoning_data", 100, success=True)
    log_data_operation("UPDATE", "evaluation_results", 50, success=False)

    # 시스템 이벤트 로깅
    log_system_event("startup", "시스템 초기화 완료", "info")
    log_system_event("database", "연결 풀 생성 실패", "error")

    # 로그 분석 예시
    try:
        analyzer = LogAnalyzer('logs/test.log')
        summary = analyzer.generate_log_summary(hours=1)
        print("로그 요약:", json.dumps(summary, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"로그 분석 실패: {e}")

                end_time = time.time()
                execution_time = end_time - start_time

                logger.error(
                    f"Function '{func.__name__}' failed after {execution_time:.2f}s",
                    exc_info=True,
                    extra={'execution_time': execution_time, 'function_name': func.__name__}
                )
                raise

        return wrapper
    return decorator


class DatabaseLogHandler(logging.Handler):
    """데이터베이스 로그 핸들러 (선택사항)"""

    def __init__(self, db_manager, table_name: str = "SYSTEM_LOGS"):
        super().__init__()
        self.db_manager = db_manager
        self.table_name = table_name
        self._ensure_log_table()

    def _ensure_log_table(self):
        """로그 테이블 생성"""
        create_sql = f"""
        CREATE TABLE {self.table_name} (
            ID NUMBER PRIMARY KEY,
            TIMESTAMP TIMESTAMP,
            LEVEL VARCHAR2(20),
            LOGGER VARCHAR2(100),
            MESSAGE CLOB,
            MODULE VARCHAR2(100),
            FUNCTION VARCHAR2(100),
            LINE_NUMBER NUMBER,
            EXCEPTION_INFO CLOB,
            CONTEXT_DATA CLOB
        )
        """

        try:
            self.db_manager.execute_dml(create_sql)
        except Exception:
            # 테이블이 이미 존재하는 경우 무시
            pass

        # 시퀀스 생성
        seq_sql = f"CREATE SEQUENCE SEQ_{self.table_name} START WITH 1 INCREMENT BY 1"
        try:
            self.db_manager.execute_dml(seq_sql)
        except Exception:
            pass

    def emit(self, record):
        """로그 레코드를 데이터베이스에 저장"""
        try:
            # 컨텍스트 데이터 수집
            context_data = {}
            for attr in ['user_id', 'request_id', 'operation_id', 'model_name']:
                if hasattr(record, attr):
                    context_data[attr] = getattr(record, attr)

            insert_sql = f"""
            INSERT INTO {self.table_name} 
            (ID, TIMESTAMP, LEVEL, LOGGER, MESSAGE, MODULE, FUNCTION, 
             LINE_NUMBER, EXCEPTION_INFO, CONTEXT_DATA)
            VALUES (SEQ_{self.table_name}.NEXTVAL, :1, :2, :3, :4, :5, :6, :7, :8, :9)
            """

            params = [
                datetime.fromtimestamp(record.created),
                record.levelname,
                record.name,
                record.getMessage(),
                record.module,
                record.funcName,
                record.lineno,
                self.format(record) if record.exc_info else None,
                json.dumps(context_data) if context_data else None
            ]

            self.db_manager.execute_dml(insert_sql, params)

        except Exception as e:
            # 로그 저장 실패 시 콘솔에 출력
            print(f"Failed to save log to database: {e}")


class LogAnalyzer:
    """로그 분석기"""

    def __init__(self, log_file: str):
        self.log_file = log_file

    def analyze_performance_logs(self, hours: int = 24) -> Dict[str, Any]:
        """성능 로그 분석"""
        try:
            from datetime import datetime, timedelta

            cutoff_time = datetime.now() - timedelta(hours=hours)
            performance_data = []

            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())

                        # 성능 로그만 필터링
                        if 'performance' in log_entry:
                            log_time = datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))

                            if log_time >= cutoff_time:
                                performance_data.append(log_entry)

                    except (json.JSONDecodeError, ValueError):
                        continue

            if not performance_data:
                return {}

            # 성능 분석
            execution_times = [p['performance']['execution_time'] for p in performance_data]
            memory_usages = [p['performance'].get('memory_usage', 0) for p in performance_data]

            function_stats = {}
            for entry in performance_data:
                func_name = entry.get('function', 'unknown')
                if func_name not in function_stats:
                    function_stats[func_name] = {
                        'count': 0,
                        'total_time': 0,
                        'max_time': 0,
                        'min_time': float('inf')
                    }

                exec_time = entry['performance']['execution_time']
                function_stats[func_name]['count'] += 1
                function_stats[func_name]['total_time'] += exec_time
                function_stats[func_name]['max_time'] = max(function_stats[func_name]['max_time'], exec_time)
                function_stats[func_name]['min_time'] = min(function_stats[func_name]['min_time'], exec_time)

            # 평균 계산
            for stats in function_stats.values():
                stats['avg_time'] = stats['total_time'] / stats['count']
                if stats['min_time'] == float('inf'):
                    stats['min_time'] = 0

            return {
                'total_operations': len(performance_data),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'max_execution_time': max(execution_times),
                'min_execution_time': min(execution_times),
                'avg_memory_usage': sum(memory_usages) / len(memory_usages),
                'function_statistics': function_stats,
                'slow_operations': [
                    {
                        'function': p.get('function', 'unknown'),
                        'execution_time': p['performance']['execution_time'],
                        'timestamp': p['timestamp']
                    }
                    for p in performance_data
                    if p['performance']['execution_time'] > 10  # 10초 이상
                ]
            }

        except Exception as e: