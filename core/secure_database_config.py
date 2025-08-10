"""
보안 강화된 데이터베이스 설정 모듈
환경 변수 우선 사용 및 Oracle 호환성 자동 설정을 지원합니다.
"""
import os
import json
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from core.database_config import DatabaseConfig, DatabaseManager
from core.oracle_compatibility import configure_oracle_compatibility, get_optimized_connection_params
from core.error_handler import handle_database_error, format_user_error

logger = logging.getLogger(__name__)


@dataclass
class SecureDBConfig:
    """보안 강화된 데이터베이스 설정"""
    username: str
    password: str
    dsn: str
    pool_min: int = 1
    pool_max: int = 10
    pool_increment: int = 1
    pool_timeout: int = 30
    oracle_mode: str = "auto"  # auto, thin, thick
    use_environment: bool = True
    config_source: str = "unknown"


class SecureDatabaseConfigLoader:
    """보안 강화된 데이터베이스 설정 로더"""

    def __init__(self):
        self.env_prefix = "DB_"
        self.required_fields = ['username', 'password', 'dsn']

    def load_secure_config(self,
                           config_file: Optional[str] = None,
                           prefer_env: bool = True) -> SecureDBConfig:
        """
        보안 강화된 설정 로드

        우선순위:
        1. 환경 변수 (prefer_env=True인 경우)
        2. 설정 파일
        3. 기본값

        Args:
            config_file: 설정 파일 경로
            prefer_env: 환경 변수 우선 사용 여부

        Returns:
            SecureDBConfig: 보안 설정
        """
        logger.info("보안 강화된 데이터베이스 설정 로드 중...")

        config_data = {}
        sources = []

        # 1. 환경 변수에서 로드
        env_config = self._load_from_environment()
        if env_config:
            config_data.update(env_config)
            sources.append("환경변수")
            logger.info("환경 변수에서 데이터베이스 설정 로드")

        # 2. 설정 파일에서 로드 (환경 변수가 없는 필드만)
        if config_file:
            file_config = self._load_from_file(config_file)
            if file_config:
                # 환경 변수가 우선이면 누락된 필드만 채움
                if prefer_env:
                    for key, value in file_config.items():
                        if key not in config_data:
                            config_data[key] = value
                else:
                    # 파일 우선이면 파일 설정으로 덮어씀
                    config_data.update(file_config)
                sources.append("설정파일")
                logger.info(f"설정 파일에서 데이터베이스 설정 로드: {config_file}")

        # 3. 필수 필드 검증
        missing_fields = [field for field in self.required_fields if not config_data.get(field)]
        if missing_fields:
            raise ValueError(f"필수 데이터베이스 설정 필드가 누락되었습니다: {missing_fields}")

        # 4. 기본값 설정
        config_data.setdefault('pool_min', 1)
        config_data.setdefault('pool_max', 10)
        config_data.setdefault('pool_increment', 1)
        config_data.setdefault('pool_timeout', 30)
        config_data.setdefault('oracle_mode', 'auto')

        # 5. SecureDBConfig 생성
        secure_config = SecureDBConfig(
            username=config_data['username'],
            password=config_data['password'],
            dsn=config_data['dsn'],
            pool_min=config_data['pool_min'],
            pool_max=config_data['pool_max'],
            pool_increment=config_data['pool_increment'],
            pool_timeout=config_data['pool_timeout'],
            oracle_mode=config_data['oracle_mode'],
            use_environment=bool(env_config),
            config_source=", ".join(sources)
        )

        logger.info(f"데이터베이스 설정 로드 완료 (출처: {secure_config.config_source})")
        return secure_config

    def _load_from_environment(self) -> Dict[str, Any]:
        """환경 변수에서 설정 로드"""
        env_mapping = {
            'username': ['DB_USERNAME', 'DB_USER', 'ORACLE_USERNAME', 'ORACLE_USER'],
            'password': ['DB_PASSWORD', 'DB_PASS', 'ORACLE_PASSWORD', 'ORACLE_PASS'],
            'dsn': ['DB_DSN', 'DB_CONNECTION_STRING', 'ORACLE_DSN', 'ORACLE_CONNECTION_STRING'],
            'pool_min': ['DB_POOL_MIN'],
            'pool_max': ['DB_POOL_MAX'],
            'pool_increment': ['DB_POOL_INCREMENT'],
            'pool_timeout': ['DB_POOL_TIMEOUT'],
            'oracle_mode': ['DB_ORACLE_MODE', 'ORACLE_MODE']
        }

        config = {}

        for config_key, env_keys in env_mapping.items():
            for env_key in env_keys:
                value = os.getenv(env_key)
                if value:
                    # 숫자 필드 변환
                    if config_key in ['pool_min', 'pool_max', 'pool_increment', 'pool_timeout']:
                        try:
                            config[config_key] = int(value)
                        except ValueError:
                            logger.warning(f"환경 변수 {env_key} 값이 유효하지 않습니다: {value}")
                            continue
                    else:
                        config[config_key] = value

                    logger.debug(f"환경 변수 {env_key}에서 {config_key} 설정 로드")
                    break  # 첫 번째로 찾은 환경 변수 사용

        return config

    def _load_from_file(self, config_file: str) -> Dict[str, Any]:
        """설정 파일에서 로드"""
        try:
            if not os.path.exists(config_file):
                logger.warning(f"설정 파일이 존재하지 않습니다: {config_file}")
                return {}

            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            logger.debug(f"설정 파일에서 {len(config)}개 설정 로드: {config_file}")
            return config

        except json.JSONDecodeError as e:
            logger.error(f"설정 파일 JSON 파싱 오류: {e}")
            raise ValueError(f"설정 파일 형식이 올바르지 않습니다: {config_file}")
        except Exception as e:
            logger.error(f"설정 파일 로드 오류: {e}")
            raise

    def create_database_manager(self, secure_config: SecureDBConfig) -> DatabaseManager:
        """보안 설정으로부터 DatabaseManager 생성"""
        try:
            logger.info("데이터베이스 매니저 생성 중...")

            # Oracle 호환성 자동 설정
            logger.info("Oracle 호환성 확인 중...")
            compatibility_info = configure_oracle_compatibility(
                username=secure_config.username,
                password=secure_config.password,
                dsn=secure_config.dsn,
                preferred_mode=secure_config.oracle_mode
            )

            # 호환성 정보 로깅
            if compatibility_info.mode:
                logger.info(f"Oracle 연결 모드: {compatibility_info.mode.value}")
                if compatibility_info.warnings:
                    for warning in compatibility_info.warnings:
                        logger.warning(f"Oracle 호환성 경고: {warning}")
            else:
                logger.error("Oracle 연결 모드를 결정할 수 없습니다")
                raise ConnectionError("Oracle 데이터베이스에 연결할 수 없습니다")

            # 최적화된 연결 파라미터 생성
            base_params = {
                'user': secure_config.username,
                'password': secure_config.password,
                'dsn': secure_config.dsn
            }
            optimized_params = get_optimized_connection_params(base_params)

            # DatabaseConfig 생성
            db_config = DatabaseConfig(
                username=secure_config.username,
                password=secure_config.password,
                dsn=secure_config.dsn,
                pool_min=secure_config.pool_min,
                pool_max=secure_config.pool_max,
                pool_increment=secure_config.pool_increment,
                pool_timeout=secure_config.pool_timeout
            )

            # DatabaseManager 생성 및 연결 테스트
            db_manager = DatabaseManager(db_config)

            # 연결 테스트
            logger.info("데이터베이스 연결 테스트 중...")
            if not db_config.test_connection():
                raise ConnectionError("데이터베이스 연결 테스트 실패")

            logger.info("✅ 데이터베이스 매니저 생성 완료")
            return db_manager

        except Exception as e:
            logger.error(f"데이터베이스 매니저 생성 실패: {e}")
            # 사용자 친화적 에러 메시지
            user_error = format_user_error(e, "데이터베이스 매니저 생성")
            raise ConnectionError(user_error) from e

    def generate_sample_env_file(self, output_path: str = ".env.example") -> bool:
        """샘플 환경 변수 파일 생성"""
        try:
            env_content = """# LLM 추론 성능 평가 시스템 - 환경 변수 설정
# 이 파일을 .env로 복사하고 실제 값으로 수정하세요

# =============================================================================
# 데이터베이스 연결 설정 (필수)
# =============================================================================

# Oracle 데이터베이스 사용자명
DB_USERNAME=your_username

# Oracle 데이터베이스 패스워드
DB_PASSWORD=your_password

# 데이터베이스 DSN (Data Source Name)
# 형식 1: host:port/service_name (예: localhost:1521/XE)
# 형식 2: TNS 별칭 (tnsnames.ora에 정의된 이름)
DB_DSN=localhost:1521/XE

# =============================================================================
# 연결 풀 설정 (선택적)
# =============================================================================

# 최소 연결 수
DB_POOL_MIN=1

# 최대 연결 수
DB_POOL_MAX=10

# 연결 증가 단위
DB_POOL_INCREMENT=1

# 연결 타임아웃 (초)
DB_POOL_TIMEOUT=30

# =============================================================================
# Oracle 연결 모드 설정 (선택적)
# =============================================================================

# Oracle 연결 모드: auto, thin, thick
# auto: 자동 선택 (권장)
# thin: Thin 모드 (Oracle 12.1+ 필요, 클라이언트 라이브러리 불필요)
# thick: Thick 모드 (모든 Oracle 버전 지원, Oracle Client 라이브러리 필요)
DB_ORACLE_MODE=auto

# =============================================================================
# 추가 환경 변수 (필요시)
# =============================================================================

# Oracle Client 라이브러리 경로 (Thick 모드 사용시)
# ORACLE_HOME=/path/to/oracle/home
# ORACLE_INSTANT_CLIENT=/path/to/instantclient

# TNS 설정 파일 경로
# TNS_ADMIN=/path/to/tns/admin

# =============================================================================
# 사용법
# =============================================================================
# 1. 이 파일을 .env로 복사:
#    cp .env.example .env
#
# 2. 실제 값으로 수정:
#    DB_USERNAME, DB_PASSWORD, DB_DSN 등을 실제 값으로 변경
#
# 3. 환경 변수 로드 (Linux/macOS):
#    source .env
#    또는
#    export $(cat .env | xargs)
#
# 4. 환경 변수 로드 (Windows):
#    각 라인을 set 명령으로 실행
#
# 주의: .env 파일은 보안상 git에 커밋하지 마세요!
"""

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(env_content)

            logger.info(f"샘플 환경 변수 파일 생성: {output_path}")
            return True

        except Exception as e:
            logger.error(f"환경 변수 파일 생성 실패: {e}")
            return False

    def validate_security_settings(self, secure_config: SecureDBConfig) -> Dict[str, Any]:
        """보안 설정 검증"""
        validation_result = {
            'secure': True,
            'warnings': [],
            'recommendations': []
        }

        # 1. 환경 변수 사용 여부 확인
        if not secure_config.use_environment:
            validation_result['warnings'].append("환경 변수를 사용하지 않습니다")
            validation_result['recommendations'].append("보안을 위해 환경 변수 사용을 권장합니다")

        # 2. 패스워드 복잡성 확인 (기본적인 체크)
        password = secure_config.password
        if len(password) < 8:
            validation_result['warnings'].append("패스워드가 너무 짧습니다 (8자 미만)")
            validation_result['secure'] = False

        if password.lower() in ['password', '123456', 'admin', 'oracle']:
            validation_result['warnings'].append("약한 패스워드가 감지되었습니다")
            validation_result['secure'] = False

        # 3. DSN 보안 확인
        if 'password' in secure_config.dsn.lower():
            validation_result['warnings'].append("DSN에 패스워드가 포함되어 있습니다")
            validation_result['secure'] = False

        # 4. 추천사항
        validation_result['recommendations'].extend([
            "패스워드는 최소 8자 이상, 대소문자/숫자/특수문자 포함",
            "환경 변수 또는 보안 저장소 사용",
            ".env 파일을 .gitignore에 추가",
            "정기적인 패스워드 변경"
        ])

        return validation_result


class SecureDatabaseConfigManager:
    """통합 보안 데이터베이스 설정 관리자"""

    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/db_config.json"
        self.loader = SecureDatabaseConfigLoader()
        self._db_manager = None
        self._secure_config = None

    def initialize(self, prefer_env: bool = True) -> DatabaseManager:
        """데이터베이스 매니저 초기화"""
        try:
            # 보안 설정 로드
            self._secure_config = self.loader.load_secure_config(
                config_file=self.config_file,
                prefer_env=prefer_env
            )

            # 보안 검증
            security_validation = self.loader.validate_security_settings(self._secure_config)
            if security_validation['warnings']:
                for warning in security_validation['warnings']:
                    logger.warning(f"보안 경고: {warning}")

            # 데이터베이스 매니저 생성
            self._db_manager = self.loader.create_database_manager(self._secure_config)

            return self._db_manager

        except Exception as e:
            logger.error(f"데이터베이스 초기화 실패: {e}")
            raise

    def get_config_info(self) -> Dict[str, Any]:
        """설정 정보 반환 (민감 정보 제외)"""
        if not self._secure_config:
            return {}

        return {
            'username': self._secure_config.username,
            'dsn': self._secure_config.dsn,
            'pool_min': self._secure_config.pool_min,
            'pool_max': self._secure_config.pool_max,
            'oracle_mode': self._secure_config.oracle_mode,
            'use_environment': self._secure_config.use_environment,
            'config_source': self._secure_config.config_source,
            'password_length': len(self._secure_config.password) if self._secure_config.password else 0
        }

    def test_connection(self) -> bool:
        """연결 테스트"""
        if not self._db_manager:
            return False
        return self._db_manager.db_config.test_connection()

    def cleanup(self):
        """리소스 정리"""
        if self._db_manager:
            try:
                self._db_manager.db_config.close_pool()
                logger.info("데이터베이스 연결 풀 정리 완료")
            except Exception as e:
                logger.error(f"연결 풀 정리 중 오류: {e}")


# 편의 함수들
def load_secure_database_config(config_file: Optional[str] = None,
                                prefer_env: bool = True) -> DatabaseManager:
    """
    보안 강화된 데이터베이스 설정 로드 및 매니저 생성

    Args:
        config_file: 설정 파일 경로 (기본: config/db_config.json)
        prefer_env: 환경 변수 우선 사용 여부

    Returns:
        DatabaseManager: 설정된 데이터베이스 매니저
    """
    manager = SecureDatabaseConfigManager(config_file)
    return manager.initialize(prefer_env)


def create_sample_env_file(output_path: str = ".env.example") -> bool:
    """샘플 환경 변수 파일 생성"""
    loader = SecureDatabaseConfigLoader()
    return loader.generate_sample_env_file(output_path)


def validate_database_security(config_file: Optional[str] = None) -> Dict[str, Any]:
    """데이터베이스 보안 설정 검증"""
    try:
        loader = SecureDatabaseConfigLoader()
        secure_config = loader.load_secure_config(config_file)
        return loader.validate_security_settings(secure_config)
    except Exception as e:
        return {
            'secure': False,
            'warnings': [f"설정 로드 실패: {e}"],
            'recommendations': ["설정 파일 및 환경 변수 확인"]
        }


def get_database_config_status(config_file: Optional[str] = None) -> Dict[str, Any]:
    """데이터베이스 설정 상태 확인"""
    status = {
        'config_file_exists': False,
        'env_vars_available': False,
        'connection_test': False,
        'oracle_compatibility': None,
        'recommendations': []
    }

    # 설정 파일 확인
    if config_file and os.path.exists(config_file):
        status['config_file_exists'] = True

    # 환경 변수 확인
    env_vars = ['DB_USERNAME', 'DB_PASSWORD', 'DB_DSN']
    if any(os.getenv(var) for var in env_vars):
        status['env_vars_available'] = True

    # 연결 테스트
    try:
        loader = SecureDatabaseConfigLoader()
        secure_config = loader.load_secure_config(config_file)
        db_manager = loader.create_database_manager(secure_config)
        status['connection_test'] = db_manager.db_config.test_connection()

        # Oracle 호환성 정보
        from core.oracle_compatibility import get_oracle_compatibility_report
        status['oracle_compatibility'] = get_oracle_compatibility_report()

    except Exception as e:
        status['connection_error'] = str(e)

    # 권장사항
    if not status['env_vars_available']:
        status['recommendations'].append("보안을 위해 환경 변수 사용 권장")

    if not status['connection_test']:
        status['recommendations'].append("데이터베이스 연결 설정 확인 필요")

    return status


# 테스트 및 진단 도구
def run_database_diagnostics(config_file: Optional[str] = None) -> str:
    """데이터베이스 진단 실행 및 리포트 생성"""
    lines = []
    lines.append("=" * 70)
    lines.append("데이터베이스 설정 진단 리포트")
    lines.append("=" * 70)

    # 1. 설정 상태 확인
    status = get_database_config_status(config_file)
    lines.append("📋 설정 상태:")
    lines.append(f"   설정 파일: {'✅ 존재' if status['config_file_exists'] else '❌ 없음'}")
    lines.append(f"   환경 변수: {'✅ 사용' if status['env_vars_available'] else '❌ 미사용'}")
    lines.append(f"   연결 테스트: {'✅ 성공' if status['connection_test'] else '❌ 실패'}")

    # 2. 보안 검증
    security = validate_database_security(config_file)
    lines.append(f"\n🔒 보안 상태: {'✅ 안전' if security['secure'] else '⚠️ 주의 필요'}")
    if security['warnings']:
        lines.append("   경고사항:")
        for warning in security['warnings'][:3]:
            lines.append(f"     • {warning}")

    # 3. Oracle 호환성
    if status['oracle_compatibility']:
        lines.append(f"\n🔗 Oracle 호환성:")
        lines.append("   " + status['oracle_compatibility'].replace('\n', '\n   '))

    # 4. 권장사항
    all_recommendations = status.get('recommendations', []) + security.get('recommendations', [])
    if all_recommendations:
        lines.append(f"\n💡 권장사항:")
        for rec in list(set(all_recommendations))[:5]:  # 중복 제거, 최대 5개
            lines.append(f"   • {rec}")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)