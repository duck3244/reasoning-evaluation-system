"""
Oracle 데이터베이스 호환성 관리 모듈
다양한 Oracle 버전과의 호환성을 보장하고 최적의 연결 모드를 자동 선택합니다.
"""
import oracledb
import os
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OracleMode(Enum):
    """Oracle 연결 모드"""
    THIN = "thin"
    THICK = "thick"
    AUTO = "auto"


@dataclass
class OracleCompatibilityInfo:
    """Oracle 호환성 정보"""
    mode: OracleMode
    version: Optional[str] = None
    supported_features: Dict[str, bool] = None
    recommendations: list = None
    warnings: list = None

    def __post_init__(self):
        if self.supported_features is None:
            self.supported_features = {}
        if self.recommendations is None:
            self.recommendations = []
        if self.warnings is None:
            self.warnings = []


class OracleCompatibilityManager:
    """Oracle 호환성 관리자"""

    def __init__(self):
        self._mode_initialized = False
        self._current_mode = None
        self._compatibility_info = None

    def auto_configure_oracle_mode(self,
                                   username: str,
                                   password: str,
                                   dsn: str,
                                   preferred_mode: OracleMode = OracleMode.THIN) -> OracleCompatibilityInfo:
        """
        Oracle 버전에 따른 자동 모드 설정

        Args:
            username: 데이터베이스 사용자명
            password: 패스워드
            dsn: 데이터베이스 DSN
            preferred_mode: 선호하는 연결 모드

        Returns:
            OracleCompatibilityInfo: 호환성 정보
        """
        logger.info("Oracle 호환성 자동 설정 시작...")

        compatibility_info = OracleCompatibilityInfo(mode=preferred_mode)

        try:
            # 1. Thin 모드 시도 (기본값)
            if preferred_mode in [OracleMode.THIN, OracleMode.AUTO]:
                try:
                    logger.info("Thin 모드 연결 시도...")
                    thin_info = self._test_thin_mode(username, password, dsn)
                    if thin_info:
                        compatibility_info = thin_info
                        self._current_mode = OracleMode.THIN
                        logger.info("✅ Thin 모드 연결 성공")
                        return compatibility_info
                except Exception as e:
                    logger.warning(f"Thin 모드 연결 실패: {e}")
                    compatibility_info.warnings.append(f"Thin 모드 실패: {str(e)}")

            # 2. Thick 모드 시도
            if preferred_mode in [OracleMode.THICK, OracleMode.AUTO]:
                try:
                    logger.info("Thick 모드 연결 시도...")
                    thick_info = self._test_thick_mode(username, password, dsn)
                    if thick_info:
                        compatibility_info = thick_info
                        self._current_mode = OracleMode.THICK
                        logger.info("✅ Thick 모드 연결 성공")
                        return compatibility_info
                except Exception as e:
                    logger.error(f"Thick 모드 연결 실패: {e}")
                    compatibility_info.warnings.append(f"Thick 모드 실패: {str(e)}")

            # 3. 모든 모드 실패
            logger.error("❌ 모든 Oracle 연결 모드 실패")
            compatibility_info.mode = None
            compatibility_info.warnings.append("모든 연결 모드가 실패했습니다.")

        except Exception as e:
            logger.error(f"Oracle 호환성 설정 중 예상치 못한 오류: {e}")
            compatibility_info.warnings.append(f"예상치 못한 오류: {str(e)}")

        self._compatibility_info = compatibility_info
        return compatibility_info

    def _test_thin_mode(self, username: str, password: str, dsn: str) -> Optional[OracleCompatibilityInfo]:
        """Thin 모드 테스트"""
        try:
            # Thin 모드 명시적 활성화
            if not self._mode_initialized:
                # 기존 Thick 모드 초기화 방지
                oracledb.defaults.config_dir = None

            conn = oracledb.connect(
                user=username,
                password=password,
                dsn=dsn,
                disable_oob=True  # Out-of-band break 비활성화
            )

            # 데이터베이스 버전 확인
            version_info = self._get_database_version(conn)

            # 지원 기능 확인
            supported_features = self._check_thin_mode_features(conn)

            conn.close()

            compatibility_info = OracleCompatibilityInfo(
                mode=OracleMode.THIN,
                version=version_info['version'],
                supported_features=supported_features
            )

            # 권장사항 추가
            if version_info['major_version'] < 12:
                compatibility_info.warnings.append(
                    f"Oracle {version_info['version']}는 Thin 모드에서 제한적 지원. "
                    "Thick 모드 사용을 권장합니다."
                )
            else:
                compatibility_info.recommendations.append(
                    "Thin 모드가 최적입니다. 추가 클라이언트 라이브러리가 필요하지 않습니다."
                )

            return compatibility_info

        except Exception as e:
            logger.debug(f"Thin 모드 테스트 실패: {e}")
            return None

    def _test_thick_mode(self, username: str, password: str, dsn: str) -> Optional[OracleCompatibilityInfo]:
        """Thick 모드 테스트"""
        try:
            # Oracle Client 라이브러리 초기화
            if not self._mode_initialized:
                self._initialize_thick_mode()

            conn = oracledb.connect(
                user=username,
                password=password,
                dsn=dsn
            )

            # 데이터베이스 버전 확인
            version_info = self._get_database_version(conn)

            # 지원 기능 확인
            supported_features = self._check_thick_mode_features(conn)

            conn.close()

            compatibility_info = OracleCompatibilityInfo(
                mode=OracleMode.THICK,
                version=version_info['version'],
                supported_features=supported_features
            )

            # 권장사항 추가
            compatibility_info.recommendations.append(
                "Thick 모드로 모든 Oracle 기능을 사용할 수 있습니다."
            )

            if version_info['major_version'] < 12:
                compatibility_info.recommendations.append(
                    f"Oracle {version_info['version']}에는 Thick 모드가 권장됩니다."
                )

            return compatibility_info

        except Exception as e:
            logger.debug(f"Thick 모드 테스트 실패: {e}")
            return None

    def _initialize_thick_mode(self):
        """Thick 모드 초기화"""
        try:
            # 환경 변수에서 Oracle Client 경로 확인
            oracle_home = os.getenv('ORACLE_HOME')
            instant_client_dir = os.getenv('ORACLE_INSTANT_CLIENT')

            init_params = {
                'driver_name': 'LLM-Reasoning-Evaluation : 1.0'
            }

            if instant_client_dir:
                init_params['lib_dir'] = instant_client_dir
                logger.info(f"Oracle Instant Client 경로 설정: {instant_client_dir}")
            elif oracle_home:
                logger.info(f"ORACLE_HOME 감지: {oracle_home}")

            oracledb.init_oracle_client(**init_params)
            self._mode_initialized = True
            logger.info("Oracle Thick 모드 초기화 완료")

        except Exception as e:
            logger.error(f"Oracle Thick 모드 초기화 실패: {e}")
            raise

    def _get_database_version(self, conn) -> Dict[str, Any]:
        """데이터베이스 버전 정보 조회"""
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT BANNER FROM V$VERSION WHERE ROWNUM = 1")
            banner = cursor.fetchone()[0]

            # 버전 파싱
            import re
            version_match = re.search(r'(\d+)\.(\d+)\.(\d+)', banner)
            if version_match:
                major, minor, patch = map(int, version_match.groups())
                version = f"{major}.{minor}.{patch}"
                major_version = major
            else:
                version = "Unknown"
                major_version = 0

            cursor.close()

            return {
                'banner': banner,
                'version': version,
                'major_version': major_version
            }

        except Exception as e:
            logger.warning(f"데이터베이스 버전 조회 실패: {e}")
            return {
                'banner': 'Unknown',
                'version': 'Unknown',
                'major_version': 0
            }

    def _check_thin_mode_features(self, conn) -> Dict[str, bool]:
        """Thin 모드 지원 기능 확인"""
        features = {
            'basic_sql': True,  # 기본 SQL은 항상 지원
            'plsql': True,  # PL/SQL도 기본 지원
            'json_support': False,
            'advanced_queuing': False,
            'native_encryption': False,
            'connection_pooling': True
        }

        try:
            cursor = conn.cursor()

            # JSON 지원 확인 (Oracle 12c+)
            try:
                cursor.execute("SELECT JSON_OBJECT('test' VALUE 'value') FROM DUAL")
                cursor.fetchone()
                features['json_support'] = True
            except:
                pass

            cursor.close()

        except Exception as e:
            logger.debug(f"기능 확인 중 오류: {e}")

        return features

    def _check_thick_mode_features(self, conn) -> Dict[str, bool]:
        """Thick 모드 지원 기능 확인"""
        features = {
            'basic_sql': True,
            'plsql': True,
            'json_support': True,
            'advanced_queuing': True,
            'native_encryption': True,
            'connection_pooling': True,
            'client_result_caching': True,
            'statement_caching': True
        }

        # Thick 모드는 거의 모든 기능을 지원
        return features

    def get_connection_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """현재 모드에 최적화된 연결 파라미터 반환"""
        params = base_params.copy()

        if self._current_mode == OracleMode.THIN:
            # Thin 모드 최적화
            params.setdefault('disable_oob', True)
            # 네트워크 최적화
            params.setdefault('tcp_connect_timeout', 30)

        elif self._current_mode == OracleMode.THICK:
            # Thick 모드 최적화
            params.setdefault('threaded', True)

        return params

    def get_compatibility_report(self) -> str:
        """호환성 리포트 생성"""
        if not self._compatibility_info:
            return "호환성 정보가 없습니다. auto_configure_oracle_mode()를 먼저 실행하세요."

        info = self._compatibility_info
        report_lines = []

        report_lines.append("=" * 60)
        report_lines.append("Oracle 데이터베이스 호환성 리포트")
        report_lines.append("=" * 60)

        if info.mode:
            report_lines.append(f"🔗 연결 모드: {info.mode.value.upper()}")
        else:
            report_lines.append("❌ 연결 실패")

        if info.version:
            report_lines.append(f"📊 데이터베이스 버전: {info.version}")

        if info.supported_features:
            report_lines.append("\n✅ 지원 기능:")
            for feature, supported in info.supported_features.items():
                status = "✓" if supported else "✗"
                report_lines.append(f"   {status} {feature}")

        if info.recommendations:
            report_lines.append(f"\n💡 권장사항:")
            for rec in info.recommendations:
                report_lines.append(f"   • {rec}")

        if info.warnings:
            report_lines.append(f"\n⚠️ 주의사항:")
            for warning in info.warnings:
                report_lines.append(f"   • {warning}")

        report_lines.append("=" * 60)

        return "\n".join(report_lines)

    def create_optimized_config(self, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """최적화된 설정 생성"""
        if not self._compatibility_info:
            return base_config

        optimized_config = base_config.copy()

        # 연결 풀 설정 최적화
        if self._current_mode == OracleMode.THIN:
            # Thin 모드용 최적화
            optimized_config.setdefault('pool_min', 1)
            optimized_config.setdefault('pool_max', 5)
            optimized_config.setdefault('pool_increment', 1)
        else:
            # Thick 모드용 최적화
            optimized_config.setdefault('pool_min', 2)
            optimized_config.setdefault('pool_max', 10)
            optimized_config.setdefault('pool_increment', 2)

        return optimized_config


# 글로벌 인스턴스
oracle_compatibility = OracleCompatibilityManager()


def configure_oracle_compatibility(username: str,
                                   password: str,
                                   dsn: str,
                                   preferred_mode: str = "auto") -> OracleCompatibilityInfo:
    """
    Oracle 호환성 설정 편의 함수

    Args:
        username: 데이터베이스 사용자명
        password: 패스워드
        dsn: 데이터베이스 DSN
        preferred_mode: 선호 모드 ("thin", "thick", "auto")

    Returns:
        OracleCompatibilityInfo: 호환성 정보
    """
    mode_map = {
        "thin": OracleMode.THIN,
        "thick": OracleMode.THICK,
        "auto": OracleMode.AUTO
    }

    preferred = mode_map.get(preferred_mode.lower(), OracleMode.AUTO)
    return oracle_compatibility.auto_configure_oracle_mode(username, password, dsn, preferred)


def get_oracle_compatibility_report() -> str:
    """현재 Oracle 호환성 리포트 반환"""
    return oracle_compatibility.get_compatibility_report()


def get_optimized_connection_params(base_params: Dict[str, Any]) -> Dict[str, Any]:
    """최적화된 연결 파라미터 반환"""
    return oracle_compatibility.get_connection_params(base_params)