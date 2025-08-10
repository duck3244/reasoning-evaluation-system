"""
향상된 에러 처리 모듈
Oracle 및 시스템 에러를 사용자 친화적 메시지로 변환하고 해결책을 제공합니다.
"""
import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import oracledb

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """에러 심각도"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """에러 정보 클래스"""
    original_error: str
    error_code: Optional[str]
    severity: ErrorSeverity
    user_message: str
    technical_details: str
    solutions: List[str]
    documentation_links: List[str]

    def __post_init__(self):
        if not self.solutions:
            self.solutions = []
        if not self.documentation_links:
            self.documentation_links = []


class EnhancedErrorHandler:
    """향상된 에러 처리기"""

    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.solution_templates = self._initialize_solution_templates()

    def _initialize_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """에러 패턴 초기화"""
        return {
            # Oracle Thin 모드 에러
            'DPY-3010': {
                'severity': ErrorSeverity.ERROR,
                'message': 'Oracle Database 버전이 너무 낮습니다',
                'details': 'Thin 모드는 Oracle Database 12.1 이상에서만 지원됩니다',
                'solutions': [
                    'Oracle Database를 12.1 이상으로 업그레이드',
                    'Thick 모드 사용 (Oracle Client 라이브러리 설치 필요)',
                    'oracledb.init_oracle_client() 호출하여 Thick 모드 활성화'
                ],
                'docs': ['https://python-oracledb.readthedocs.io/en/latest/user_guide/initialization.html']
            },

            'DPY-3015': {
                'severity': ErrorSeverity.ERROR,
                'message': '패스워드 인증 방식이 호환되지 않습니다',
                'details': 'Oracle Database 10G 패스워드 검증자는 Thin 모드에서 지원되지 않습니다',
                'solutions': [
                    '데이터베이스 관리자에게 패스워드 재설정 요청',
                    'ALTER USER 명령으로 패스워드 업데이트',
                    'Thick 모드 사용 (10G 패스워드 검증자 지원)',
                    'sec_case_sensitive_logon 파라미터 확인'
                ],
                'docs': ['https://python-oracledb.readthedocs.io/en/latest/user_guide/troubleshooting.html#dpy-3015']
            },

            'DPY-3001': {
                'severity': ErrorSeverity.ERROR,
                'message': 'Native Network Encryption은 Thick 모드에서만 지원됩니다',
                'details': '데이터베이스에 NNE(Native Network Encryption)가 활성화되어 있습니다',
                'solutions': [
                    'Thick 모드로 전환 (oracledb.init_oracle_client() 호출)',
                    '데이터베이스에서 NNE 설정 비활성화',
                    'Oracle Client 라이브러리 설치 후 Thick 모드 사용'
                ],
                'docs': ['https://python-oracledb.readthedocs.io/en/latest/user_guide/troubleshooting.html#dpy-3001']
            },

            'DPY-4011': {
                'severity': ErrorSeverity.ERROR,
                'message': '데이터베이스 연결이 끊어졌습니다',
                'details': '네트워크 또는 데이터베이스 서버에서 연결을 종료했습니다',
                'solutions': [
                    '네트워크 연결 상태 확인',
                    '데이터베이스 서버 상태 확인',
                    'disable_oob=True 옵션 사용',
                    '방화벽 설정 확인',
                    'TNS_ADMIN 환경변수 설정'
                ],
                'docs': ['https://python-oracledb.readthedocs.io/en/latest/user_guide/troubleshooting.html#dpy-4011']
            },

            'DPY-4027': {
                'severity': ErrorSeverity.ERROR,
                'message': 'TNS 설정 디렉토리를 찾을 수 없습니다',
                'details': 'tnsnames.ora 파일을 찾기 위한 설정 디렉토리가 지정되지 않았습니다',
                'solutions': [
                    'TNS_ADMIN 환경변수 설정',
                    'oracledb.defaults.config_dir 설정',
                    'Easy Connect 문자열 사용 (host:port/service_name)',
                    'tnsnames.ora 파일 경로 확인'
                ],
                'docs': ['https://python-oracledb.readthedocs.io/en/latest/user_guide/troubleshooting.html#dpy-4027']
            },

            'DPI-1047': {
                'severity': ErrorSeverity.CRITICAL,
                'message': 'Oracle Client 라이브러리를 찾을 수 없습니다',
                'details': 'Thick 모드용 Oracle Client 라이브러리가 설치되지 않았거나 경로에 없습니다',
                'solutions': [
                    'Oracle Instant Client 다운로드 및 설치',
                    'ORACLE_HOME 환경변수 설정',
                    'LD_LIBRARY_PATH (Linux) 또는 PATH (Windows) 설정',
                    'oracledb.init_oracle_client(lib_dir="경로") 사용'
                ],
                'docs': ['https://python-oracledb.readthedocs.io/en/latest/user_guide/installation.html']
            },

            'DPI-1072': {
                'severity': ErrorSeverity.ERROR,
                'message': 'Oracle Client 라이브러리 버전이 지원되지 않습니다',
                'details': 'Thick 모드는 Oracle Client 11.2 이상이 필요합니다',
                'solutions': [
                    'Oracle Client를 11.2 이상으로 업그레이드',
                    'Thin 모드 사용 (Oracle Database 12.1+ 필요)',
                    'init_oracle_client() 호출 제거하여 Thin 모드로 전환'
                ],
                'docs': ['https://python-oracledb.readthedocs.io/en/latest/user_guide/installation.html']
            },

            # 일반적인 연결 에러
            'TNS': {
                'severity': ErrorSeverity.ERROR,
                'message': 'TNS 연결 오류가 발생했습니다',
                'details': '데이터베이스 주소(DSN) 설정에 문제가 있습니다',
                'solutions': [
                    'DSN 형식 확인 (host:port/service_name)',
                    'tnsnames.ora 파일 내용 확인',
                    '네트워크 연결 테스트 (ping, telnet)',
                    '포트 번호 확인 (기본값: 1521)',
                    'service_name 또는 SID 확인'
                ],
                'docs': ['https://docs.oracle.com/en/database/oracle/oracle-database/']
            },

            'ORA-00942': {
                'severity': ErrorSeverity.ERROR,
                'message': '테이블 또는 뷰가 존재하지 않습니다',
                'details': '참조하려는 테이블이나 뷰에 대한 권한이 없거나 존재하지 않습니다',
                'solutions': [
                    '테이블명 철자 확인',
                    '스키마명 포함하여 테이블 참조',
                    '테이블 생성 여부 확인',
                    'SELECT 권한 확인',
                    'init_database.py 스크립트 실행'
                ],
                'docs': []
            },

            'ORA-01017': {
                'severity': ErrorSeverity.ERROR,
                'message': '잘못된 사용자명/패스워드입니다',
                'details': '인증 정보가 올바르지 않습니다',
                'solutions': [
                    '사용자명과 패스워드 확인',
                    '대소문자 구분 확인',
                    '계정 잠금 상태 확인',
                    '패스워드 만료 여부 확인'
                ],
                'docs': []
            },

            'ORA-12541': {
                'severity': ErrorSeverity.ERROR,
                'message': 'TNS 리스너가 없습니다',
                'details': '데이터베이스 리스너가 실행되지 않거나 연결할 수 없습니다',
                'solutions': [
                    '데이터베이스 서버 상태 확인',
                    '리스너 서비스 시작',
                    '호스트명과 포트 확인',
                    '방화벽 설정 확인'
                ],
                'docs': []
            },

            # 시스템 에러
            'ConnectionRefusedError': {
                'severity': ErrorSeverity.ERROR,
                'message': '연결이 거부되었습니다',
                'details': '대상 서버가 연결을 거부했습니다',
                'solutions': [
                    '서버 주소와 포트 확인',
                    '서버 실행 상태 확인',
                    '방화벽 설정 확인',
                    'telnet으로 포트 연결 테스트'
                ],
                'docs': []
            },

            'TimeoutError': {
                'severity': ErrorSeverity.WARNING,
                'message': '연결 시간이 초과되었습니다',
                'details': '지정된 시간 내에 연결을 완료할 수 없습니다',
                'solutions': [
                    '네트워크 상태 확인',
                    'tcp_connect_timeout 증가',
                    'VPN 연결 확인',
                    '서버 부하 상태 확인'
                ],
                'docs': []
            },

            'MemoryError': {
                'severity': ErrorSeverity.CRITICAL,
                'message': '메모리가 부족합니다',
                'details': '사용 가능한 메모리가 충분하지 않습니다',
                'solutions': [
                    '배치 크기 감소',
                    '메모리 정리 함수 호출',
                    '시스템 메모리 확인',
                    '불필요한 프로세스 종료'
                ],
                'docs': []
            }
        }

    def _initialize_solution_templates(self) -> Dict[str, str]:
        """해결책 템플릿 초기화"""
        return {
            'oracle_client_install': """
Oracle Instant Client 설치 방법:

1. Oracle 공식 사이트에서 Instant Client 다운로드
   https://www.oracle.com/database/technologies/instant-client.html

2. 적절한 버전 선택 (Linux/Windows/macOS)

3. 설치 및 환경변수 설정:
   Linux: export LD_LIBRARY_PATH=/path/to/instantclient
   Windows: PATH에 instant client 경로 추가
   macOS: export DYLD_LIBRARY_PATH=/path/to/instantclient

4. 또는 Python 코드에서 직접 지정:
   oracledb.init_oracle_client(lib_dir="/path/to/instantclient")
""",

            'thick_mode_setup': """
Thick 모드 설정 방법:

1. Oracle Client 라이브러리 설치 (위 참조)

2. Python 코드에서 Thick 모드 활성화:
   import oracledb
   oracledb.init_oracle_client()

3. 연결 생성:
   conn = oracledb.connect(user=user, password=pwd, dsn=dsn)
""",

            'database_setup': """
데이터베이스 설정 확인:

1. 데이터베이스 초기화:
   python scripts/init_database.py

2. 연결 테스트:
   python -c "from core.database_config import load_db_config_from_file; 
   config = load_db_config_from_file('config/db_config.json'); 
   print('연결 성공' if config.test_connection() else '연결 실패')"

3. 설정 파일 확인:
   config/db_config.json 파일의 username, password, dsn 확인
"""
        }

    def handle_error(self, error: Exception, context: str = "") -> ErrorInfo:
        """
        에러를 분석하고 사용자 친화적 정보 반환

        Args:
            error: 발생한 예외
            context: 에러 발생 컨텍스트

        Returns:
            ErrorInfo: 처리된 에러 정보
        """
        error_str = str(error)
        error_type = type(error).__name__

        logger.debug(f"에러 처리 중: {error_type} - {error_str}")

        # 에러 코드 추출
        error_code = self._extract_error_code(error_str)

        # 패턴 매칭으로 에러 정보 찾기
        error_info = self._match_error_pattern(error_str, error_code, error_type)

        # 컨텍스트 정보 추가
        if context:
            error_info.technical_details += f"\n컨텍스트: {context}"

        return error_info

    def _extract_error_code(self, error_str: str) -> Optional[str]:
        """에러 문자열에서 에러 코드 추출"""
        patterns = [
            r'(DPY-\d+)',  # python-oracledb 에러
            r'(DPI-\d+)',  # ODPI-C 에러
            r'(ORA-\d+)',  # Oracle 에러
        ]

        for pattern in patterns:
            match = re.search(pattern, error_str)
            if match:
                return match.group(1)

        return None

    def _match_error_pattern(self, error_str: str, error_code: Optional[str], error_type: str) -> ErrorInfo:
        """에러 패턴 매칭"""

        # 1. 에러 코드로 매칭
        if error_code and error_code in self.error_patterns:
            pattern = self.error_patterns[error_code]
            return self._create_error_info(error_str, error_code, pattern)

        # 2. 에러 타입으로 매칭
        if error_type in self.error_patterns:
            pattern = self.error_patterns[error_type]
            return self._create_error_info(error_str, error_code, pattern)

        # 3. 부분 문자열 매칭
        for key, pattern in self.error_patterns.items():
            if key.upper() in error_str.upper():
                return self._create_error_info(error_str, error_code, pattern)

        # 4. 기본 에러 정보
        return self._create_default_error_info(error_str, error_code, error_type)

    def _create_error_info(self, error_str: str, error_code: Optional[str], pattern: Dict[str, Any]) -> ErrorInfo:
        """패턴으로부터 ErrorInfo 생성"""
        return ErrorInfo(
            original_error=error_str,
            error_code=error_code,
            severity=pattern['severity'],
            user_message=pattern['message'],
            technical_details=pattern['details'],
            solutions=pattern['solutions'].copy(),
            documentation_links=pattern['docs'].copy()
        )

    def _create_default_error_info(self, error_str: str, error_code: Optional[str], error_type: str) -> ErrorInfo:
        """기본 에러 정보 생성"""
        return ErrorInfo(
            original_error=error_str,
            error_code=error_code,
            severity=ErrorSeverity.ERROR,
            user_message=f"{error_type} 오류가 발생했습니다",
            technical_details=error_str,
            solutions=[
                "에러 메시지의 상세 내용을 확인하세요",
                "설정 파일과 네트워크 상태를 점검하세요",
                "로그 파일에서 추가 정보를 확인하세요"
            ],
            documentation_links=[]
        )

    def format_error_message(self, error_info: ErrorInfo, include_solutions: bool = True) -> str:
        """에러 정보를 포맷된 메시지로 변환"""
        lines = []

        # 헤더
        severity_emoji = {
            ErrorSeverity.INFO: "ℹ️",
            ErrorSeverity.WARNING: "⚠️",
            ErrorSeverity.ERROR: "❌",
            ErrorSeverity.CRITICAL: "🚨"
        }

        emoji = severity_emoji.get(error_info.severity, "❓")
        lines.append(f"{emoji} {error_info.user_message}")

        if error_info.error_code:
            lines.append(f"   코드: {error_info.error_code}")

        # 기술적 세부사항
        if error_info.technical_details:
            lines.append(f"\n📋 상세 정보:")
            lines.append(f"   {error_info.technical_details}")

        # 해결책
        if include_solutions and error_info.solutions:
            lines.append(f"\n💡 해결 방법:")
            for i, solution in enumerate(error_info.solutions[:3], 1):  # 최대 3개만 표시
                lines.append(f"   {i}. {solution}")

        # 문서 링크
        if error_info.documentation_links:
            lines.append(f"\n📖 관련 문서:")
            for link in error_info.documentation_links[:2]:  # 최대 2개만 표시
                lines.append(f"   • {link}")

        return "\n".join(lines)

    def get_troubleshooting_guide(self, error_info: ErrorInfo) -> str:
        """트러블슈팅 가이드 생성"""
        guide_lines = []

        guide_lines.append("=" * 60)
        guide_lines.append("🔧 상세 트러블슈팅 가이드")
        guide_lines.append("=" * 60)

        # 에러 요약
        guide_lines.append(f"에러: {error_info.user_message}")
        if error_info.error_code:
            guide_lines.append(f"코드: {error_info.error_code}")
        guide_lines.append("")

        # 단계별 해결 방법
        guide_lines.append("📝 단계별 해결 방법:")
        for i, solution in enumerate(error_info.solutions, 1):
            guide_lines.append(f"\n{i}. {solution}")

            # 특정 해결책에 대한 상세 가이드 추가
            if "thick 모드" in solution.lower():
                guide_lines.append(self.solution_templates['thick_mode_setup'])
            elif "oracle client" in solution.lower():
                guide_lines.append(self.solution_templates['oracle_client_install'])
            elif "데이터베이스 초기화" in solution.lower():
                guide_lines.append(self.solution_templates['database_setup'])

        # 추가 정보
        if error_info.documentation_links:
            guide_lines.append(f"\n📚 참고 문서:")
            for link in error_info.documentation_links:
                guide_lines.append(f"• {link}")

        guide_lines.append("\n" + "=" * 60)

        return "\n".join(guide_lines)

    def log_error(self, error_info: ErrorInfo, context: str = ""):
        """구조화된 에러 로깅"""
        log_data = {
            'error_code': error_info.error_code,
            'severity': error_info.severity.value,
            'user_message': error_info.user_message,
            'technical_details': error_info.technical_details,
            'context': context,
            'solutions_count': len(error_info.solutions)
        }

        log_level = {
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }

        level = log_level.get(error_info.severity, logging.ERROR)
        logger.log(level, error_info.user_message, extra=log_data)


# 글로벌 에러 핸들러 인스턴스
error_handler = EnhancedErrorHandler()


def handle_database_error(error: Exception, context: str = "") -> ErrorInfo:
    """데이터베이스 에러 처리 편의 함수"""
    return error_handler.handle_error(error, context)


def format_user_error(error: Exception, context: str = "") -> str:
    """사용자용 에러 메시지 포맷팅 편의 함수"""
    error_info = error_handler.handle_error(error, context)
    return error_handler.format_error_message(error_info)


def get_error_solutions(error: Exception) -> List[str]:
    """에러 해결책 목록 반환"""
    error_info = error_handler.handle_error(error)
    return error_info.solutions


def create_troubleshooting_guide(error: Exception, context: str = "") -> str:
    """트러블슈팅 가이드 생성 편의 함수"""
    error_info = error_handler.handle_error(error, context)
    return error_handler.get_troubleshooting_guide(error_info)