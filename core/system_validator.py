"""
시스템 요구사항 검증 도구
프로젝트 실행에 필요한 모든 요구사항을 검증하고 문제점을 진단합니다.
"""
import sys
import os
import platform
import json
import subprocess
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """검증 상태"""
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_APPLICABLE = "n/a"


@dataclass
class ValidationResult:
    """검증 결과"""
    name: str
    status: ValidationStatus
    message: str
    details: str = ""
    recommendations: List[str] = field(default_factory=list)
    required: bool = True


@dataclass
class SystemValidationReport:
    """시스템 검증 리포트"""
    overall_status: ValidationStatus
    results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.summary:
            self.summary = {status.value: 0 for status in ValidationStatus}


class SystemValidator:
    """시스템 요구사항 검증기"""

    def __init__(self):
        self.python_min_version = (3, 8)
        self.python_max_version = (3, 12)
        self.required_memory_gb = 4.0
        self.recommended_memory_gb = 8.0
        self.required_disk_space_gb = 1.0

    def validate_all(self, config_file: Optional[str] = None) -> SystemValidationReport:
        """전체 시스템 검증"""
        logger.info("시스템 요구사항 검증 시작...")

        results = []

        # 1. Python 환경 검증
        results.extend(self._validate_python_environment())

        # 2. 필수 패키지 검증
        results.extend(self._validate_required_packages())

        # 3. 시스템 리소스 검증
        results.extend(self._validate_system_resources())

        # 4. Oracle 환경 검증
        results.extend(self._validate_oracle_environment())

        # 5. 설정 파일 검증
        if config_file:
            results.extend(self._validate_configuration(config_file))

        # 6. 프로젝트 구조 검증
        results.extend(self._validate_project_structure())

        # 7. 권한 및 접근성 검증
        results.extend(self._validate_permissions())

        # 리포트 생성
        report = self._generate_report(results)

        logger.info(f"시스템 검증 완료: {report.overall_status.value}")
        return report

    def _validate_python_environment(self) -> List[ValidationResult]:
        """Python 환경 검증"""
        results = []

        # Python 버전 확인
        current_version = sys.version_info[:2]
        if current_version < self.python_min_version:
            results.append(ValidationResult(
                name="Python 버전",
                status=ValidationStatus.FAIL,
                message=f"Python {'.'.join(map(str, self.python_min_version))} 이상이 필요합니다",
                details=f"현재 버전: {'.'.join(map(str, current_version))}",
                recommendations=[
                    f"Python {'.'.join(map(str, self.python_min_version))} 이상으로 업그레이드",
                    "pyenv 또는 conda를 사용한 Python 버전 관리 고려"
                ]
            ))
        elif current_version > self.python_max_version:
            results.append(ValidationResult(
                name="Python 버전",
                status=ValidationStatus.WARNING,
                message=f"Python {'.'.join(map(str, current_version))}는 테스트되지 않은 버전입니다",
                details=f"권장 버전: {'.'.join(map(str, self.python_min_version))} - {'.'.join(map(str, self.python_max_version))}",
                recommendations=["호환성 문제 발생 시 권장 버전으로 다운그레이드"],
                required=False
            ))
        else:
            results.append(ValidationResult(
                name="Python 버전",
                status=ValidationStatus.PASS,
                message=f"Python {'.'.join(map(str, current_version))} (호환 가능)",
                details=f"지원 범위: {'.'.join(map(str, self.python_min_version))} - {'.'.join(map(str, self.python_max_version))}"
            ))

        # 가상환경 확인
        if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            results.append(ValidationResult(
                name="가상환경",
                status=ValidationStatus.PASS,
                message="가상환경이 활성화되어 있습니다",
                details=f"가상환경 경로: {sys.prefix}",
                required=False
            ))
        else:
            results.append(ValidationResult(
                name="가상환경",
                status=ValidationStatus.WARNING,
                message="가상환경을 사용하지 않습니다",
                details="시스템 Python을 직접 사용 중",
                recommendations=[
                    "venv 또는 conda로 가상환경 생성 권장",
                    "의존성 격리 및 충돌 방지를 위해 가상환경 사용"
                ],
                required=False
            ))

        # pip 버전 확인
        try:
            import pip
            pip_version = pip.__version__
            results.append(ValidationResult(
                name="pip 버전",
                status=ValidationStatus.PASS,
                message=f"pip {pip_version}",
                details="패키지 설치 도구 사용 가능",
                required=False
            ))
        except ImportError:
            results.append(ValidationResult(
                name="pip",
                status=ValidationStatus.FAIL,
                message="pip이 설치되지 않았습니다",
                recommendations=["pip 설치 또는 Python 재설치"]
            ))

        return results

    def _validate_required_packages(self) -> List[ValidationResult]:
        """필수 패키지 검증"""
        results = []

        required_packages = {
            'oracledb': '1.4.0',
            'requests': '2.31.0',
            'psutil': '5.9.0'
        }

        optional_packages = {
            'pandas': '2.0.0',
            'datasets': '2.14.0',
            'matplotlib': '3.6.0'
        }

        # 필수 패키지 확인
        for package, min_version in required_packages.items():
            result = self._check_package(package, min_version, required=True)
            results.append(result)

        # 선택적 패키지 확인
        for package, min_version in optional_packages.items():
            result = self._check_package(package, min_version, required=False)
            results.append(result)

        return results

    def _check_package(self, package_name: str, min_version: str, required: bool = True) -> ValidationResult:
        """개별 패키지 확인"""
        try:
            import importlib
            module = importlib.import_module(package_name)

            # 버전 확인
            version = getattr(module, '__version__', 'Unknown')

            if version != 'Unknown':
                from packaging import version as pkg_version
                if pkg_version.parse(version) >= pkg_version.parse(min_version):
                    status = ValidationStatus.PASS
                    message = f"{package_name} {version} (OK)"
                else:
                    status = ValidationStatus.WARNING if not required else ValidationStatus.FAIL
                    message = f"{package_name} {version} (최소 {min_version} 필요)"
            else:
                status = ValidationStatus.WARNING
                message = f"{package_name} 설치됨 (버전 불명)"

            return ValidationResult(
                name=f"패키지 {package_name}",
                status=status,
                message=message,
                details=f"설치 위치: {module.__file__ if hasattr(module, '__file__') else 'Unknown'}",
                recommendations=[
                    f"pip install {package_name}>={min_version}"] if status != ValidationStatus.PASS else [],
                required=required
            )

        except ImportError:
            status = ValidationStatus.FAIL if required else ValidationStatus.WARNING
            return ValidationResult(
                name=f"패키지 {package_name}",
                status=status,
                message=f"{package_name} 패키지가 설치되지 않았습니다",
                recommendations=[f"pip install {package_name}>={min_version}"],
                required=required
            )

    def _validate_system_resources(self) -> List[ValidationResult]:
        """시스템 리소스 검증"""
        results = []

        try:
            import psutil

            # 메모리 확인
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024 ** 3)
            total_gb = memory.total / (1024 ** 3)

            if available_gb < self.required_memory_gb:
                results.append(ValidationResult(
                    name="메모리",
                    status=ValidationStatus.FAIL,
                    message=f"사용 가능한 메모리가 부족합니다 ({available_gb:.1f}GB)",
                    details=f"전체: {total_gb:.1f}GB, 사용 가능: {available_gb:.1f}GB, 필요: {self.required_memory_gb}GB",
                    recommendations=[
                        "불필요한 프로그램 종료",
                        "시스템 메모리 업그레이드",
                        "배치 크기 감소 설정"
                    ]
                ))
            elif available_gb < self.recommended_memory_gb:
                results.append(ValidationResult(
                    name="메모리",
                    status=ValidationStatus.WARNING,
                    message=f"메모리가 권장 사양보다 적습니다 ({available_gb:.1f}GB)",
                    details=f"권장: {self.recommended_memory_gb}GB, 현재: {available_gb:.1f}GB",
                    recommendations=["대용량 데이터 처리 시 성능 저하 가능"],
                    required=False
                ))
            else:
                results.append(ValidationResult(
                    name="메모리",
                    status=ValidationStatus.PASS,
                    message=f"충분한 메모리 ({available_gb:.1f}GB)",
                    details=f"전체: {total_gb:.1f}GB, 사용 가능: {available_gb:.1f}GB"
                ))

            # 디스크 공간 확인
            disk = psutil.disk_usage('.')
            available_disk_gb = disk.free / (1024 ** 3)

            if available_disk_gb < self.required_disk_space_gb:
                results.append(ValidationResult(
                    name="디스크 공간",
                    status=ValidationStatus.FAIL,
                    message=f"디스크 공간이 부족합니다 ({available_disk_gb:.1f}GB)",
                    details=f"필요: {self.required_disk_space_gb}GB",
                    recommendations=["불필요한 파일 삭제", "로그 파일 정리"]
                ))
            else:
                results.append(ValidationResult(
                    name="디스크 공간",
                    status=ValidationStatus.PASS,
                    message=f"충분한 디스크 공간 ({available_disk_gb:.1f}GB)",
                    details=f"사용 가능: {available_disk_gb:.1f}GB"
                ))

            # CPU 확인
            cpu_count = psutil.cpu_count()
            results.append(ValidationResult(
                name="CPU",
                status=ValidationStatus.PASS,
                message=f"{cpu_count}개 CPU 코어",
                details=f"논리 프로세서: {psutil.cpu_count(logical=True)}개",
                required=False
            ))

        except ImportError:
            results.append(ValidationResult(
                name="시스템 리소스",
                status=ValidationStatus.WARNING,
                message="psutil 패키지가 없어 시스템 리소스를 확인할 수 없습니다",
                recommendations=["pip install psutil"],
                required=False
            ))

        return results

    def _validate_oracle_environment(self) -> List[ValidationResult]:
        """Oracle 환경 검증"""
        results = []

        # Oracle DB 패키지 확인
        try:
            import oracledb
            results.append(ValidationResult(
                name="Oracle DB 드라이버",
                status=ValidationStatus.PASS,
                message=f"python-oracledb {oracledb.__version__}",
                details="Oracle 데이터베이스 연결 가능"
            ))

            # Thin/Thick 모드 지원 확인
            thin_available = True  # Thin 모드는 기본적으로 사용 가능

            # Thick 모드 지원 확인
            thick_available = self._check_oracle_thick_mode()

            mode_details = []
            if thin_available:
                mode_details.append("Thin 모드: 사용 가능 (Oracle 12.1+ 지원)")
            if thick_available:
                mode_details.append("Thick 모드: 사용 가능 (모든 Oracle 버전 지원)")

            results.append(ValidationResult(
                name="Oracle 연결 모드",
                status=ValidationStatus.PASS,
                message="Oracle 연결 모드 지원",
                details="\n".join(mode_details),
                required=False
            ))

        except ImportError:
            results.append(ValidationResult(
                name="Oracle DB 드라이버",
                status=ValidationStatus.FAIL,
                message="python-oracledb 패키지가 설치되지 않았습니다",
                recommendations=["pip install oracledb>=1.4.0"]
            ))

        # Oracle Client 라이브러리 확인 (Thick 모드용)
        oracle_client_paths = self._find_oracle_client_libraries()
        if oracle_client_paths:
            results.append(ValidationResult(
                name="Oracle Client 라이브러리",
                status=ValidationStatus.PASS,
                message="Oracle Client 라이브러리 발견",
                details=f"경로: {', '.join(oracle_client_paths)}",
                required=False
            ))
        else:
            results.append(ValidationResult(
                name="Oracle Client 라이브러리",
                status=ValidationStatus.WARNING,
                message="Oracle Client 라이브러리를 찾을 수 없습니다",
                details="Thick 모드 사용 불가, Thin 모드만 사용 가능",
                recommendations=[
                    "Oracle Instant Client 설치",
                    "ORACLE_HOME 환경변수 설정",
                    "LD_LIBRARY_PATH (Linux) 또는 PATH (Windows) 설정"
                ],
                required=False
            ))

        return results

    def _check_oracle_thick_mode(self) -> bool:
        """Oracle Thick 모드 사용 가능 여부 확인"""
        try:
            import oracledb
            # 임시로 Thick 모드 초기화 시도
            # 실제로는 초기화하지 않고 가능 여부만 확인
            oracle_client_paths = self._find_oracle_client_libraries()
            return len(oracle_client_paths) > 0
        except Exception:
            return False

    def _find_oracle_client_libraries(self) -> List[str]:
        """Oracle Client 라이브러리 경로 찾기"""
        paths = []

        # 환경 변수 확인
        oracle_home = os.getenv('ORACLE_HOME')
        if oracle_home:
            lib_path = os.path.join(oracle_home, 'lib')
            if os.path.exists(lib_path):
                paths.append(lib_path)

        instant_client = os.getenv('ORACLE_INSTANT_CLIENT')
        if instant_client and os.path.exists(instant_client):
            paths.append(instant_client)

        # 일반적인 설치 경로 확인
        common_paths = [
            '/opt/oracle/instantclient',
            '/usr/lib/oracle',
            'C:\\oracle\\instantclient',
            'C:\\app\\oracle\\product'
        ]

        for path in common_paths:
            if os.path.exists(path):
                paths.append(path)

        return paths

    def _validate_configuration(self, config_file: str) -> List[ValidationResult]:
        """설정 파일 검증"""
        results = []

        # 설정 파일 존재 확인
        if not os.path.exists(config_file):
            results.append(ValidationResult(
                name="설정 파일",
                status=ValidationStatus.FAIL,
                message=f"설정 파일을 찾을 수 없습니다: {config_file}",
                recommendations=[
                    f"cp {config_file}.example {config_file}",
                    "python scripts/init_database.py --create-config"
                ]
            ))
            return results

        # 설정 파일 내용 검증
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)

            required_fields = ['username', 'password', 'dsn']
            missing_fields = [field for field in required_fields if not config.get(field)]

            if missing_fields:
                results.append(ValidationResult(
                    name="설정 파일 내용",
                    status=ValidationStatus.FAIL,
                    message=f"필수 필드가 누락되었습니다: {', '.join(missing_fields)}",
                    details=f"파일: {config_file}",
                    recommendations=["설정 파일에 누락된 필드 추가"]
                ))
            else:
                results.append(ValidationResult(
                    name="설정 파일 내용",
                    status=ValidationStatus.PASS,
                    message="설정 파일이 올바르게 구성되었습니다",
                    details=f"파일: {config_file}"
                ))

                # DSN 형식 검증
                dsn = config.get('dsn', '')
                if self._validate_dsn_format(dsn):
                    results.append(ValidationResult(
                        name="DSN 형식",
                        status=ValidationStatus.PASS,
                        message="DSN 형식이 올바릅니다",
                        details=f"DSN: {dsn}",
                        required=False
                    ))
                else:
                    results.append(ValidationResult(
                        name="DSN 형식",
                        status=ValidationStatus.WARNING,
                        message="DSN 형식을 확인하세요",
                        details=f"DSN: {dsn}",
                        recommendations=[
                            "Easy Connect 형식: host:port/service_name",
                            "TNS 별칭 사용 시 tnsnames.ora 파일 확인"
                        ],
                        required=False
                    ))

        except json.JSONDecodeError as e:
            results.append(ValidationResult(
                name="설정 파일 형식",
                status=ValidationStatus.FAIL,
                message=f"JSON 형식 오류: {e}",
                recommendations=["설정 파일의 JSON 문법 확인"]
            ))

        except Exception as e:
            results.append(ValidationResult(
                name="설정 파일",
                status=ValidationStatus.FAIL,
                message=f"설정 파일 읽기 오류: {e}",
                recommendations=["파일 권한 및 인코딩 확인"]
            ))

        return results

    def _validate_dsn_format(self, dsn: str) -> bool:
        """DSN 형식 검증"""
        if not dsn:
            return False

        # Easy Connect 형식: host:port/service_name
        import re
        easy_connect_pattern = r'^[^:]+:\d+/[^/]+'

        # TNS 별칭 형식 (단순 문자열)
        tns_alias_pattern = r'^[a-zA-Z][a-zA-Z0-9_]*'

        return bool(re.match(easy_connect_pattern, dsn) or re.match(tns_alias_pattern, dsn))

    def _validate_project_structure(self) -> List[ValidationResult]:
        """프로젝트 구조 검증"""
        results = []

        required_dirs = [
            'core',
            'config',
            'scripts',
            'evaluation',
            'data_loaders',
            'monitoring'
        ]

        required_files = [
            'main.py',
            'requirements.txt',
            'README.md'
        ]

        # 디렉토리 확인
        missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
        if missing_dirs:
            results.append(ValidationResult(
                name="프로젝트 디렉토리",
                status=ValidationStatus.FAIL,
                message=f"필수 디렉토리가 없습니다: {', '.join(missing_dirs)}",
                recommendations=["프로젝트를 올바른 디렉토리에서 실행하세요"]
            ))
        else:
            results.append(ValidationResult(
                name="프로젝트 디렉토리",
                status=ValidationStatus.PASS,
                message="모든 필수 디렉토리가 존재합니다",
                required=False
            ))

        # 파일 확인
        missing_files = [f for f in required_files if not os.path.isfile(f)]
        if missing_files:
            results.append(ValidationResult(
                name="프로젝트 파일",
                status=ValidationStatus.WARNING,
                message=f"일부 파일이 없습니다: {', '.join(missing_files)}",
                recommendations=["누락된 파일들을 확인하세요"],
                required=False
            ))
        else:
            results.append(ValidationResult(
                name="프로젝트 파일",
                status=ValidationStatus.PASS,
                message="모든 주요 파일이 존재합니다",
                required=False
            ))

        return results

    def _validate_permissions(self) -> List[ValidationResult]:
        """권한 및 접근성 검증"""
        results = []

        # 현재 디렉토리 쓰기 권한
        try:
            test_file = '.write_test'
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)

            results.append(ValidationResult(
                name="디렉토리 쓰기 권한",
                status=ValidationStatus.PASS,
                message="현재 디렉토리에 쓰기 권한이 있습니다",
                required=False
            ))
        except Exception as e:
            results.append(ValidationResult(
                name="디렉토리 쓰기 권한",
                status=ValidationStatus.FAIL,
                message=f"현재 디렉토리에 쓰기 권한이 없습니다: {e}",
                recommendations=["디렉토리 권한 확인", "관리자 권한으로 실행"]
            ))

        # logs 디렉토리 생성/쓰기 권한
        logs_dir = 'logs'
        try:
            os.makedirs(logs_dir, exist_ok=True)
            test_log = os.path.join(logs_dir, 'test.log')
            with open(test_log, 'w') as f:
                f.write('test')
            os.remove(test_log)

            results.append(ValidationResult(
                name="로그 디렉토리 권한",
                status=ValidationStatus.PASS,
                message="로그 디렉토리 생성/쓰기 가능",
                required=False
            ))
        except Exception as e:
            results.append(ValidationResult(
                name="로그 디렉토리 권한",
                status=ValidationStatus.WARNING,
                message=f"로그 디렉토리 권한 문제: {e}",
                recommendations=["logs 디렉토리 권한 확인"],
                required=False
            ))

        return results

    def _generate_report(self, results: List[ValidationResult]) -> SystemValidationReport:
        """검증 리포트 생성"""
        summary = {status.value: 0 for status in ValidationStatus}

        # 상태별 집계
        for result in results:
            summary[result.status.value] += 1

        # 전체 상태 결정
        if summary[ValidationStatus.FAIL.value] > 0:
            overall_status = ValidationStatus.FAIL
        elif summary[ValidationStatus.WARNING.value] > 0:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASS

        # 권장사항 수집
        recommendations = []
        for result in results:
            if result.status in [ValidationStatus.FAIL, ValidationStatus.WARNING]:
                recommendations.extend(result.recommendations)

        return SystemValidationReport(
            overall_status=overall_status,
            results=results,
            summary=summary,
            recommendations=list(set(recommendations))  # 중복 제거
        )

    def format_report(self, report: SystemValidationReport, detailed: bool = False) -> str:
        """리포트 포맷팅"""
        lines = []

        # 헤더
        status_emoji = {
            ValidationStatus.PASS: "✅",
            ValidationStatus.WARNING: "⚠️",
            ValidationStatus.FAIL: "❌",
            ValidationStatus.NOT_APPLICABLE: "ℹ️"
        }

        emoji = status_emoji.get(report.overall_status, "❓")
        lines.append("=" * 70)
        lines.append(f"{emoji} 시스템 요구사항 검증 리포트")
        lines.append("=" * 70)

        # 요약
        lines.append(f"전체 상태: {report.overall_status.value.upper()}")
        lines.append(f"통과: {report.summary[ValidationStatus.PASS.value]}개")
        lines.append(f"경고: {report.summary[ValidationStatus.WARNING.value]}개")
        lines.append(f"실패: {report.summary[ValidationStatus.FAIL.value]}개")
        lines.append("")

        # 상세 결과
        if detailed:
            lines.append("📋 상세 검증 결과:")
            lines.append("")

            for result in report.results:
                emoji = status_emoji.get(result.status, "❓")
                required_mark = " (필수)" if result.required else ""
                lines.append(f"{emoji} {result.name}{required_mark}")
                lines.append(f"   {result.message}")

                if result.details:
                    lines.append(f"   상세: {result.details}")

                if result.recommendations:
                    lines.append("   권장사항:")
                    for rec in result.recommendations[:2]:  # 최대 2개
                        lines.append(f"     • {rec}")

                lines.append("")

        # 주요 권장사항
        if report.recommendations:
            lines.append("💡 주요 권장사항:")
            for i, rec in enumerate(report.recommendations[:5], 1):  # 최대 5개
                lines.append(f"   {i}. {rec}")
            lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def create_fix_script(self, report: SystemValidationReport) -> str:
        """자동 수정 스크립트 생성"""
        script_lines = []
        script_lines.append("#!/bin/bash")
        script_lines.append("# 시스템 요구사항 자동 수정 스크립트")
        script_lines.append("# 주의: 실행 전 내용을 검토하세요")
        script_lines.append("")

        for result in report.results:
            if result.status == ValidationStatus.FAIL and result.recommendations:
                script_lines.append(f"# {result.name} 수정")
                for rec in result.recommendations:
                    if rec.startswith("pip install"):
                        script_lines.append(f"echo '설치 중: {rec}'")
                        script_lines.append(rec)
                    elif rec.startswith("mkdir"):
                        script_lines.append(rec)
                script_lines.append("")

        return "\n".join(script_lines)


# 편의 함수들
def validate_system(config_file: Optional[str] = None, detailed: bool = True) -> str:
    """시스템 검증 실행 및 리포트 반환"""
    validator = SystemValidator()
    report = validator.validate_all(config_file)
    return validator.format_report(report, detailed)


def check_system_requirements() -> bool:
    """기본 시스템 요구사항 확인 (True/False 반환)"""
    validator = SystemValidator()
    report = validator.validate_all()
    return report.overall_status != ValidationStatus.FAIL


def get_system_info() -> Dict[str, Any]:
    """시스템 정보 수집"""
    info = {
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'platform': platform.platform(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'hostname': platform.node(),
    }

    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('.')

        info.update({
            'memory_total_gb': memory.total / (1024 ** 3),
            'memory_available_gb': memory.available / (1024 ** 3),
            'disk_total_gb': disk.total / (1024 ** 3),
            'disk_free_gb': disk.free / (1024 ** 3),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True)
        })
    except ImportError:
        pass

    return info


if __name__ == "__main__":
    # 명령행에서 직접 실행 시
    import argparse

    parser = argparse.ArgumentParser(description="시스템 요구사항 검증")
    parser.add_argument("--config", help="설정 파일 경로")
    parser.add_argument("--detailed", action="store_true", help="상세 리포트")
    parser.add_argument("--fix-script", help="수정 스크립트 생성 경로")

    args = parser.parse_args()

    validator = SystemValidator()
    report = validator.validate_all(args.config)

    # 리포트 출력
    print(validator.format_report(report, args.detailed))

    # 수정 스크립트 생성
    if args.fix_script:
        script = validator.create_fix_script(report)
        with open(args.fix_script, 'w') as f:
            f.write(script)
        print(f"\n수정 스크립트 생성: {args.fix_script}")

    # 종료 코드 설정
    sys.exit(0 if report.overall_status != ValidationStatus.FAIL else 1)