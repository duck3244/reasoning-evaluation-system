#!/usr/bin/env python3
"""
설치 검증 스크립트
LLM 추론 성능 평가 시스템의 설치가 올바르게 완료되었는지 검증합니다.
"""
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.system_validator import SystemValidator, ValidationStatus
from core.secure_database_config import get_database_config_status, run_database_diagnostics
from core.oracle_compatibility import configure_oracle_compatibility
from core.error_handler import handle_database_error, format_user_error
from monitoring.logging_system import setup_application_logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SetupValidator:
    """설치 검증기"""

    def __init__(self, config_file: str = None):
        self.config_file = config_file or "config/db_config.json"
        self.validator = SystemValidator()
        self.validation_results = {}

    def run_complete_validation(self, skip_db_test: bool = False) -> Dict[str, Any]:
        """완전한 설치 검증 실행"""
        print("🔍 LLM 추론 성능 평가 시스템 설치 검증 시작...")
        print("=" * 60)

        results = {
            'overall_status': 'unknown',
            'validations': {},
            'recommendations': [],
            'next_steps': []
        }

        # 1. 시스템 요구사항 검증
        print("1️⃣ 시스템 요구사항 검증 중...")
        sys_validation = self._validate_system_requirements()
        results['validations']['system'] = sys_validation

        # 2. 프로젝트 구조 검증
        print("2️⃣ 프로젝트 구조 검증 중...")
        structure_validation = self._validate_project_structure()
        results['validations']['structure'] = structure_validation

        # 3. 의존성 검증
        print("3️⃣ Python 의존성 검증 중...")
        deps_validation = self._validate_dependencies()
        results['validations']['dependencies'] = deps_validation

        # 4. 설정 파일 검증
        print("4️⃣ 설정 파일 검증 중...")
        config_validation = self._validate_configuration()
        results['validations']['configuration'] = config_validation

        # 5. 데이터베이스 연결 테스트 (선택적)
        if not skip_db_test:
            print("5️⃣ 데이터베이스 연결 테스트 중...")
            db_validation = self._validate_database_connection()
            results['validations']['database'] = db_validation
        else:
            print("5️⃣ 데이터베이스 연결 테스트 건너뜀")
            results['validations']['database'] = {
                'status': 'skipped',
                'message': '데이터베이스 테스트를 건너뛰었습니다'
            }

        # 6. Oracle 호환성 검증
        if not skip_db_test and results['validations']['database']['status'] == 'pass':
            print("6️⃣ Oracle 호환성 검증 중...")
            oracle_validation = self._validate_oracle_compatibility()
            results['validations']['oracle'] = oracle_validation

        # 7. 전체 결과 종합
        results['overall_status'] = self._determine_overall_status(results['validations'])
        results['recommendations'] = self._generate_recommendations(results['validations'])
        results['next_steps'] = self._generate_next_steps(results['overall_status'])

        return results

    def _validate_system_requirements(self) -> Dict[str, Any]:
        """시스템 요구사항 검증"""
        try:
            # SystemValidator 사용
            validation_results = self.validator._validate_python_environment()
            validation_results.extend(self.validator._validate_system_resources())

            # 결과 요약
            failed_count = sum(1 for r in validation_results if r.status == ValidationStatus.FAIL)
            warning_count = sum(1 for r in validation_results if r.status == ValidationStatus.WARNING)

            if failed_count > 0:
                status = 'fail'
                message = f"{failed_count}개 필수 요구사항 미충족"
            elif warning_count > 0:
                status = 'warning'
                message = f"{warning_count}개 권장사항 미충족"
            else:
                status = 'pass'
                message = "모든 시스템 요구사항 충족"

            details = []
            for result in validation_results:
                if result.status in [ValidationStatus.FAIL, ValidationStatus.WARNING]:
                    details.append(f"{result.name}: {result.message}")

            return {
                'status': status,
                'message': message,
                'details': details,
                'recommendations': [r.recommendations for r in validation_results if r.recommendations]
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"시스템 검증 중 오류: {e}",
                'details': [str(e)]
            }

    def _validate_project_structure(self) -> Dict[str, Any]:
        """프로젝트 구조 검증"""
        required_dirs = [
            'core', 'config', 'scripts', 'evaluation',
            'data_loaders', 'monitoring', 'docs'
        ]

        required_files = [
            'main.py', 'requirements.txt', 'README.md'
        ]

        missing_dirs = [d for d in required_dirs if not os.path.isdir(d)]
        missing_files = [f for f in required_files if not os.path.isfile(f)]

        if missing_dirs or missing_files:
            status = 'fail' if missing_dirs else 'warning'
            message = "프로젝트 구조에 누락된 항목이 있습니다"
            details = []
            if missing_dirs:
                details.append(f"누락된 디렉토리: {', '.join(missing_dirs)}")
            if missing_files:
                details.append(f"누락된 파일: {', '.join(missing_files)}")
        else:
            status = 'pass'
            message = "프로젝트 구조가 올바릅니다"
            details = []

        return {
            'status': status,
            'message': message,
            'details': details
        }

    def _validate_dependencies(self) -> Dict[str, Any]:
        """Python 의존성 검증"""
        required_packages = {
            'oracledb': '1.4.0',
            'requests': '2.31.0',
            'psutil': '5.9.0'
        }

        missing_packages = []
        outdated_packages = []

        for package, min_version in required_packages.items():
            try:
                import importlib
                module = importlib.import_module(package)

                version = getattr(module, '__version__', 'Unknown')
                if version != 'Unknown':
                    from packaging import version as pkg_version
                    if pkg_version.parse(version) < pkg_version.parse(min_version):
                        outdated_packages.append(f"{package} {version} (필요: {min_version}+)")

            except ImportError:
                missing_packages.append(f"{package}>={min_version}")

        if missing_packages:
            status = 'fail'
            message = f"{len(missing_packages)}개 필수 패키지 누락"
            details = [f"누락 패키지: {', '.join(missing_packages)}"]
        elif outdated_packages:
            status = 'warning'
            message = f"{len(outdated_packages)}개 패키지 업데이트 필요"
            details = [f"업데이트 필요: {', '.join(outdated_packages)}"]
        else:
            status = 'pass'
            message = "모든 필수 패키지가 설치되어 있습니다"
            details = []

        return {
            'status': status,
            'message': message,
            'details': details,
            'install_command': f"pip install {' '.join(missing_packages)}" if missing_packages else None
        }

    def _validate_configuration(self) -> Dict[str, Any]:
        """설정 파일 검증"""
        try:
            # 데이터베이스 설정 상태 확인
            db_status = get_database_config_status(self.config_file)

            issues = []
            if not db_status['config_file_exists']:
                issues.append("설정 파일이 존재하지 않습니다")

            if not db_status['env_vars_available']:
                issues.append("환경 변수가 설정되지 않았습니다")

            if issues:
                status = 'warning' if db_status['env_vars_available'] or db_status['config_file_exists'] else 'fail'
                message = "설정에 문제가 있습니다"
                details = issues
            else:
                status = 'pass'
                message = "설정이 올바르게 구성되었습니다"
                details = []

            return {
                'status': status,
                'message': message,
                'details': details,
                'has_config_file': db_status['config_file_exists'],
                'has_env_vars': db_status['env_vars_available']
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"설정 검증 중 오류: {e}",
                'details': [str(e)]
            }

    def _validate_database_connection(self) -> Dict[str, Any]:
        """데이터베이스 연결 검증"""
        try:
            from core.secure_database_config import SecureDatabaseConfigLoader

            loader = SecureDatabaseConfigLoader()
            secure_config = loader.load_secure_config(self.config_file)
            db_manager = loader.create_database_manager(secure_config)

            # 연결 테스트
            if db_manager.db_config.test_connection():
                status = 'pass'
                message = "데이터베이스 연결 성공"
                details = [f"DSN: {secure_config.dsn}", f"모드: {secure_config.oracle_mode}"]
            else:
                status = 'fail'
                message = "데이터베이스 연결 실패"
                details = ["연결 테스트가 실패했습니다"]

            return {
                'status': status,
                'message': message,
                'details': details
            }

        except Exception as e:
            error_info = handle_database_error(e, "데이터베이스 연결 테스트")

            return {
                'status': 'fail',
                'message': "데이터베이스 연결 실패",
                'details': [error_info.user_message],
                'error_code': error_info.error_code,
                'solutions': error_info.solutions[:3]  # 최대 3개 해결책
            }

    def _validate_oracle_compatibility(self) -> Dict[str, Any]:
        """Oracle 호환성 검증"""
        try:
            from core.secure_database_config import SecureDatabaseConfigLoader

            loader = SecureDatabaseConfigLoader()
            secure_config = loader.load_secure_config(self.config_file)

            # Oracle 호환성 확인
            compatibility_info = configure_oracle_compatibility(
                username=secure_config.username,
                password=secure_config.password,
                dsn=secure_config.dsn,
                preferred_mode=secure_config.oracle_mode
            )

            if compatibility_info.mode:
                status = 'pass'
                message = f"Oracle {compatibility_info.mode.value} 모드 호환"
                details = [f"버전: {compatibility_info.version or 'Unknown'}"]

                if compatibility_info.warnings:
                    details.extend([f"경고: {w}" for w in compatibility_info.warnings[:2]])

            else:
                status = 'fail'
                message = "Oracle 호환성 문제"
                details = compatibility_info.warnings or ["호환성 확인 실패"]

            return {
                'status': status,
                'message': message,
                'details': details,
                'mode': compatibility_info.mode.value if compatibility_info.mode else None,
                'version': compatibility_info.version
            }

        except Exception as e:
            return {
                'status': 'error',
                'message': f"Oracle 호환성 확인 중 오류: {e}",
                'details': [str(e)]
            }

    def _determine_overall_status(self, validations: Dict[str, Any]) -> str:
        """전체 상태 결정"""
        statuses = [v.get('status', 'unknown') for v in validations.values()]

        if 'fail' in statuses or 'error' in statuses:
            return 'fail'
        elif 'warning' in statuses:
            return 'warning'
        elif all(s in ['pass', 'skipped'] for s in statuses):
            return 'pass'
        else:
            return 'unknown'

    def _generate_recommendations(self, validations: Dict[str, Any]) -> List[str]:
        """권장사항 생성"""
        recommendations = []

        for validation_name, validation in validations.items():
            if validation.get('status') in ['fail', 'warning', 'error']:

                # 시스템 요구사항 관련
                if validation_name == 'system':
                    recommendations.append("시스템 요구사항을 확인하고 필요한 업그레이드를 진행하세요")

                # 의존성 관련
                elif validation_name == 'dependencies':
                    if validation.get('install_command'):
                        recommendations.append(f"다음 명령으로 누락된 패키지를 설치하세요: {validation['install_command']}")

                # 설정 관련
                elif validation_name == 'configuration':
                    if not validation.get('has_config_file'):
                        recommendations.append("python scripts/init_database.py --create-config 명령으로 설정 파일을 생성하세요")
                    if not validation.get('has_env_vars'):
                        recommendations.append("보안을 위해 환경 변수 사용을 권장합니다 (.env 파일 생성)")

                # 데이터베이스 관련
                elif validation_name == 'database':
                    if validation.get('solutions'):
                        recommendations.extend(validation['solutions'])
                    recommendations.append("데이터베이스 연결 정보와 서버 상태를 확인하세요")

        return list(set(recommendations))  # 중복 제거

    def _generate_next_steps(self, overall_status: str) -> List[str]:
        """다음 단계 생성"""
        if overall_status == 'pass':
            return [
                "🎉 설치가 완료되었습니다!",
                "python scripts/init_database.py 명령으로 데이터베이스를 초기화하세요",
                "python scripts/load_sample_data.py 명령으로 샘플 데이터를 로드하세요",
                "python main.py 명령으로 시스템을 실행해보세요",
                "docs/USER_GUIDE.md 문서를 참조하여 사용법을 익히세요"
            ]
        elif overall_status == 'warning':
            return [
                "⚠️ 일부 권장사항을 확인하세요",
                "위의 권장사항을 따라 설정을 개선하세요",
                "개선 후 다시 검증을 실행하세요: python scripts/validate_setup.py",
                "문제가 지속되면 docs/INSTALLATION_GUIDE.md를 참조하세요"
            ]
        else:  # fail or error
            return [
                "❌ 설치에 문제가 있습니다",
                "위의 권장사항을 따라 문제를 해결하세요",
                "docs/INSTALLATION_GUIDE.md의 트러블슈팅 섹션을 확인하세요",
                "해결 후 다시 검증하세요: python scripts/validate_setup.py",
                "도움이 필요하면 GitHub Issues에 문의하세요"
            ]

    def format_results(self, results: Dict[str, Any], detailed: bool = False) -> str:
        """결과 포맷팅"""
        lines = []

        # 헤더
        status_emoji = {
            'pass': '✅',
            'warning': '⚠️',
            'fail': '❌',
            'error': '🚨',
            'skipped': 'ℹ️'
        }

        overall_emoji = status_emoji.get(results['overall_status'], '❓')
        lines.append("=" * 60)
        lines.append(f"{overall_emoji} 설치 검증 결과: {results['overall_status'].upper()}")
        lines.append("=" * 60)

        # 개별 검증 결과
        if detailed:
            lines.append("\n📋 상세 검증 결과:")
            for name, validation in results['validations'].items():
                emoji = status_emoji.get(validation.get('status', 'unknown'), '❓')
                lines.append(f"\n{emoji} {name.title()}: {validation.get('message', 'Unknown')}")

                if validation.get('details'):
                    for detail in validation['details']:
                        lines.append(f"   • {detail}")

        # 권장사항
        if results['recommendations']:
            lines.append(f"\n💡 권장사항:")
            for i, rec in enumerate(results['recommendations'], 1):
                lines.append(f"   {i}. {rec}")

        # 다음 단계
        if results['next_steps']:
            lines.append(f"\n🚀 다음 단계:")
            for step in results['next_steps']:
                lines.append(f"   • {step}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LLM 추론 평가 시스템 설치 검증")

    parser.add_argument(
        "--config", "-c",
        default="config/db_config.json",
        help="데이터베이스 설정 파일 경로"
    )

    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="데이터베이스 연결 테스트 건너뛰기"
    )

    parser.add_argument(
        "--detailed", "-d",
        action="store_true",
        help="상세 결과 표시"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="최소한의 출력만 표시"
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help="자동 수정 제안 생성"
    )

    args = parser.parse_args()

    # 로깅 레벨 조정
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # 검증 실행
        validator = SetupValidator(args.config)
        results = validator.run_complete_validation(skip_db_test=args.skip_db)

        # 결과 출력
        if not args.quiet:
            report = validator.format_results(results, detailed=args.detailed)
            print(report)

        # 자동 수정 제안
        if args.fix and results['overall_status'] != 'pass':
            print("\n🔧 자동 수정 제안:")
            print("다음 명령들을 실행해보세요:\n")

            for validation_name, validation in results['validations'].items():
                if validation.get('status') in ['fail', 'error'] and validation.get('install_command'):
                    print(f"# {validation_name} 수정")
                    print(validation['install_command'])
                    print()

        # 종료 코드 설정
        if results['overall_status'] == 'pass':
            sys.exit(0)
        elif results['overall_status'] == 'warning':
            sys.exit(1) if not args.quiet else sys.exit(0)
        else:
            sys.exit(2)

    except KeyboardInterrupt:
        print("\n중단됨")
        sys.exit(130)
    except Exception as e:
        logger.error(f"검증 중 예상치 못한 오류: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(3)


if __name__ == "__main__":
    main()