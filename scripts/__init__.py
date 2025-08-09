"""
LLM 추론 성능 평가 시스템 - 스크립트 모듈

이 모듈은 시스템 관리를 위한 다양한 유틸리티 스크립트들을 포함합니다.

스크립트 목록:
- init_database.py: 데이터베이스 초기화
- load_sample_data.py: 샘플 데이터 로드
- backup_restore.py: 백업 및 복원

사용법:
    # 데이터베이스 초기화
    python scripts/init_database.py --config config/db_config.json

    # 샘플 데이터 로드
    python scripts/load_sample_data.py --basic --external all

    # 백업 생성
    python scripts/backup_restore.py backup --output backups/backup_$(date +%Y%m%d).json.gz

    # 백업 복원
    python scripts/backup_restore.py restore --backup backups/backup_20240101.json.gz
"""

__version__ = '1.0.0'
__author__ = 'LLM Reasoning Evaluation Team'

# 스크립트 정보
SCRIPTS = {
    'init_database': {
        'file': 'init_database.py',
        'description': '데이터베이스 테이블, 인덱스, 시퀀스 초기화',
        'usage': 'python scripts/init_database.py [options]'
    },
    'load_sample_data': {
        'file': 'load_sample_data.py',
        'description': '샘플 데이터 로드 및 데이터 관리',
        'usage': 'python scripts/load_sample_data.py [options]'
    },
    'backup_restore': {
        'file': 'backup_restore.py',
        'description': '데이터 백업 및 복원',
        'usage': 'python scripts/backup_restore.py [backup|restore] [options]'
    }
}


def get_script_info(script_name: str = None):
    """스크립트 정보 조회"""
    if script_name:
        return SCRIPTS.get(script_name)
    return SCRIPTS


def list_scripts():
    """사용 가능한 스크립트 목록 출력"""
    print("LLM 추론 성능 평가 시스템 - 관리 스크립트")
    print("=" * 60)

    for name, info in SCRIPTS.items():
        print(f"\n📜 {name}")
        print(f"   파일: {info['file']}")
        print(f"   설명: {info['description']}")
        print(f"   사용법: {info['usage']}")

    print("\n" + "=" * 60)
    print("상세한 옵션은 각 스크립트에 --help 옵션을 사용하세요.")
    print("예: python scripts/init_database.py --help")


if __name__ == "__main__":
    list_scripts()