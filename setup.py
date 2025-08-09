"""
LLM 추론 성능 평가 시스템 설치 스크립트
"""
import os
from setuptools import setup, find_packages

# 현재 디렉토리
here = os.path.abspath(os.path.dirname(__file__))

# README 파일 읽기
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# requirements.txt 읽기
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# 버전 정보
__version__ = '1.0.0'

setup(
    name='llm-reasoning-evaluation',
    version=__version__,
    description='Oracle 데이터베이스 기반 LLM 추론 성능 평가 시스템',
    long_description=long_description,
    long_description_content_type='text/markdown',

    # 프로젝트 정보
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/your-username/llm-reasoning-evaluation',

    # 라이선스
    license='MIT',

    # 분류
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Database :: Database Engines/Servers',
    ],

    # 키워드
    keywords='llm language-model evaluation reasoning oracle database performance ai ml',

    # 패키지 설정
    packages=find_packages(exclude=['tests*', 'docs*', 'examples*']),

    # Python 버전 요구사항
    python_requires='>=3.8',

    # 의존성
    install_requires=requirements,

    # 추가 의존성 (선택적 설치)
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'pytest-mock>=3.10.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
            'pre-commit>=3.0.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
            'myst-parser>=1.0.0',
        ],
        'performance': [
            'psutil>=5.9.0',
            'memory-profiler>=0.60.0',
            'line-profiler>=4.0.0',
        ],
        'visualization': [
            'matplotlib>=3.6.0',
            'seaborn>=0.12.0',
            'plotly>=5.15.0',
        ],
        'extended': [
            'datasets>=2.14.0',  # Hugging Face datasets
            'transformers>=4.30.0',  # Hugging Face transformers
            'torch>=2.0.0',  # PyTorch (선택적)
            'openai>=1.0.0',  # OpenAI API (선택적)
        ]
    },

    # 데이터 파일 포함
    package_data={
        'llm_reasoning_evaluation': [
            'config/*.json',
            'config/*.yaml',
            'data/samples/*.json',
            'docs/*.md',
        ],
    },

    # 추가 데이터 파일
    include_package_data=True,

    # 실행 가능한 스크립트
    entry_points={
        'console_scripts': [
            'llm-eval=main:main',
            'llm-eval-init=scripts.init_database:main',
            'llm-eval-load=scripts.load_sample_data:main',
            'llm-eval-backup=scripts.backup_restore:backup_main',
            'llm-eval-restore=scripts.backup_restore:restore_main',
        ],
    },

    # 프로젝트 URL들
    project_urls={
        'Bug Reports': 'https://github.com/your-username/llm-reasoning-evaluation/issues',
        'Source': 'https://github.com/your-username/llm-reasoning-evaluation',
        'Documentation': 'https://llm-reasoning-evaluation.readthedocs.io/',
        'Changelog': 'https://github.com/your-username/llm-reasoning-evaluation/blob/main/CHANGELOG.md',
    },

    # zip_safe 설정
    zip_safe=False,

    # 테스트 설정
    test_suite='tests',
    tests_require=[
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'pytest-mock>=3.10.0',
    ],

    # 명령어 별칭
    cmdclass={},
)

# 설치 후 메시지
print("""
🎉 LLM 추론 성능 평가 시스템이 설치되었습니다!

다음 단계:
1. 데이터베이스 설정: cp config/db_config.json.example config/db_config.json
2. 설정 파일 편집: config/db_config.json
3. 데이터베이스 초기화: llm-eval-init
4. 샘플 데이터 로드: llm-eval-load
5. 시스템 실행: llm-eval

도움말: llm-eval --help

문서: https://llm-reasoning-evaluation.readthedocs.io/
""")