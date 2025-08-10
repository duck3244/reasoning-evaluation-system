# LLM 추론 성능 평가 시스템 - 설치 가이드

이 가이드는 LLM 추론 성능 평가 시스템의 완전한 설치 및 설정 방법을 제공합니다.

## 📋 목차

1. [시스템 요구사항](#시스템-요구사항)
2. [사전 준비](#사전-준비)
3. [설치 단계](#설치-단계)
4. [Oracle 데이터베이스 설정](#oracle-데이터베이스-설정)
5. [환경 변수 설정](#환경-변수-설정)
6. [설치 검증](#설치-검증)
7. [트러블슈팅](#트러블슈팅)

## 🖥️ 시스템 요구사항

### 필수 요구사항
- **Python**: 3.8 이상 3.12 이하
- **메모리**: 최소 4GB (권장 8GB 이상)
- **디스크 공간**: 최소 1GB
- **Oracle Database**: 
  - Thin 모드: Oracle Database 12.1 이상
  - Thick 모드: Oracle Database 9.2 이상

### 권장 환경
- 가상환경 사용 (venv, conda 등)
- SSD 스토리지
- 안정적인 네트워크 연결

## 🔧 사전 준비

### 1. Python 환경 확인

```bash
# Python 버전 확인
python --version
# 또는
python3 --version

# pip 확인
pip --version
```

### 2. 가상환경 생성 (권장)

```bash
# venv 사용
python -m venv llm-eval-env
source llm-eval-env/bin/activate  # Linux/macOS
# 또는
llm-eval-env\Scripts\activate     # Windows

# conda 사용
conda create -n llm-eval python=3.11
conda activate llm-eval
```

## 📦 설치 단계

### 1. 프로젝트 클론

```bash
git clone https://github.com/your-repo/llm-reasoning-evaluation.git
cd llm-reasoning-evaluation
```

### 2. 의존성 설치

```bash
# 기본 패키지 설치
pip install -r requirements.txt

# 전체 기능 사용 시 (선택적)
pip install -r requirements-optional.txt
```

### 3. 시스템 요구사항 검증

```bash
# 자동 검증 도구 실행
python -c "from core.system_validator import validate_system; print(validate_system())"
```

## 🗄️ Oracle 데이터베이스 설정

### Oracle 버전별 설정

#### Option 1: Thin 모드 (권장)
- **장점**: Oracle Client 라이브러리 불필요, 설치 간단
- **요구사항**: Oracle Database 12.1 이상
- **설정**: 추가 설정 불필요

#### Option 2: Thick 모드
- **장점**: 모든 Oracle 버전 지원, 전체 기능 사용 가능
- **요구사항**: Oracle Client 라이브러리 설치 필요

##### Thick 모드 설정 (Oracle Client 설치)

**Linux:**
```bash
# Oracle Instant Client 다운로드
wget https://download.oracle.com/otn_software/linux/instantclient/instantclient-basic-linux.x64-21.9.0.0.0.zip

# 압축 해제
sudo mkdir -p /opt/oracle
cd /opt/oracle
sudo unzip instantclient-basic-linux.x64-21.9.0.0.0.zip

# 라이브러리 경로 설정
export LD_LIBRARY_PATH=/opt/oracle/instantclient_21_9:$LD_LIBRARY_PATH

# 또는 시스템에 영구 추가
echo 'export LD_LIBRARY_PATH=/opt/oracle/instantclient_21_9:$LD_LIBRARY_PATH' >> ~/.bashrc
```

**macOS:**
```bash
# Homebrew 사용 (권장)
brew install instantclient-basic

# 또는 수동 설치
mkdir -p ~/oracle
cd ~/oracle
# Oracle 사이트에서 macOS용 Instant Client 다운로드 후
unzip instantclient-basic-macos.x64-21.9.0.0.0.zip
export DYLD_LIBRARY_PATH=~/oracle/instantclient_21_9:$DYLD_LIBRARY_PATH
```

**Windows:**
```cmd
REM Oracle 사이트에서 Windows용 Instant Client 다운로드
REM C:\oracle\instantclient_21_9에 압축 해제
REM 시스템 PATH에 추가: 제어판 > 시스템 > 고급 시스템 설정 > 환경 변수
REM PATH에 C:\oracle\instantclient_21_9 추가
```

### 데이터베이스 연결 정보 확인

다음 정보를 데이터베이스 관리자에게 확인하세요:

- **호스트명/IP**: 데이터베이스 서버 주소
- **포트**: 일반적으로 1521
- **서비스명 또는 SID**: 데이터베이스 인스턴스 식별자
- **사용자명**: 데이터베이스 계정
- **패스워드**: 계정 비밀번호

## 🔐 환경 변수 설정

### 1. 환경 변수 파일 생성

```bash
# 샘플 파일 생성
python -c "from core.secure_database_config import create_sample_env_file; create_sample_env_file()"

# .env 파일 생성
cp .env.example .env
```

### 2. .env 파일 편집

```bash
# 필수 설정
DB_USERNAME=your_username
DB_PASSWORD=your_password
DB_DSN=your_host:1521/your_service

# 선택적 설정
DB_ORACLE_MODE=auto
DB_POOL_MAX=10
```

### 3. 환경 변수 로드

**Linux/macOS:**
```bash
# 현재 세션에만 적용
source .env
# 또는
export $(cat .env | xargs)

# 영구 적용 (.bashrc에 추가)
echo 'source /path/to/project/.env' >> ~/.bashrc
```

**Windows:**
```cmd
REM PowerShell
Get-Content .env | ForEach-Object { 
    $name, $value = $_.split('=', 2)
    [Environment]::SetEnvironmentVariable($name, $value, "Process")
}
```

## ✅ 설치 검증

### 1. 자동 검증 도구 실행

```bash
# 전체 시스템 검증
python core/system_validator.py --config config/db_config.json --detailed

# 데이터베이스 연결만 테스트
python -c "
from core.secure_database_config import get_database_config_status
print('연결 상태:', get_database_config_status())
"
```

### 2. 데이터베이스 초기화

```bash
# 데이터베이스 초기화
python scripts/init_database.py --config config/db_config.json

# 샘플 데이터 로드
python scripts/load_sample_data.py --basic --external all
```

### 3. 기본 테스트 실행

```bash
# 시스템 테스트
python main.py

# 진단 리포트 생성
python -c "
from core.secure_database_config import run_database_diagnostics
print(run_database_diagnostics())
"
```

## 🚨 트러블슈팅

### 일반적인 문제들

#### 1. Python 버전 호환성 문제

**문제**: `python-oracledb` 설치 실패
```
ERROR: Could not find a version that satisfies the requirement oracledb
```

**해결책**:
```bash
# Python 버전 확인
python --version

# 3.8 이상인지 확인
# 필요시 Python 업그레이드 또는 pyenv 사용
pyenv install 3.11.0
pyenv local 3.11.0
```

#### 2. Oracle 연결 실패

**문제**: DPY-3010 오류 (데이터베이스 버전 미지원)
```
DPY-3010: connections to this database server version are not supported
```

**해결책**:
```python
# Thick 모드로 전환
import oracledb
oracledb.init_oracle_client()
```

**문제**: DPY-3015 오류 (패스워드 검증자 호환성)
```
DPY-3015: password verifier type 0x939 is not supported
```

**해결책**:
1. 데이터베이스 관리자에게 패스워드 재설정 요청
2. Thick 모드 사용
3. 다음 SQL로 패스워드 업데이트:
```sql
ALTER USER username IDENTIFIED BY new_password;
```

#### 3. Oracle Client 라이브러리 문제

**문제**: DPI-1047 오류 (Oracle Client 라이브러리 없음)
```
DPI-1047: Cannot locate a 64-bit Oracle Client library
```

**해결책**:
```bash
# 1. Oracle Instant Client 설치 확인
ls -la /opt/oracle/instantclient*

# 2. 환경 변수 설정
export LD_LIBRARY_PATH=/opt/oracle/instantclient_21_9:$LD_LIBRARY_PATH

# 3. Python에서 직접 경로 지정
python -c "
import oracledb
oracledb.init_oracle_client(lib_dir='/opt/oracle/instantclient_21_9')
"
```

#### 4. 메모리 부족 문제

**문제**: 대용량 데이터 처리 시 메모리 오류
```
MemoryError: Unable to allocate memory
```

**해결책**:
```python
# 배치 크기 감소
from core.data_collector import ReasoningDatasetCollector
collector = ReasoningDatasetCollector(db_manager, batch_size=500)

# 메모리 정리 활성화
from core.performance_optimizer import MemoryManager
memory_manager = MemoryManager()
memory_manager.cleanup_if_needed(force=True)
```

#### 5. 권한 문제

**문제**: 파일 쓰기 권한 없음
```
PermissionError: [Errno 13] Permission denied
```

**해결책**:
```bash
# 디렉토리 권한 확인
ls -la logs/

# 권한 수정
chmod 755 logs/
chmod 644 logs/*.log

# 또는 다른 디렉토리에서 실행
cd ~/workspace/
python /path/to/project/main.py
```

### 환경별 특이사항

#### Docker 환경

```dockerfile
# Dockerfile 예시
FROM python:3.11-slim

# Oracle Instant Client 설치
RUN apt-get update && apt-get install -y \
    wget unzip libaio1 && \
    mkdir -p /opt/oracle && \
    cd /opt/oracle && \
    wget https://download.oracle.com/otn_software/linux/instantclient/instantclient-basic-linux.x64-21.9.0.0.0.zip && \
    unzip instantclient-basic-linux.x64-21.9.0.0.0.zip && \
    rm instantclient-basic-linux.x64-21.9.0.0.0.zip

ENV LD_LIBRARY_PATH=/opt/oracle/instantclient_21_9:$LD_LIBRARY_PATH

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app
```

#### AWS EC2 환경

```bash
# Oracle Client 설치
sudo yum install -y oracle-instantclient-basic oracle-instantclient-devel

# 또는 Amazon Linux 2
sudo yum install -y libaio
wget https://download.oracle.com/otn_software/linux/instantclient/instantclient-basic-linux.x64-21.9.0.0.0.zip
sudo unzip -d /opt/oracle instantclient-basic-linux.x64-21.9.0.0.0.zip
```

### 고급 진단 도구

#### 1. 자동 수정 스크립트 생성

```bash
# 문제 진단 및 수정 스크립트 생성
python core/system_validator.py --fix-script fix_issues.sh
chmod +x fix_issues.sh
./fix_issues.sh
```

#### 2. 상세 로그 활성화

```python
# 디버그 로깅 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# Oracle 드라이버 디버그
import os
os.environ['DPI_DEBUG_LEVEL'] = '64'
```

#### 3. 연결 테스트 도구

```python
# 단계별 연결 테스트
from core.oracle_compatibility import OracleCompatibilityManager

manager = OracleCompatibilityManager()
result = manager.auto_configure_oracle_mode(
    username='user', 
    password='pass', 
    dsn='host:port/service'
)
print(manager.get_compatibility_report())
```

## 📞 지원 및 도움

### 문서 링크
- [Oracle python-oracledb 문서](https://python-oracledb.readthedocs.io/)
- [Oracle Instant Client 다운로드](https://www.oracle.com/database/technologies/instant-client.html)
- [프로젝트 GitHub Issues](https://github.com/your-repo/issues)

### 로그 수집 방법

문제 해결을 위해 다음 정보를 수집해주세요:

1. **시스템 정보**:
```bash
python -c "
from core.system_validator import get_system_info
import json
print(json.dumps(get_system_info(), indent=2))
"
```

2. **에러 로그**:
```bash
# 로그 파일 확인
cat logs/application.log | tail -50

# 에러만 필터링
grep -i error logs/application.log
```

3. **Oracle 연결 진단**:
```bash
python -c "
from core.secure_database_config import run_database_diagnostics
print(run_database_diagnostics())
"
```

## 🚀 다음 단계

설치가 완료되면 다음 가이드를 참조하세요:

1. **[사용자 가이드](USER_GUIDE.md)**: 기본 사용법
2. **[API 문서](API_REFERENCE.md)**: 개발자용 API 가이드  
3. **[성능 튜닝 가이드](PERFORMANCE_TUNING.md)**: 최적화 방법
4. **[백업 가이드](BACKUP_GUIDE.md)**: 데이터 백업 및 복원

---

**⚠️ 중요**: 이 시스템은 데이터베이스에 연결하므로 보안에 주의하세요. 프로덕션 환경에서는 반드시 환경 변수를 사용하고, 패스워드를 안전하게 관리하세요.