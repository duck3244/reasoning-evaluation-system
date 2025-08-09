#!/usr/bin/env python3
"""
샘플 데이터 로드 스크립트
다양한 소스에서 샘플 데이터를 로드하여 데이터베이스에 저장합니다.
"""
import sys
import os
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

# 프로젝트 루트 디렉토리를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.database_config import DatabaseConfig, DatabaseManager, load_db_config_from_file
from core.data_collector import ReasoningDatasetCollector
from data_loaders.sample_data_generator import SampleDataGenerator
from data_loaders.external_data_loader import ExternalDatasetLoader
from monitoring.logging_system import setup_application_logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SampleDataLoader:
    """샘플 데이터 로더"""

    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.collector = ReasoningDatasetCollector(db_manager)
        self.sample_generator = SampleDataGenerator(self.collector)
        self.external_loader = ExternalDatasetLoader(self.collector)

    def load_basic_samples(self) -> int:
        """기본 샘플 데이터 로드"""
        logger.info("기본 샘플 데이터 로드 시작...")

        try:
            count = self.sample_generator.add_all_sample_data()
            logger.info(f"✅ 기본 샘플 데이터 로드 완료: {count}개")
            return count
        except Exception as e:
            logger.error(f"❌ 기본 샘플 데이터 로드 실패: {e}")
            return 0

    def load_external_samples(self, sources: List[str] = None) -> Dict[str, int]:
        """외부 데이터셋 샘플 로드"""
        if sources is None:
            sources = ['gsm8k', 'arc', 'hellaswag', 'korean']

        logger.info(f"외부 데이터셋 샘플 로드 시작: {', '.join(sources)}")

        results = {}

        try:
            if 'gsm8k' in sources:
                logger.info("GSM8K 샘플 로드 중...")
                count = self.external_loader.load_gsm8k_sample()
                results['gsm8k'] = count
                logger.info(f"  GSM8K: {count}개")

            if 'arc' in sources:
                logger.info("ARC 샘플 로드 중...")
                count = self.external_loader.load_arc_sample()
                results['arc'] = count
                logger.info(f"  ARC: {count}개")

            if 'hellaswag' in sources:
                logger.info("HellaSwag 샘플 로드 중...")
                count = self.external_loader.load_hellaswag_sample()
                results['hellaswag'] = count
                logger.info(f"  HellaSwag: {count}개")

            if 'korean' in sources:
                logger.info("한국어 데이터셋 로드 중...")
                count = self.external_loader.load_korean_datasets()
                results['korean'] = count
                logger.info(f"  한국어: {count}개")

            total = sum(results.values())
            logger.info(f"✅ 외부 데이터셋 샘플 로드 완료: 총 {total}개")
            return results

        except Exception as e:
            logger.error(f"❌ 외부 데이터셋 샘플 로드 실패: {e}")
            return results

    def load_from_file(self, file_path: str, file_format: str = None) -> int:
        """파일에서 데이터 로드"""
        logger.info(f"파일에서 데이터 로드: {file_path}")

        if not os.path.exists(file_path):
            logger.error(f"파일이 존재하지 않습니다: {file_path}")
            return 0

        try:
            if file_format is None:
                # 파일 확장자로 형식 판단
                if file_path.endswith('.json'):
                    file_format = 'json'
                elif file_path.endswith('.csv'):
                    file_format = 'csv'
                else:
                    logger.error("지원하지 않는 파일 형식입니다. JSON 또는 CSV 파일을 사용해주세요.")
                    return 0

            if file_format == 'json':
                count = self.collector.load_from_json(file_path)
            elif file_format == 'csv':
                count = self.collector.load_from_csv(file_path)
            else:
                logger.error(f"지원하지 않는 파일 형식: {file_format}")
                return 0

            logger.info(f"✅ 파일에서 데이터 로드 완료: {count}개")
            return count

        except Exception as e:
            logger.error(f"❌ 파일 로드 실패: {e}")
            return 0

    def load_custom_data(self, custom_data: List[Dict[str, Any]]) -> int:
        """커스텀 데이터 로드"""
        logger.info(f"커스텀 데이터 로드 시작: {len(custom_data)}개")

        try:
            from core.data_models import ReasoningDataPoint

            data_points = []
            for item in custom_data:
                try:
                    data_point = ReasoningDataPoint.from_dict(item)
                    data_points.append(data_point)
                except Exception as e:
                    logger.warning(f"커스텀 데이터 항목 파싱 실패: {e}")
                    continue

            if data_points:
                count = self.collector.add_batch_data_points(data_points)
                logger.info(f"✅ 커스텀 데이터 로드 완료: {count}개")
                return count
            else:
                logger.warning("유효한 커스텀 데이터가 없습니다.")
                return 0

        except Exception as e:
            logger.error(f"❌ 커스텀 데이터 로드 실패: {e}")
            return 0

    def create_demo_evaluation_set(self, size: int = 50) -> int:
        """데모용 평가 세트 생성"""
        logger.info(f"데모용 평가 세트 생성: {size}개")

        try:
            # 기존 데이터 조회
            existing_data = self.collector.get_data(limit=size * 2)  # 여유있게 조회

            if len(existing_data) < size:
                logger.warning(f"충분한 데이터가 없습니다. 요청: {size}개, 사용 가능: {len(existing_data)}개")
                size = len(existing_data)

            # 균형잡힌 선택
            from collections import defaultdict
            import random

            category_data = defaultdict(list)
            for item in existing_data:
                category_data[item.category].append(item)

            # 카테고리별 균등 분배
            demo_data = []
            categories = list(category_data.keys())
            per_category = max(1, size // len(categories))

            for category in categories:
                category_items = category_data[category]
                selected_count = min(per_category, len(category_items))
                selected_items = random.sample(category_items, selected_count)
                demo_data.extend(selected_items)

            # 부족한 경우 추가 선택
            if len(demo_data) < size:
                remaining = size - len(demo_data)
                all_remaining = [item for item in existing_data if item not in demo_data]
                if all_remaining:
                    additional = random.sample(all_remaining, min(remaining, len(all_remaining)))
                    demo_data.extend(additional)

            # 데모 세트를 파일로 저장
            demo_file = "data/demo_evaluation_set.json"
            os.makedirs(os.path.dirname(demo_file), exist_ok=True)

            success = self.collector.export_to_json(demo_file)
            if success:
                logger.info(f"✅ 데모 평가 세트 생성 완료: {demo_file} ({len(demo_data)}개)")

            return len(demo_data)

        except Exception as e:
            logger.error(f"❌ 데모 평가 세트 생성 실패: {e}")
            return 0

    def get_data_statistics(self) -> Dict[str, Any]:
        """데이터 통계 조회"""
        try:
            stats = self.collector.get_statistics()
            return {
                'total_count': stats.total_count,
                'category_counts': stats.category_counts,
                'difficulty_counts': stats.difficulty_counts,
                'source_counts': stats.source_counts,
                'created_at': stats.created_at
            }
        except Exception as e:
            logger.error(f"데이터 통계 조회 실패: {e}")
            return {}

    def validate_data_integrity(self) -> Dict[str, Any]:
        """데이터 무결성 검증"""
        logger.info("데이터 무결성 검증 시작...")

        try:
            # 기본 통계
            stats = self.get_data_statistics()

            # 품질 리포트
            quality_report = self.collector.get_data_quality_report()

            # 중복 데이터 확인
            duplicate_sql = f"""
            SELECT COUNT(*) - COUNT(DISTINCT QUESTION, CORRECT_ANSWER) as duplicates
            FROM {self.collector.db_manager.db_config.username}.REASONING_DATA
            """

            try:
                result = self.collector.db_manager.execute_query(duplicate_sql)
                duplicates = result[0][0] if result else 0
            except:
                duplicates = 0

            validation_result = {
                'total_records': stats.get('total_count', 0),
                'categories': len(stats.get('category_counts', {})),
                'difficulties': len(stats.get('difficulty_counts', {})),
                'sources': len(stats.get('source_counts', {})),
                'quality_issues': len(quality_report.get('data_quality_issues', [])),
                'duplicates': duplicates,
                'validation_passed': True
            }

            # 검증 조건
            if validation_result['total_records'] == 0:
                validation_result['validation_passed'] = False
                logger.warning("⚠️ 데이터가 없습니다.")

            if validation_result['quality_issues'] > 0:
                logger.warning(f"⚠️ 데이터 품질 이슈: {validation_result['quality_issues']}개")

            if validation_result['duplicates'] > 0:
                logger.warning(f"⚠️ 중복 데이터: {validation_result['duplicates']}개")

            if validation_result['validation_passed']:
                logger.info("✅ 데이터 무결성 검증 통과")
            else:
                logger.warning("⚠️ 데이터 무결성 검증에서 문제가 발견되었습니다.")

            return validation_result

        except Exception as e:
            logger.error(f"❌ 데이터 무결성 검증 실패: {e}")
            return {'validation_passed': False, 'error': str(e)}

    def cleanup_data(self, dry_run: bool = True) -> Dict[str, int]:
        """데이터 정리"""
        logger.info(f"데이터 정리 {'시뮬레이션' if dry_run else '실행'}...")

        results = {
            'duplicates_removed': 0,
            'invalid_removed': 0,
            'old_data_removed': 0
        }

        try:
            if not dry_run:
                # 중복 데이터 제거
                optimization_results = self.collector.optimize_storage()
                results['duplicates_removed'] = optimization_results.get('deleted_duplicates', 0)

                # 오래된 임시 데이터 정리
                old_data_count = self.collector.cleanup_old_data(days_old=30)
                results['old_data_removed'] = old_data_count

                logger.info(f"✅ 데이터 정리 완료: {sum(results.values())}개 정리")
            else:
                logger.info("데이터 정리 시뮬레이션 완료 (실제 삭제 없음)")

            return results

        except Exception as e:
            logger.error(f"❌ 데이터 정리 실패: {e}")
            return results


def create_sample_datasets():
    """샘플 데이터셋 파일 생성"""
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 수학 문제 샘플
    math_sample = [
        {
            "id": "math_sample_1",
            "category": "math",
            "difficulty": "easy",
            "question": "25 + 17 = ?",
            "correct_answer": "42",
            "explanation": "25와 17을 더하면 42입니다.",
            "source": "sample_file"
        },
        {
            "id": "math_sample_2",
            "category": "math",
            "difficulty": "medium",
            "question": "한 원의 반지름이 7cm일 때, 둘레는? (π=3.14)",
            "correct_answer": "43.96",
            "explanation": "원의 둘레 = 2πr = 2 × 3.14 × 7 = 43.96 cm",
            "source": "sample_file"
        }
    ]

    # 논리 문제 샘플
    logic_sample = [
        {
            "id": "logic_sample_1",
            "category": "logic",
            "difficulty": "medium",
            "question": "A는 B보다 크고, B는 C보다 크다. C는 D보다 크다면, 가장 큰 것은?",
            "correct_answer": "A",
            "explanation": "A > B > C > D 순서이므로 A가 가장 큽니다.",
            "source": "sample_file"
        }
    ]

    # 파일 저장
    import json

    samples = {
        "math_samples.json": math_sample,
        "logic_samples.json": logic_sample
    }

    for filename, data in samples.items():
        file_path = data_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"샘플 파일 생성: {file_path}")

    return list(samples.keys())


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="LLM 추론 평가 시스템 샘플 데이터 로더")

    parser.add_argument(
        "--config", "-c",
        default="config/db_config.json",
        help="데이터베이스 설정 파일 경로"
    )

    parser.add_argument(
        "--basic", "-b",
        action="store_true",
        help="기본 샘플 데이터만 로드"
    )

    parser.add_argument(
        "--external", "-e",
        nargs="*",
        choices=["gsm8k", "arc", "hellaswag", "korean", "all"],
        help="외부 데이터셋 샘플 로드 (기본: all)"
    )

    parser.add_argument(
        "--file", "-f",
        help="특정 파일에서 데이터 로드"
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        help="파일 형식 지정"
    )

    parser.add_argument(
        "--demo-set",
        type=int,
        metavar="SIZE",
        help="데모용 평가 세트 생성 (크기 지정)"
    )

    parser.add_argument(
        "--stats", "-s",
        action="store_true",
        help="데이터 통계만 조회"
    )

    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="데이터 무결성 검증"
    )

    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="데이터 정리 수행"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 변경 없이 시뮬레이션만 수행"
    )

    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="샘플 데이터 파일 생성"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="배치 처리 크기 (기본: 1000)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력"
    )

    args = parser.parse_args()

    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 로깅 시스템 설정
    setup_application_logging({
        'log_level': 'DEBUG' if args.verbose else 'INFO',
        'log_format': 'simple',
        'enable_console': True
    })

    try:
        # 샘플 파일 생성
        if args.create_samples:
            created_files = create_sample_datasets()
            logger.info(f"샘플 데이터 파일 생성 완료: {len(created_files)}개")
            for filename in created_files:
                logger.info(f"  - data/samples/{filename}")
            return 0

        # 설정 파일 확인
        if not os.path.exists(args.config):
            logger.error(f"설정 파일이 존재하지 않습니다: {args.config}")
            logger.info("먼저 'python scripts/init_database.py --create-config' 명령을 실행해주세요.")
            return 1

        # 데이터베이스 설정 로드
        logger.info(f"설정 파일 로드: {args.config}")
        db_config = load_db_config_from_file(args.config)

        # 연결 테스트
        if not db_config.test_connection():
            logger.error("데이터베이스 연결 실패")
            return 1

        # 데이터 로더 생성
        db_manager = DatabaseManager(db_config)
        loader = SampleDataLoader(db_manager)

        # 데이터 통계만 조회
        if args.stats:
            stats = loader.get_data_statistics()
            print("\n" + "=" * 50)
            print("데이터베이스 통계")
            print("=" * 50)
            print(f"총 데이터 수: {stats.get('total_count', 0):,}개")
            print(f"카테고리 수: {len(stats.get('category_counts', {})):,}개")
            print(f"난이도 수: {len(stats.get('difficulty_counts', {})):,}개")
            print(f"소스 수: {len(stats.get('source_counts', {})):,}개")

            if stats.get('category_counts'):
                print("\n카테고리별 분포:")
                for category, count in stats['category_counts'].items():
                    print(f"  {category}: {count:,}개")

            if stats.get('difficulty_counts'):
                print("\n난이도별 분포:")
                for difficulty, count in stats['difficulty_counts'].items():
                    print(f"  {difficulty}: {count:,}개")

            print("=" * 50)
            return 0

        # 데이터 무결성 검증
        if args.validate:
            validation_result = loader.validate_data_integrity()

            print("\n" + "=" * 50)
            print("데이터 무결성 검증 결과")
            print("=" * 50)
            print(f"총 레코드: {validation_result.get('total_records', 0):,}개")
            print(f"카테고리: {validation_result.get('categories', 0)}개")
            print(f"난이도: {validation_result.get('difficulties', 0)}개")
            print(f"소스: {validation_result.get('sources', 0)}개")
            print(f"품질 이슈: {validation_result.get('quality_issues', 0)}개")
            print(f"중복 데이터: {validation_result.get('duplicates', 0)}개")
            print(f"검증 결과: {'✅ 통과' if validation_result.get('validation_passed') else '❌ 실패'}")
            print("=" * 50)
            return 0 if validation_result.get('validation_passed') else 1

        # 데이터 정리
        if args.cleanup:
            cleanup_results = loader.cleanup_data(dry_run=args.dry_run)

            print("\n" + "=" * 50)
            print(f"데이터 정리 {'시뮬레이션' if args.dry_run else '실행'} 결과")
            print("=" * 50)
            print(f"중복 데이터 제거: {cleanup_results.get('duplicates_removed', 0)}개")
            print(f"무효 데이터 제거: {cleanup_results.get('invalid_removed', 0)}개")
            print(f"오래된 데이터 제거: {cleanup_results.get('old_data_removed', 0)}개")
            print(f"총 정리: {sum(cleanup_results.values())}개")
            print("=" * 50)
            return 0

        # 메인 데이터 로딩 프로세스
        logger.info("=" * 50)
        logger.info("샘플 데이터 로딩 시작")
        logger.info("=" * 50)

        total_loaded = 0

        # 특정 파일 로드
        if args.file:
            count = loader.load_from_file(args.file, args.format)
            total_loaded += count

        # 기본 샘플 데이터 로드
        if args.basic or (not args.external and not args.file):
            count = loader.load_basic_samples()
            total_loaded += count

        # 외부 데이터셋 로드
        if args.external is not None:
            if 'all' in args.external or not args.external:
                sources = ['gsm8k', 'arc', 'hellaswag', 'korean']
            else:
                sources = args.external

            external_results = loader.load_external_samples(sources)
            total_loaded += sum(external_results.values())

        # 데모 평가 세트 생성
        if args.demo_set:
            demo_count = loader.create_demo_evaluation_set(args.demo_set)
            logger.info(f"데모 평가 세트: {demo_count}개")

        # 최종 통계
        final_stats = loader.get_data_statistics()

        logger.info("=" * 50)
        logger.info("샘플 데이터 로딩 완료")
        logger.info("=" * 50)
        logger.info(f"이번 세션에서 로드된 데이터: {total_loaded:,}개")
        logger.info(f"총 데이터베이스 크기: {final_stats.get('total_count', 0):,}개")

        if final_stats.get('category_counts'):
            logger.info("카테고리별 분포:")
            for category, count in final_stats['category_counts'].items():
                logger.info(f"  {category}: {count:,}개")

        logger.info("=" * 50)
        logger.info("다음 단계:")
        logger.info("1. 데이터 검증: python scripts/load_sample_data.py --validate")
        logger.info("2. 시스템 실행: python main.py")
        logger.info("3. 평가 실행: python examples/basic_usage.py")

        return 0

    except KeyboardInterrupt:
        logger.info("사용자에 의해 중단되었습니다.")
        return 1
    except Exception as e:
        logger.error(f"예상하지 못한 오류: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # 리소스 정리
        try:
            if 'db_config' in locals():
                db_config.close_pool()
        except:
            pass


if __name__ == "__main__":
    exit(main())