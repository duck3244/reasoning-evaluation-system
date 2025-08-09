"""
LLM 추론 성능 평가 시스템 메인
"""
import json
import logging
from typing import Dict, Any

from database_config import DatabaseConfig, DatabaseManager, load_db_config_from_file
from data_collector import ReasoningDatasetCollector
from sample_data_generator import SampleDataGenerator
from external_data_loader import ExternalDatasetLoader
from evaluation_system import ReasoningEvaluator, dummy_model_simple, dummy_model_random

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReasoningEvaluationSystem:
    """추론 평가 시스템 메인 클래스"""

    def __init__(self, db_config: DatabaseConfig):
        """시스템 초기화"""
        self.db_config = db_config
        self.db_manager = DatabaseManager(db_config)
        self.collector = ReasoningDatasetCollector(self.db_manager)
        self.sample_generator = SampleDataGenerator(self.collector)
        self.external_loader = ExternalDatasetLoader(self.collector)
        self.evaluator = ReasoningEvaluator(self.db_manager, self.collector)

        logger.info("추론 평가 시스템 초기화 완료")

    def setup_database(self):
        """데이터베이스 설정"""
        try:
            logger.info("데이터베이스 초기화 중...")
            self.db_manager.init_database()
            logger.info("데이터베이스 초기화 완료")
        except Exception as e:
            logger.error(f"데이터베이스 설정 실패: {e}")
            raise

    def load_sample_data(self) -> Dict[str, int]:
        """샘플 데이터 로드"""
        try:
            logger.info("샘플 데이터 로딩 시작...")

            # 기본 샘플 데이터 추가
            basic_count = self.sample_generator.add_all_sample_data()

            # 외부 데이터셋 샘플 추가
            external_counts = self.external_loader.load_all_samples()

            results = {
                'basic_samples': basic_count,
                **external_counts
            }

            total = sum(results.values())
            logger.info(f"총 {total}개의 샘플 데이터 로드 완료")

            return results

        except Exception as e:
            logger.error(f"샘플 데이터 로드 실패: {e}")
            return {}

    def run_evaluation(self, model_name: str = "test_model", test_size: int = 50) -> Dict[str, Any]:
        """평가 실행"""
        try:
            logger.info(f"모델 '{model_name}' 평가 시작...")

            # 평가 데이터셋 생성
            evaluation_set = self.evaluator.create_evaluation_set(test_size)

            if not evaluation_set:
                logger.error("평가 데이터셋이 비어있습니다.")
                return {}

            # 더미 모델로 평가 (실제 사용시 실제 모델 함수로 교체)
            results = self.evaluator.evaluate_model(
                model_name=model_name,
                evaluation_set=evaluation_set,
                model_function=dummy_model_simple
            )

            return results

        except Exception as e:
            logger.error(f"평가 실행 실패: {e}")
            return {}

    def get_system_statistics(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        try:
            # 데이터셋 통계
            dataset_stats = self.collector.get_statistics()

            # 모델 성능 통계 (예시)
            test_model_perf = self.evaluator.get_model_performance("test_model")

            return {
                'dataset_statistics': {
                    'total_questions': dataset_stats.total_count,
                    'by_category': dataset_stats.category_counts,
                    'by_difficulty': dataset_stats.difficulty_counts,
                    'by_source': dataset_stats.source_counts
                },
                'model_performance': {
                    'test_model': test_model_perf
                },
                'system_info': {
                    'database_type': 'Oracle',
                    'last_updated': dataset_stats.created_at
                }
            }

        except Exception as e:
            logger.error(f"통계 조회 실패: {e}")
            return {}

    def export_data(self, category: str = None, format_type: str = "json") -> bool:
        """데이터 내보내기"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reasoning_dataset_{category or 'all'}_{timestamp}.{format_type}"

            if format_type == "json":
                success = self.collector.export_to_json(filename, category)
            elif format_type == "csv":
                success = self.collector.export_to_csv(filename, category)
            else:
                logger.error(f"지원하지 않는 형식: {format_type}")
                return False

            if success:
                logger.info(f"데이터 내보내기 완료: {filename}")

            return success

        except Exception as e:
            logger.error(f"데이터 내보내기 실패: {e}")
            return False

    def import_data(self, file_path: str, format_type: str = None) -> int:
        """데이터 가져오기"""
        try:
            if format_type is None:
                # 파일 확장자로 형식 판단
                if file_path.endswith('.json'):
                    format_type = 'json'
                elif file_path.endswith('.csv'):
                    format_type = 'csv'
                else:
                    logger.error("지원하지 않는 파일 형식")
                    return 0

            if format_type == "json":
                count = self.collector.load_from_json(file_path)
            elif format_type == "csv":
                count = self.collector.load_from_csv(file_path)
            else:
                logger.error(f"지원하지 않는 형식: {format_type}")
                return 0

            logger.info(f"데이터 가져오기 완료: {count}개")
            return count

        except Exception as e:
            logger.error(f"데이터 가져오기 실패: {e}")
            return 0

    def cleanup(self):
        """시스템 정리"""
        try:
            self.db_config.close_pool()
            logger.info("시스템 정리 완료")
        except Exception as e:
            logger.error(f"시스템 정리 실패: {e}")


def create_sample_config():
    """샘플 설정 파일 생성"""
    config = {
        "username": "your_username",
        "password": "your_password",
        "dsn": "localhost:1521/XE",
        "pool_min": 1,
        "pool_max": 10,
        "pool_increment": 1
    }

    with open("db_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("샘플 설정 파일 'db_config.json'이 생성되었습니다.")
    print("실제 데이터베이스 정보로 수정해주세요.")


def main():
    """메인 실행 함수"""
    try:
        # 설정 파일 확인
        try:
            db_config = load_db_config_from_file("db_config.json")
        except FileNotFoundError:
            print("설정 파일이 없습니다. 샘플 설정 파일을 생성합니다.")
            create_sample_config()
            return

        # 시스템 초기화
        system = ReasoningEvaluationSystem(db_config)

        # 데이터베이스 설정
        system.setup_database()

        # 샘플 데이터 로드
        print("\n=== 샘플 데이터 로딩 ===")
        load_results = system.load_sample_data()
        for source, count in load_results.items():
            print(f"{source}: {count}개")

        # 시스템 통계 출력
        print("\n=== 시스템 통계 ===")
        stats = system.get_system_statistics()

        dataset_stats = stats.get('dataset_statistics', {})
        print(f"총 문제 수: {dataset_stats.get('total_questions', 0)}")
        print(f"카테고리별 분포: {dataset_stats.get('by_category', {})}")
        print(f"난이도별 분포: {dataset_stats.get('by_difficulty', {})}")

        # 평가 실행
        print("\n=== 모델 평가 실행 ===")
        eval_results = system.run_evaluation("test_model_v1", test_size=20)

        if eval_results:
            print(f"총 문제 수: {eval_results.get('total_questions', 0)}")
            print(f"정답 수: {eval_results.get('correct_answers', 0)}")
            print(f"정확도: {eval_results.get('accuracy', 0):.2%}")
            print(f"평균 실행 시간: {eval_results.get('average_execution_time', 0):.3f}초")

            # 카테고리별 정확도
            category_acc = eval_results.get('category_accuracy', {})
            if category_acc:
                print("\n카테고리별 정확도:")
                for category, stats in category_acc.items():
                    print(f"  {category}: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")

        # 평가용 데이터셋 내보내기
        print("\n=== 평가용 데이터셋 생성 ===")
        eval_count = system.evaluator.save_evaluation_format("llm_evaluation_set.json", test_size=50)
        print(f"평가용 데이터셋 생성: {eval_count}개 문제")

        # 데이터 내보내기 예시
        print("\n=== 데이터 내보내기 ===")
        system.export_data("math", "json")
        system.export_data(None, "csv")

        # 시스템 정리
        system.cleanup()

        print("\n모든 작업이 완료되었습니다!")

    except Exception as e:
        logger.error(f"메인 실행 오류: {e}")
        print(f"오류가 발생했습니다: {e}")


def demo_custom_evaluation():
    """커스텀 평가 데모"""
    try:
        # 실제 LLM 모델 함수 예시
        def my_custom_model(prompt: str) -> str:
            """커스텀 모델 함수 예시"""
            # 여기에 실제 LLM API 호출 코드를 작성
            # 예: OpenAI API, Anthropic API, 로컬 모델 등

            # 간단한 패턴 매칭 예시
            if "더하기" in prompt or "+" in prompt:
                return "수학 계산 결과"
            elif "논리" in prompt:
                return "예"
            else:
                return "답변"

        # 시스템 초기화
        db_config = load_db_config_from_file("db_config.json")
        system = ReasoningEvaluationSystem(db_config)

        # 커스텀 평가 실행
        evaluation_set = system.evaluator.create_evaluation_set(test_size=10)

        results = system.evaluator.evaluate_model(
            model_name="my_custom_model",
            evaluation_set=evaluation_set,
            model_function=my_custom_model,
            save_results=True
        )

        print("커스텀 모델 평가 결과:")
        print(f"정확도: {results.get('accuracy', 0):.2%}")

        # 모델 성능 기록 조회
        performance = system.evaluator.get_model_performance("my_custom_model")
        print(f"누적 평가 기록: {performance}")

        system.cleanup()

    except Exception as e:
        print(f"커스텀 평가 오류: {e}")


if __name__ == "__main__":
    import sys
    from datetime import datetime

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_custom_evaluation()
    elif len(sys.argv) > 1 and sys.argv[1] == "config":
        create_sample_config()
    else:
        main()