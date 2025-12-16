"""
Ablation Study 메트릭 수집 시스템

이 모듈은 Active Retrieval의 효과를 정량적으로 측정하기 위한
메트릭을 수집하고 분석합니다.

사용 예시:
    # 실험 시작
    metrics = AblationMetrics(experiment_name="active_retrieval_on")

    # 쿼리 실행
    metrics.record_query(state)

    # 결과 저장
    metrics.save_results()

    # 비교 분석
    compare_experiments("active_retrieval_on", "active_retrieval_off")
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class QueryMetrics:
    """단일 쿼리의 메트릭"""
    query_id: str
    query_text: str
    timestamp: str

    # Active Retrieval 관련
    needs_retrieval: bool
    dynamic_k: Optional[int]
    query_complexity: str
    classification_time_ms: float

    # 검색 관련
    retrieval_executed: bool
    retrieval_time_ms: float
    num_docs_retrieved: int

    # 생성 관련
    generation_time_ms: float
    answer_length: int

    # 품질 관련
    quality_score: float
    needs_refine: bool
    iteration_count: int

    # 비용 관련
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float

    # 전체 레이턴시
    total_latency_ms: float

    # Context Compression 관련 (선택적)
    compression_applied: bool = False
    compression_strategy: Optional[str] = None
    original_doc_tokens: Optional[int] = None
    compressed_doc_tokens: Optional[int] = None
    compression_ratio: Optional[float] = None
    tokens_saved_by_compression: Optional[int] = None

    # Hierarchical Memory 관련 (선택적)
    hierarchical_memory_enabled: bool = False
    total_turns: Optional[int] = None
    working_memory_size: Optional[int] = None
    compressed_memory_count: Optional[int] = None
    semantic_conditions_count: Optional[int] = None
    semantic_medications_count: Optional[int] = None
    tier_retrieval_time_ms: Optional[float] = None


class AblationMetrics:
    """
    Ablation Study 메트릭 수집기

    기능:
    1. 쿼리별 메트릭 수집
    2. 통계 계산 (평균, 표준편차 등)
    3. JSON 저장 및 로드
    4. 실험 간 비교
    """

    def __init__(
        self,
        experiment_name: str,
        save_dir: str = "experiments/ablation"
    ):
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 메트릭 저장소
        self.query_metrics: List[QueryMetrics] = []

        # 시작 시간
        self.experiment_start = datetime.now().isoformat()

        # 집계 통계
        self.aggregate_stats = {}

    def record_query(
        self,
        state: Dict[str, Any],
        start_time: float,
        end_time: float
    ) -> QueryMetrics:
        """
        쿼리 실행 결과를 기록

        Args:
            state: 최종 AgentState
            start_time: 쿼리 시작 시간
            end_time: 쿼리 종료 시간

        Returns:
            QueryMetrics 객체
        """
        # 기본 정보
        query_id = f"{self.experiment_name}_{len(self.query_metrics) + 1}"
        total_latency = (end_time - start_time) * 1000  # ms

        # Active Retrieval 메트릭
        needs_retrieval = state.get('needs_retrieval', True)
        dynamic_k = state.get('dynamic_k')
        query_complexity = state.get('query_complexity', 'unknown')
        classification_time = state.get('classification_time_ms', 0.0)

        # 검색 메트릭
        retrieved_docs = state.get('retrieved_docs', [])
        retrieval_executed = len(retrieved_docs) > 0
        num_docs = len(retrieved_docs)

        # 생성 메트릭
        answer = state.get('answer', '')
        answer_length = len(answer)

        # 품질 메트릭
        quality_score = state.get('quality_score', 0.0)
        iteration_count = state.get('iteration_count', 0)

        # 토큰 및 비용 추정
        token_plan = state.get('token_plan', {})
        total_tokens = self._estimate_total_tokens(state)
        prompt_tokens = total_tokens - answer_length // 4  # 대략적 추정
        completion_tokens = answer_length // 4
        estimated_cost = self._estimate_cost(prompt_tokens, completion_tokens)

        # Context Compression 메트릭
        compression_stats = state.get('compression_stats', {})
        compression_applied = compression_stats.get('compression_applied', False)
        compression_strategy = compression_stats.get('strategy') if compression_applied else None
        original_doc_tokens = compression_stats.get('original_tokens') if compression_applied else None
        compressed_doc_tokens = compression_stats.get('compressed_tokens') if compression_applied else None
        compression_ratio = compression_stats.get('compression_ratio') if compression_applied else None
        tokens_saved = compression_stats.get('tokens_saved') if compression_applied else None

        # Hierarchical Memory 메트릭
        hierarchical_memory_stats = state.get('hierarchical_memory_stats', {})
        hierarchical_memory_enabled = bool(hierarchical_memory_stats)
        total_turns = hierarchical_memory_stats.get('total_turns') if hierarchical_memory_enabled else None
        working_memory_size = hierarchical_memory_stats.get('working_memory_size') if hierarchical_memory_enabled else None
        compressed_memory_count = hierarchical_memory_stats.get('compressed_memory_count') if hierarchical_memory_enabled else None
        semantic_conditions_count = hierarchical_memory_stats.get('chronic_conditions_count') if hierarchical_memory_enabled else None
        semantic_medications_count = hierarchical_memory_stats.get('chronic_medications_count') if hierarchical_memory_enabled else None
        # 티어 검색 시간은 근사치 (assemble_context 시간의 일부)
        tier_retrieval_time = 0.0  # TODO: 정확한 측정을 위해 타이머 추가 필요

        # 시간 분해 (근사치)
        retrieval_time = classification_time if retrieval_executed else 0.0
        generation_time = total_latency - retrieval_time - classification_time

        # QueryMetrics 생성
        metrics = QueryMetrics(
            query_id=query_id,
            query_text=state['user_text'],
            timestamp=datetime.now().isoformat(),
            needs_retrieval=needs_retrieval,
            dynamic_k=dynamic_k,
            query_complexity=query_complexity,
            classification_time_ms=classification_time,
            retrieval_executed=retrieval_executed,
            retrieval_time_ms=retrieval_time,
            num_docs_retrieved=num_docs,
            generation_time_ms=generation_time,
            answer_length=answer_length,
            quality_score=quality_score,
            needs_refine=iteration_count > 0,
            iteration_count=iteration_count,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimated_cost,
            total_latency_ms=total_latency,
            compression_applied=compression_applied,
            compression_strategy=compression_strategy,
            original_doc_tokens=original_doc_tokens,
            compressed_doc_tokens=compressed_doc_tokens,
            compression_ratio=compression_ratio,
            tokens_saved_by_compression=tokens_saved,
            hierarchical_memory_enabled=hierarchical_memory_enabled,
            total_turns=total_turns,
            working_memory_size=working_memory_size,
            compressed_memory_count=compressed_memory_count,
            semantic_conditions_count=semantic_conditions_count,
            semantic_medications_count=semantic_medications_count,
            tier_retrieval_time_ms=tier_retrieval_time
        )

        self.query_metrics.append(metrics)
        return metrics

    def _estimate_total_tokens(self, state: Dict[str, Any]) -> int:
        """총 토큰 수 추정"""
        from context.token_manager import TokenManager

        token_manager = TokenManager()

        # 시스템 프롬프트
        system_tokens = token_manager.count_tokens(state.get('system_prompt', ''))

        # 사용자 프롬프트
        user_tokens = token_manager.count_tokens(state.get('user_prompt', ''))

        # 컨텍스트
        context_tokens = token_manager.count_tokens(state.get('context_prompt', ''))

        # 답변
        answer_tokens = token_manager.count_tokens(state.get('answer', ''))

        return system_tokens + user_tokens + context_tokens + answer_tokens

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """비용 추정 (GPT-4o-mini 기준)"""
        # GPT-4o-mini: $0.150 / 1M input, $0.600 / 1M output
        input_cost = (prompt_tokens / 1_000_000) * 0.150
        output_cost = (completion_tokens / 1_000_000) * 0.600
        return input_cost + output_cost

    def calculate_statistics(self) -> Dict[str, Any]:
        """
        집계 통계 계산

        Returns:
            통계 딕셔너리
        """
        if not self.query_metrics:
            return {}

        import numpy as np

        # 리스트 변환
        latencies = [m.total_latency_ms for m in self.query_metrics]
        costs = [m.estimated_cost_usd for m in self.query_metrics]
        quality_scores = [m.quality_score for m in self.query_metrics]

        # 검색 관련
        retrieval_executed = [m.retrieval_executed for m in self.query_metrics]
        retrieval_skip_rate = 1 - (sum(retrieval_executed) / len(retrieval_executed))

        # 복잡도 분포
        complexities = [m.query_complexity for m in self.query_metrics]
        complexity_dist = {
            'simple': complexities.count('simple') / len(complexities),
            'moderate': complexities.count('moderate') / len(complexities),
            'complex': complexities.count('complex') / len(complexities)
        }

        stats = {
            # 기본 정보
            'experiment_name': self.experiment_name,
            'experiment_start': self.experiment_start,
            'total_queries': len(self.query_metrics),

            # 레이턴시
            'avg_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),

            # 비용
            'avg_cost_usd': np.mean(costs),
            'total_cost_usd': np.sum(costs),
            'std_cost_usd': np.std(costs),

            # 품질
            'avg_quality_score': np.mean(quality_scores),
            'median_quality_score': np.median(quality_scores),
            'std_quality_score': np.std(quality_scores),

            # 검색
            'retrieval_skip_rate': retrieval_skip_rate,
            'avg_docs_retrieved': np.mean([m.num_docs_retrieved for m in self.query_metrics]),

            # 복잡도
            'complexity_distribution': complexity_dist,

            # 반복
            'avg_iterations': np.mean([m.iteration_count for m in self.query_metrics]),
            'refine_rate': sum([m.needs_refine for m in self.query_metrics]) / len(self.query_metrics),

            # Context Compression
            'compression_applied_count': sum([m.compression_applied for m in self.query_metrics]),
            'compression_rate': sum([m.compression_applied for m in self.query_metrics]) / len(self.query_metrics),
            'avg_compression_ratio': np.mean([m.compression_ratio for m in self.query_metrics if m.compression_ratio is not None]) if any(m.compression_ratio for m in self.query_metrics) else 0.0,
            'total_tokens_saved_by_compression': sum([m.tokens_saved_by_compression for m in self.query_metrics if m.tokens_saved_by_compression is not None]),

            # Hierarchical Memory
            'hierarchical_memory_enabled_count': sum([m.hierarchical_memory_enabled for m in self.query_metrics]),
            'hierarchical_memory_enabled_rate': sum([m.hierarchical_memory_enabled for m in self.query_metrics]) / len(self.query_metrics),
            'avg_working_memory_size': np.mean([m.working_memory_size for m in self.query_metrics if m.working_memory_size is not None]) if any(m.working_memory_size for m in self.query_metrics) else 0.0,
            'avg_compressed_memory_count': np.mean([m.compressed_memory_count for m in self.query_metrics if m.compressed_memory_count is not None]) if any(m.compressed_memory_count for m in self.query_metrics) else 0.0,
            'avg_semantic_conditions': np.mean([m.semantic_conditions_count for m in self.query_metrics if m.semantic_conditions_count is not None]) if any(m.semantic_conditions_count for m in self.query_metrics) else 0.0,
            'avg_semantic_medications': np.mean([m.semantic_medications_count for m in self.query_metrics if m.semantic_medications_count is not None]) if any(m.semantic_medications_count for m in self.query_metrics) else 0.0
        }

        self.aggregate_stats = stats
        return stats

    def save_results(self, filename: Optional[str] = None):
        """
        결과를 JSON 파일로 저장

        Args:
            filename: 파일명 (기본값: {experiment_name}_{timestamp}.json)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.experiment_name}_{timestamp}.json"

        filepath = self.save_dir / filename

        # 통계 계산
        if not self.aggregate_stats:
            self.calculate_statistics()

        # 저장할 데이터
        data = {
            'metadata': {
                'experiment_name': self.experiment_name,
                'experiment_start': self.experiment_start,
                'experiment_end': datetime.now().isoformat(),
                'total_queries': len(self.query_metrics)
            },
            'aggregate_stats': self.aggregate_stats,
            'query_metrics': [asdict(m) for m in self.query_metrics]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[Metrics] Saved to {filepath}")
        return filepath

    @staticmethod
    def load_results(filepath: str) -> 'AblationMetrics':
        """
        JSON 파일에서 결과 로드

        Args:
            filepath: 파일 경로

        Returns:
            AblationMetrics 객체
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metrics = AblationMetrics(experiment_name=data['metadata']['experiment_name'])
        metrics.experiment_start = data['metadata']['experiment_start']
        metrics.aggregate_stats = data['aggregate_stats']

        # QueryMetrics 복원
        for qm_dict in data['query_metrics']:
            qm = QueryMetrics(**qm_dict)
            metrics.query_metrics.append(qm)

        return metrics


def compare_experiments(
    baseline_path: str,
    treatment_path: str,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    두 실험 결과를 비교

    Args:
        baseline_path: 베이스라인 실험 JSON 경로
        treatment_path: 처리 실험 JSON 경로
        output_path: 비교 결과 저장 경로 (선택)

    Returns:
        비교 결과 딕셔너리
    """
    # 로드
    baseline = AblationMetrics.load_results(baseline_path)
    treatment = AblationMetrics.load_results(treatment_path)

    # 통계 가져오기
    baseline_stats = baseline.aggregate_stats or baseline.calculate_statistics()
    treatment_stats = treatment.aggregate_stats or treatment.calculate_statistics()

    # 비교
    comparison = {
        'experiments': {
            'baseline': baseline.experiment_name,
            'treatment': treatment.experiment_name
        },
        'sample_sizes': {
            'baseline': len(baseline.query_metrics),
            'treatment': len(treatment.query_metrics)
        },
        'metrics_comparison': {}
    }

    # 주요 메트릭 비교
    metrics_to_compare = [
        'avg_latency_ms',
        'p95_latency_ms',
        'avg_cost_usd',
        'total_cost_usd',
        'avg_quality_score',
        'retrieval_skip_rate',
        'avg_docs_retrieved'
    ]

    for metric in metrics_to_compare:
        baseline_val = baseline_stats.get(metric, 0)
        treatment_val = treatment_stats.get(metric, 0)

        if baseline_val != 0:
            percent_change = ((treatment_val - baseline_val) / baseline_val) * 100
        else:
            percent_change = 0

        comparison['metrics_comparison'][metric] = {
            'baseline': baseline_val,
            'treatment': treatment_val,
            'absolute_change': treatment_val - baseline_val,
            'percent_change': percent_change
        }

    # 통계적 유의성 검정 (t-test)
    try:
        from scipy import stats

        baseline_latencies = [m.total_latency_ms for m in baseline.query_metrics]
        treatment_latencies = [m.total_latency_ms for m in treatment.query_metrics]

        t_stat, p_value = stats.ttest_ind(baseline_latencies, treatment_latencies)

        comparison['statistical_test'] = {
            'test': 'independent_t_test',
            'metric': 'total_latency_ms',
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    except ImportError:
        comparison['statistical_test'] = {
            'error': 'scipy not installed - cannot perform statistical test'
        }

    # 결과 저장
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        print(f"[Comparison] Saved to {output_path}")

    # 요약 출력
    print("\n" + "="*60)
    print("ABLATION STUDY COMPARISON")
    print("="*60)
    print(f"Baseline:  {baseline.experiment_name} (n={len(baseline.query_metrics)})")
    print(f"Treatment: {treatment.experiment_name} (n={len(treatment.query_metrics)})")
    print("-"*60)

    for metric, values in comparison['metrics_comparison'].items():
        print(f"{metric}:")
        print(f"  Baseline:  {values['baseline']:.4f}")
        print(f"  Treatment: {values['treatment']:.4f}")
        print(f"  Change:    {values['percent_change']:+.2f}%")

    if 'statistical_test' in comparison and 'p_value' in comparison['statistical_test']:
        p_val = comparison['statistical_test']['p_value']
        sig = "✓" if comparison['statistical_test']['significant'] else "✗"
        print(f"\nStatistical Significance: {sig} (p={p_val:.4f})")

    print("="*60 + "\n")

    return comparison


def generate_ablation_report(experiments_dir: str = "experiments/ablation"):
    """
    모든 실험 결과를 요약한 HTML 보고서 생성

    Args:
        experiments_dir: 실험 결과 디렉토리
    """
    import os

    experiments_path = Path(experiments_dir)
    if not experiments_path.exists():
        print(f"[Warning] {experiments_dir} not found")
        return

    # 모든 JSON 파일 찾기
    json_files = list(experiments_path.glob("*.json"))

    if not json_files:
        print(f"[Warning] No experiment results found in {experiments_dir}")
        return

    # 실험 로드
    experiments = []
    for filepath in json_files:
        try:
            exp = AblationMetrics.load_results(str(filepath))
            experiments.append(exp)
        except Exception as e:
            print(f"[Warning] Failed to load {filepath}: {e}")

    if not experiments:
        return

    # HTML 생성
    html = """
    <html>
    <head>
        <title>Active Retrieval Ablation Study Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .metric-good { color: green; font-weight: bold; }
            .metric-bad { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Active Retrieval Ablation Study Report</h1>
        <p>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

        <h2>Experiment Summary</h2>
        <table>
            <tr>
                <th>Experiment</th>
                <th>Queries</th>
                <th>Avg Latency (ms)</th>
                <th>Total Cost ($)</th>
                <th>Avg Quality</th>
                <th>Skip Rate (%)</th>
            </tr>
    """

    for exp in experiments:
        stats = exp.aggregate_stats or exp.calculate_statistics()
        html += f"""
            <tr>
                <td>{exp.experiment_name}</td>
                <td>{stats['total_queries']}</td>
                <td>{stats['avg_latency_ms']:.2f}</td>
                <td>{stats['total_cost_usd']:.4f}</td>
                <td>{stats['avg_quality_score']:.3f}</td>
                <td>{stats['retrieval_skip_rate']*100:.1f}</td>
            </tr>
        """

    html += """
        </table>
    </body>
    </html>
    """

    # 저장
    report_path = experiments_path / "ablation_report.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[Report] Generated {report_path}")
    return str(report_path)
