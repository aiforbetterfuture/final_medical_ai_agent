"""
멀티턴 실험 메인 러너
Basic RAG vs Agentic RAG 비교 실험
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import argparse

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.multiturn.patient_scenarios import PatientScenarioGenerator, PatientScenario
from experiments.multiturn.question_bank import QuestionBank
from experiments.multiturn.multiturn_simulator import (
    MultiTurnSimulator, 
    PatientResponseGenerator,
    MultiTurnDialogue
)
from experiments.multiturn.evaluation_rubric import EvaluationRubric, LLMJudge


class MultiTurnExperiment:
    """멀티턴 실험 관리"""
    
    def __init__(
        self,
        output_dir: str = "results/multiturn",
        protocol: str = "P6",
        seed: int = 42
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.protocol = protocol
        self.seed = seed
        
        # 컴포넌트 초기화
        self.question_bank = QuestionBank()
        self.response_generator = PatientResponseGenerator(temperature=0.0, seed=seed)
        self.simulator = MultiTurnSimulator(
            self.question_bank, 
            self.response_generator, 
            protocol=protocol
        )
        
        # 평가기
        self.rubric_basic = EvaluationRubric(model_type="basic")
        self.rubric_agentic = EvaluationRubric(model_type="agentic")
        
        # 결과 저장
        self.results = {
            "basic_rag": [],
            "agentic_rag": []
        }
    
    def generate_dialogues(
        self, 
        scenarios: List[PatientScenario],
        mode: str = "cooperative"
    ) -> List[MultiTurnDialogue]:
        """모든 시나리오에 대한 대화 생성"""
        dialogues = []
        
        for i, patient in enumerate(scenarios, 1):
            print(f"[{i}/{len(scenarios)}] 대화 생성 중: {patient.patient_id} ({patient.name})")
            
            dialogue = self.simulator.generate_dialogue(patient, mode=mode)
            dialogues.append(dialogue)
            
            # 중간 저장
            self._save_dialogue(dialogue, "dialogues")
        
        return dialogues
    
    def run_basic_rag(self, dialogues: List[MultiTurnDialogue]):
        """Basic RAG 실행 및 평가"""
        print("\n=== Basic RAG 실행 ===")
        
        for dialogue in dialogues:
            print(f"처리 중: {dialogue.dialogue_id}")
            
            dialogue_result = {
                "dialogue_id": dialogue.dialogue_id,
                "patient_id": dialogue.patient_id,
                "cohort": dialogue.cohort,
                "scenario_level": dialogue.scenario_level,
                "turns": []
            }
            
            for turn in dialogue.turns:
                # Basic RAG 답변 생성 (실제로는 RAG 시스템 호출)
                model_answer = self._call_basic_rag(turn)
                
                # 평가
                evaluation = self.rubric_basic.evaluate_turn(
                    turn_data=turn.to_dict(),
                    model_answer=model_answer,
                    gold_answer=turn.gold_answer
                )
                
                turn_result = {
                    "turn_idx": turn.turn_idx,
                    "turn_type": turn.turn_type,
                    "template_id": turn.template_id,
                    "question": turn.question,
                    "model_answer": model_answer,
                    "gold_answer": turn.gold_answer,
                    "evaluation": evaluation.to_dict()
                }
                
                dialogue_result["turns"].append(turn_result)
            
            self.results["basic_rag"].append(dialogue_result)
        
        # 결과 저장
        self._save_results("basic_rag")
    
    def run_agentic_rag(self, dialogues: List[MultiTurnDialogue]):
        """Agentic RAG 실행 및 평가"""
        print("\n=== Agentic RAG 실행 ===")
        
        for dialogue in dialogues:
            print(f"처리 중: {dialogue.dialogue_id}")
            
            dialogue_result = {
                "dialogue_id": dialogue.dialogue_id,
                "patient_id": dialogue.patient_id,
                "cohort": dialogue.cohort,
                "scenario_level": dialogue.scenario_level,
                "turns": []
            }
            
            for turn in dialogue.turns:
                # Agentic RAG 답변 생성 (실제로는 에이전트 시스템 호출)
                model_answer, retrieved_context = self._call_agentic_rag(turn)
                
                # 평가
                evaluation = self.rubric_agentic.evaluate_turn(
                    turn_data=turn.to_dict(),
                    model_answer=model_answer,
                    gold_answer=turn.gold_answer,
                    retrieved_context=retrieved_context
                )
                
                turn_result = {
                    "turn_idx": turn.turn_idx,
                    "turn_type": turn.turn_type,
                    "template_id": turn.template_id,
                    "question": turn.question,
                    "model_answer": model_answer,
                    "gold_answer": turn.gold_answer,
                    "retrieved_context": retrieved_context,
                    "evaluation": evaluation.to_dict()
                }
                
                dialogue_result["turns"].append(turn_result)
            
            self.results["agentic_rag"].append(dialogue_result)
        
        # 결과 저장
        self._save_results("agentic_rag")
    
    def _call_basic_rag(self, turn) -> str:
        """
        Basic RAG 시스템 호출 (더미 구현)
        실제로는 retrieval.hybrid_retriever + core.llm_client 사용
        """
        # TODO: 실제 Basic RAG 구현 연결
        # from retrieval.hybrid_retriever import HybridRetriever
        # from core.llm_client import LLMClient
        # 
        # retriever = HybridRetriever(...)
        # llm = LLMClient(...)
        # 
        # docs = retriever.retrieve(turn.question)
        # answer = llm.generate(prompt_with_docs)
        
        # 더미 답변
        return f"[Basic RAG 답변] {turn.question[:50]}에 대한 답변입니다."
    
    def _call_agentic_rag(self, turn) -> tuple[str, List[str]]:
        """
        Agentic RAG 시스템 호출 (더미 구현)
        실제로는 agent.graph 사용
        """
        # TODO: 실제 Agentic RAG 구현 연결
        # from agent.graph import create_graph
        # 
        # graph = create_graph()
        # result = graph.invoke({"query": turn.question, ...})
        # 
        # answer = result["final_answer"]
        # context = result["retrieved_docs"]
        
        # 더미 답변
        answer = f"[Agentic RAG 답변] {turn.question[:50]}에 대한 답변입니다. (근거 포함)"
        context = ["문서1: ...", "문서2: ..."]
        
        return answer, context
    
    def _save_dialogue(self, dialogue: MultiTurnDialogue, subdir: str):
        """대화 저장"""
        save_dir = self.output_dir / subdir
        save_dir.mkdir(exist_ok=True)
        
        filepath = save_dir / f"{dialogue.dialogue_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dialogue.to_dict(), f, ensure_ascii=False, indent=2)
    
    def _save_results(self, model_type: str):
        """결과 저장"""
        filepath = self.output_dir / f"{model_type}_results.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results[model_type], f, ensure_ascii=False, indent=2)
        
        print(f"결과 저장 완료: {filepath}")
    
    def analyze_results(self):
        """결과 분석 및 통계"""
        print("\n=== 결과 분석 ===")
        
        for model_type in ["basic_rag", "agentic_rag"]:
            print(f"\n[{model_type.upper()}]")
            
            results = self.results[model_type]
            if not results:
                print("  결과 없음")
                continue
            
            # 전체 턴 수집
            all_turns = []
            for dialogue_result in results:
                all_turns.extend(dialogue_result["turns"])
            
            # 평균 점수 계산
            total_scores = [t["evaluation"]["total_score"] for t in all_turns]
            weighted_scores = [t["evaluation"]["weighted_score"] for t in all_turns]
            
            print(f"  총 대화 수: {len(results)}")
            print(f"  총 턴 수: {len(all_turns)}")
            print(f"  평균 총점: {sum(total_scores) / len(total_scores):.2f}")
            print(f"  평균 가중 점수: {sum(weighted_scores) / len(weighted_scores):.2f}")
            
            # 턴 타입별 점수
            turn_type_scores = {}
            for turn in all_turns:
                turn_type = turn["turn_type"]
                if turn_type not in turn_type_scores:
                    turn_type_scores[turn_type] = []
                turn_type_scores[turn_type].append(turn["evaluation"]["weighted_score"])
            
            print("\n  턴 타입별 평균 가중 점수:")
            for turn_type, scores in sorted(turn_type_scores.items()):
                avg_score = sum(scores) / len(scores)
                print(f"    {turn_type}: {avg_score:.2f} (n={len(scores)})")
            
            # 서브스코어 평균
            subscore_totals = {}
            for turn in all_turns:
                for subscore in turn["evaluation"]["subscores"]:
                    metric = subscore["metric"]
                    if metric not in subscore_totals:
                        subscore_totals[metric] = []
                    subscore_totals[metric].append(subscore["score"])
            
            print("\n  서브스코어 평균:")
            for metric, scores in sorted(subscore_totals.items()):
                avg_score = sum(scores) / len(scores)
                print(f"    {metric}: {avg_score:.2f}/2.0")
        
        # 비교 분석
        if self.results["basic_rag"] and self.results["agentic_rag"]:
            print("\n=== Basic vs Agentic 비교 ===")
            
            basic_turns = []
            for d in self.results["basic_rag"]:
                basic_turns.extend(d["turns"])
            
            agentic_turns = []
            for d in self.results["agentic_rag"]:
                agentic_turns.extend(d["turns"])
            
            basic_avg = sum(t["evaluation"]["weighted_score"] for t in basic_turns) / len(basic_turns)
            agentic_avg = sum(t["evaluation"]["weighted_score"] for t in agentic_turns) / len(agentic_turns)
            
            print(f"Basic RAG 평균: {basic_avg:.2f}")
            print(f"Agentic RAG 평균: {agentic_avg:.2f}")
            print(f"차이: {agentic_avg - basic_avg:.2f} ({(agentic_avg - basic_avg) / basic_avg * 100:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="멀티턴 대화 실험 실행")
    parser.add_argument("--output-dir", default="results/multiturn", help="결과 저장 디렉토리")
    parser.add_argument("--protocol", default="P6", help="프로토콜 (P6)")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    parser.add_argument("--num-scenarios", type=int, default=13, help="시나리오 개수")
    parser.add_argument("--mode", default="cooperative", choices=["cooperative", "minimal", "noisy"],
                        help="환자 응답 모드")
    parser.add_argument("--skip-generation", action="store_true", help="대화 생성 스킵 (기존 대화 사용)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("멀티턴 대화 실험 시작")
    print("=" * 60)
    print(f"프로토콜: {args.protocol}")
    print(f"시드: {args.seed}")
    print(f"모드: {args.mode}")
    print(f"출력 디렉토리: {args.output_dir}")
    print()
    
    # 실험 초기화
    experiment = MultiTurnExperiment(
        output_dir=args.output_dir,
        protocol=args.protocol,
        seed=args.seed
    )
    
    # 환자 시나리오 생성
    print("환자 시나리오 생성 중...")
    generator = PatientScenarioGenerator()
    scenarios = generator.generate_all_scenarios()[:args.num_scenarios]
    
    print(f"총 {len(scenarios)}개 시나리오:")
    for cohort in ["Full", "No-Meds", "No-Trend"]:
        count = len([s for s in scenarios if s.cohort == cohort])
        print(f"  - {cohort}: {count}개")
    print()
    
    # 대화 생성
    if not args.skip_generation:
        dialogues = experiment.generate_dialogues(scenarios, mode=args.mode)
        print(f"\n총 {len(dialogues)}개 대화 생성 완료")
    else:
        print("대화 생성 스킵 (기존 대화 사용)")
        # TODO: 기존 대화 로드
        dialogues = []
    
    # Basic RAG 실행
    if dialogues:
        experiment.run_basic_rag(dialogues)
    
    # Agentic RAG 실행
    if dialogues:
        experiment.run_agentic_rag(dialogues)
    
    # 결과 분석
    experiment.analyze_results()
    
    print("\n" + "=" * 60)
    print("실험 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()

