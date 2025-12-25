"""
Hierarchical Memory System: Multi-tier memory architecture

Implements a three-tier memory hierarchy:
1. Working Memory: Recent conversation turns (limited capacity)
2. Compressing Memory: Compressed summaries of older turns
3. Semantic Memory: Extracted long-term facts (conditions, medications, allergies)
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    turn_id: int
    user_query: str
    agent_response: str
    extracted_slots: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CompressedMemory:
    """Compressed summary of multiple conversation turns."""
    summary: str
    turn_range: tuple  # (start_turn_id, end_turn_id)
    key_facts: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class SemanticMemory:
    """
    Long-term semantic memory storage.

    Stores extracted medical facts in structured categories.
    """

    def __init__(self):
        """Initialize semantic memory structures."""
        self.chronic_conditions: List[Dict[str, Any]] = []
        self.chronic_medications: List[Dict[str, Any]] = []
        self.allergies: List[Dict[str, Any]] = []
        self.procedures: List[Dict[str, Any]] = []
        self.other_facts: Dict[str, Any] = {}

    def add_chronic_condition(self, condition: str, metadata: Optional[Dict] = None) -> None:
        """Add a chronic condition."""
        entry = {
            'condition': condition,
            'added_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        # Avoid duplicates
        if not any(c['condition'] == condition for c in self.chronic_conditions):
            self.chronic_conditions.append(entry)

    def add_chronic_medication(self, medication: str, metadata: Optional[Dict] = None) -> None:
        """Add a chronic medication."""
        entry = {
            'medication': medication,
            'added_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        if not any(m['medication'] == medication for m in self.chronic_medications):
            self.chronic_medications.append(entry)

    def add_allergy(self, allergy: str, metadata: Optional[Dict] = None) -> None:
        """Add an allergy."""
        entry = {
            'allergy': allergy,
            'added_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        if not any(a['allergy'] == allergy for a in self.allergies):
            self.allergies.append(entry)

    def extract_from_slots(self, slots: Dict[str, Any]) -> None:
        """
        Extract semantic facts from slot dictionary.

        Args:
            slots: Extracted slot dictionary
        """
        # Extract chronic conditions
        if 'chronic_conditions' in slots:
            conditions = slots['chronic_conditions']
            if isinstance(conditions, list):
                for cond in conditions:
                    if isinstance(cond, str):
                        self.add_chronic_condition(cond)
                    elif isinstance(cond, dict) and 'name' in cond:
                        self.add_chronic_condition(cond['name'], metadata=cond)

        # Extract medications
        if 'medications' in slots:
            meds = slots['medications']
            if isinstance(meds, list):
                for med in meds:
                    if isinstance(med, str):
                        self.add_chronic_medication(med)
                    elif isinstance(med, dict) and 'name' in med:
                        self.add_chronic_medication(med['name'], metadata=med)

        # Extract allergies
        if 'allergies' in slots:
            allergies = slots['allergies']
            if isinstance(allergies, list):
                for allergy in allergies:
                    if isinstance(allergy, str):
                        self.add_allergy(allergy)
                    elif isinstance(allergy, dict) and 'name' in allergy:
                        self.add_allergy(allergy['name'], metadata=allergy)

        # Store other facts
        for key, value in slots.items():
            if key not in ['chronic_conditions', 'medications', 'allergies']:
                self.other_facts[key] = value

    def get_summary(self) -> str:
        """Generate a summary of semantic memory."""
        parts = []

        if self.chronic_conditions:
            conds = [c['condition'] for c in self.chronic_conditions]
            parts.append(f"만성 질환: {', '.join(conds)}")

        if self.chronic_medications:
            meds = [m['medication'] for m in self.chronic_medications]
            parts.append(f"복용 중인 약물: {', '.join(meds)}")

        if self.allergies:
            allergies = [a['allergy'] for a in self.allergies]
            parts.append(f"알레르기: {', '.join(allergies)}")

        return "; ".join(parts) if parts else ""


class HierarchicalMemorySystem:
    """
    Hierarchical memory system with three tiers.

    Manages conversation memory across working, compressing, and semantic tiers.
    """

    def __init__(
        self,
        user_id: str,
        llm_client: Any = None,
        medcat_adapter: Any = None,
        feature_flags: Optional[Dict] = None,
        working_capacity: int = 5,
        compression_threshold: int = 5
    ):
        """
        Initialize hierarchical memory system.

        Args:
            user_id: User/patient identifier
            llm_client: LLM client for compression (optional)
            medcat_adapter: MedCAT adapter for entity extraction (optional)
            feature_flags: Feature flags configuration
            working_capacity: Maximum turns in working memory
            compression_threshold: Turns before compression
        """
        self.user_id = user_id
        self.llm_client = llm_client
        self.medcat_adapter = medcat_adapter
        self.feature_flags = feature_flags or {}

        self.working_capacity = working_capacity
        self.compression_threshold = compression_threshold

        # Three-tier memory
        self.working_memory: List[ConversationTurn] = []
        self.compressing_memory: List[CompressedMemory] = []
        self.semantic_memory = SemanticMemory()

        self.total_turns = 0

    def add_turn(
        self,
        user_query: str,
        agent_response: str,
        extracted_slots: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a conversation turn to memory.

        Args:
            user_query: User's query
            agent_response: Agent's response
            extracted_slots: Extracted slot information
        """
        self.total_turns += 1

        turn = ConversationTurn(
            turn_id=self.total_turns,
            user_query=user_query,
            agent_response=agent_response,
            extracted_slots=extracted_slots or {},
            timestamp=datetime.now()
        )

        # Add to working memory
        self.working_memory.append(turn)

        # Extract semantic facts
        if extracted_slots:
            self.semantic_memory.extract_from_slots(extracted_slots)

        # Compress if needed
        if len(self.working_memory) > self.working_capacity:
            self._compress_oldest_turns()

    def _compress_oldest_turns(self) -> None:
        """
        Compress oldest turns from working memory to compressing memory.

        Uses LLM to generate a summary if available, otherwise uses simple concatenation.
        """
        # Determine how many turns to compress
        num_to_compress = len(self.working_memory) - self.working_capacity + 1

        if num_to_compress <= 0:
            return

        # Take oldest turns
        turns_to_compress = self.working_memory[:num_to_compress]
        self.working_memory = self.working_memory[num_to_compress:]

        # Generate summary
        if self.llm_client:
            summary = self._llm_compress(turns_to_compress)
        else:
            summary = self._simple_compress(turns_to_compress)

        # Extract key facts
        key_facts = self._extract_key_facts(turns_to_compress)

        compressed = CompressedMemory(
            summary=summary,
            turn_range=(turns_to_compress[0].turn_id, turns_to_compress[-1].turn_id),
            key_facts=key_facts,
            timestamp=datetime.now()
        )

        self.compressing_memory.append(compressed)

    def _llm_compress(self, turns: List[ConversationTurn]) -> str:
        """
        Use LLM to compress conversation turns.

        Args:
            turns: List of conversation turns to compress

        Returns:
            Compressed summary
        """
        # Build conversation text
        conversation = []
        for turn in turns:
            conversation.append(f"User: {turn.user_query}")
            conversation.append(f"Agent: {turn.agent_response}")

        conversation_text = "\n".join(conversation)

        # Create compression prompt
        prompt = f"""다음 대화를 간결하게 요약해주세요. 핵심 정보만 포함하세요:

{conversation_text}

요약:"""

        try:
            # Call LLM (simplified - actual implementation depends on llm_client interface)
            if hasattr(self.llm_client, 'generate'):
                summary = self.llm_client.generate(prompt)
            else:
                summary = self._simple_compress(turns)
        except Exception as e:
            print(f"[WARNING] LLM compression failed: {e}")
            summary = self._simple_compress(turns)

        return summary

    def _simple_compress(self, turns: List[ConversationTurn]) -> str:
        """
        Simple compression without LLM.

        Args:
            turns: List of conversation turns

        Returns:
            Simple concatenated summary
        """
        summaries = []
        for turn in turns:
            # Take first 100 chars of query and response
            query_snippet = turn.user_query[:100]
            response_snippet = turn.agent_response[:100]
            summaries.append(f"Turn {turn.turn_id}: Q: {query_snippet}... A: {response_snippet}...")

        return "; ".join(summaries)

    def _extract_key_facts(self, turns: List[ConversationTurn]) -> List[str]:
        """
        Extract key facts from conversation turns.

        Args:
            turns: List of conversation turns

        Returns:
            List of key fact strings
        """
        facts = []

        for turn in turns:
            # Extract from slots
            for key, value in turn.extracted_slots.items():
                if value and key not in ['session_id', 'timestamp']:
                    facts.append(f"{key}: {value}")

        return facts

    def get_working_context(self) -> str:
        """
        Get recent conversation context from working memory.

        Returns:
            Formatted context string
        """
        if not self.working_memory:
            return ""

        lines = []
        for turn in self.working_memory:
            lines.append(f"User: {turn.user_query}")
            lines.append(f"Agent: {turn.agent_response}")

        return "\n".join(lines)

    def get_full_context(self) -> str:
        """
        Get full conversation context including compressed memory.

        Returns:
            Formatted full context
        """
        parts = []

        # Add compressed memory summaries
        if self.compressing_memory:
            parts.append("=== Earlier Conversation ===")
            for comp in self.compressing_memory:
                parts.append(f"Turns {comp.turn_range[0]}-{comp.turn_range[1]}: {comp.summary}")

        # Add working memory
        if self.working_memory:
            parts.append("=== Recent Conversation ===")
            parts.append(self.get_working_context())

        # Add semantic memory
        semantic_summary = self.semantic_memory.get_summary()
        if semantic_summary:
            parts.append("=== Patient Profile ===")
            parts.append(semantic_summary)

        return "\n\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Export memory state as dictionary."""
        return {
            'user_id': self.user_id,
            'total_turns': self.total_turns,
            'working_memory': [
                {
                    'turn_id': t.turn_id,
                    'user_query': t.user_query,
                    'agent_response': t.agent_response,
                    'extracted_slots': t.extracted_slots,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in self.working_memory
            ],
            'compressing_memory': [
                {
                    'summary': c.summary,
                    'turn_range': c.turn_range,
                    'key_facts': c.key_facts,
                    'timestamp': c.timestamp.isoformat()
                }
                for c in self.compressing_memory
            ],
            'semantic_memory': {
                'chronic_conditions': self.semantic_memory.chronic_conditions,
                'chronic_medications': self.semantic_memory.chronic_medications,
                'allergies': self.semantic_memory.allergies,
                'procedures': self.semantic_memory.procedures,
                'other_facts': self.semantic_memory.other_facts
            }
        }