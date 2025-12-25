"""
Profile Store: 3-Tier Memory Architecture

Manages patient profile information across three tiers:
- Tier 1: Session Memory (short-term, current conversation)
- Tier 2: Profile Memory (medium-term, weighted by frequency)
- Tier 3: Long-term Memory (persistent, historical)
"""

from typing import Dict, Any, List
import json
from datetime import datetime


class ProfileStore:
    """
    3-Tier memory architecture for patient profile management.

    Stores and retrieves patient information with temporal weighting.
    """

    def __init__(self):
        """Initialize the three-tier memory structure."""
        self.session_memory: Dict[str, Any] = {}      # Tier 1: Current session
        self.profile_memory: Dict[str, Dict] = {}     # Tier 2: Weighted profile
        self.longterm_memory: Dict[str, List] = {}    # Tier 3: Historical data

        self.last_update = datetime.now()
        self.update_count = 0

    def update_slots(self, slot_out: Dict[str, Any]) -> None:
        """
        Update session memory with newly extracted slots.

        Args:
            slot_out: Dictionary of extracted slots from current turn
        """
        if not slot_out:
            return

        # Update session memory
        self.session_memory.update(slot_out)
        self.update_count += 1

        # Promote session data to profile memory with weights
        for key, value in slot_out.items():
            if key not in self.profile_memory:
                self.profile_memory[key] = {
                    'value': value,
                    'weight': 1.0,
                    'count': 1,
                    'last_updated': datetime.now().isoformat()
                }
            else:
                # Update existing profile entry
                self.profile_memory[key]['value'] = value
                self.profile_memory[key]['weight'] = min(
                    self.profile_memory[key]['weight'] + 0.5,
                    10.0  # Max weight
                )
                self.profile_memory[key]['count'] += 1
                self.profile_memory[key]['last_updated'] = datetime.now().isoformat()

        self.last_update = datetime.now()

    def apply_temporal_weights(self) -> None:
        """
        Apply temporal decay to profile memory weights.

        Reduces weights over time to prioritize recent information.
        """
        decay_factor = 0.9

        for key in self.profile_memory:
            self.profile_memory[key]['weight'] *= decay_factor

            # Remove entries with very low weight
            if self.profile_memory[key]['weight'] < 0.1:
                # Move to long-term memory before removal
                if key not in self.longterm_memory:
                    self.longterm_memory[key] = []
                self.longterm_memory[key].append({
                    'value': self.profile_memory[key]['value'],
                    'archived_at': datetime.now().isoformat(),
                    'final_weight': self.profile_memory[key]['weight']
                })

    def get_profile_summary(self) -> str:
        """
        Generate a human-readable summary of the patient profile.

        Returns:
            String summary of profile information
        """
        if not self.profile_memory and not self.session_memory:
            return ""

        summary_parts = []

        # Sort by weight (most important first)
        sorted_profile = sorted(
            self.profile_memory.items(),
            key=lambda x: x[1]['weight'],
            reverse=True
        )

        for key, data in sorted_profile:
            if data['weight'] > 0.3:  # Only include significant entries
                value = data['value']
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                summary_parts.append(f"{key}: {value}")

        # Include recent session data
        for key, value in self.session_memory.items():
            if key not in self.profile_memory:
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, ensure_ascii=False)
                summary_parts.append(f"{key}: {value} (new)")

        return "\n".join(summary_parts)

    def get_slot_value(self, key: str, default=None) -> Any:
        """
        Retrieve a slot value from memory tiers.

        Args:
            key: Slot name
            default: Default value if not found

        Returns:
            Slot value from the highest priority tier
        """
        # Check session first
        if key in self.session_memory:
            return self.session_memory[key]

        # Then profile
        if key in self.profile_memory:
            return self.profile_memory[key]['value']

        # Finally long-term
        if key in self.longterm_memory and self.longterm_memory[key]:
            return self.longterm_memory[key][-1]['value']

        return default

    def clear_session(self) -> None:
        """Clear session memory (e.g., at end of conversation)."""
        self.session_memory.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Export profile store state as dictionary."""
        return {
            'session_memory': self.session_memory,
            'profile_memory': self.profile_memory,
            'longterm_memory': self.longterm_memory,
            'update_count': self.update_count,
            'last_update': self.last_update.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfileStore':
        """Restore profile store from dictionary."""
        store = cls()
        store.session_memory = data.get('session_memory', {})
        store.profile_memory = data.get('profile_memory', {})
        store.longterm_memory = data.get('longterm_memory', {})
        store.update_count = data.get('update_count', 0)

        last_update_str = data.get('last_update')
        if last_update_str:
            store.last_update = datetime.fromisoformat(last_update_str)

        return store