from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class AgentConfig:
    """Configuration constants used by all agents"""
    # Price constraints
    MIN_PRICE: float = 8.0
    MAX_PRICE: float = 15.0
    
    # Production constraints  
    MIN_PRODUCTION: int = 10
    MAX_PRODUCTION: int = 150
    
    # Marketing constraints
    MIN_MARKETING: int = 0
    MAX_MARKETING: int = 2000
    
    # Cost parameters
    UNIT_PRODUCTION_COST: float = 3.0
    UNIT_HOLDING_COST: float = 0.5
    TARGET_PROFIT_MARGIN: float = 0.35
    
    # FIXED: Added the missing inventory parameter
    DESIRED_INVENTORY_WEEKS: float = 2.5

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.decision_history = []
        self.current_context = {}
        self.performance_metrics = {}
    
    @abstractmethod
    async def make_decision(self, context: Dict[str, Any], framework_context: str = "") -> Dict[str, Any]:
        pass
    
    def add_decision_to_history(self, decision: Dict[str, Any]) -> None:
        self.decision_history.append(decision)
        if len(self.decision_history) > 20:
            self.decision_history = self.decision_history[-20:]
    
    def get_agent_stats(self) -> Dict[str, Any]:
        return {
            'agent_type': self.agent_type,
            'decisions_made': len(self.decision_history),
            'last_decision': self.decision_history[-1] if self.decision_history else None,
            'average_confidence': self._calculate_average_confidence(),
            'decision_distribution': self._analyze_decision_patterns()
        }
    
    def _calculate_average_confidence(self) -> float:
        if not self.decision_history:
            return 0.0
        confidences = [decision.get('confidence', 0.5) for decision in self.decision_history]
        return sum(confidences) / len(confidences)
    
    def _analyze_decision_patterns(self) -> Dict[str, Any]:
        if not self.decision_history:
            return {'status': 'no_decisions_to_analyze'}
        
        ai_enhanced_count = len([d for d in self.decision_history if d.get('ai_enhanced', False)])
        rule_based_count = len(self.decision_history) - ai_enhanced_count
        
        return {
            'total_decisions': len(self.decision_history),
            'ai_enhanced_decisions': ai_enhanced_count,
            'rule_based_decisions': rule_based_count,
            'ai_success_rate': ai_enhanced_count / len(self.decision_history),
        }
    
    def validate_decision_bounds(self, decision_type: str, value: float) -> float:
        if decision_type == "price":
            return max(AgentConfig.MIN_PRICE, min(AgentConfig.MAX_PRICE, value))
        elif decision_type == "production":
            return max(AgentConfig.MIN_PRODUCTION, min(AgentConfig.MAX_PRODUCTION, int(value)))
        elif decision_type == "marketing":
            return max(AgentConfig.MIN_MARKETING, min(AgentConfig.MAX_MARKETING, int(value)))
        else:
            return value