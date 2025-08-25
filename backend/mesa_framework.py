# mesa_framework.py - Mesa Multi-Agent System with Gemini AI Integration

import asyncio
import json
import random
import numpy as np
from typing import Dict, Any, List, Optional
from mesa import Agent, Model
from mesa.time import RandomActivation
import google.generativeai as genai

class GameConfig:
    """Mesa-specific game configuration"""
    DESIRED_INVENTORY_WEEKS = 2.5
    PROFIT_MARGIN_TARGET = 0.35
    UNIT_PRODUCTION_COST = 3.0
    UNIT_HOLDING_COST = 0.5
    BASE_MARKET_PRICE = 10.0
    MIN_PRICE = 8.0
    MAX_PRICE = 15.0

class MesaGeminiHelper:
    """Gemini AI helper specifically for Mesa agents"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.enabled = False
        self.model = None
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.enabled = True
                print("✅ Mesa Gemini AI: Enabled")
            except Exception as e:
                print(f"❌ Mesa Gemini AI initialization failed: {e}")
        else:
            print("⚠️ Mesa Gemini AI: Disabled (no API key)")
    
    async def get_agent_decision(self, agent_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-enhanced decision for a specific Mesa agent"""
        if not self.enabled:
            return self._fallback_decision(agent_type, context)
        
        try:
            prompt = self._create_mesa_agent_prompt(agent_type, context)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate_content(prompt)
            )
            
            return self._parse_mesa_response(response.text, agent_type)
            
        except Exception as e:
            print(f"Mesa Gemini error for {agent_type} agent: {e}")
            return self._fallback_decision(agent_type, context)
    
    def _create_mesa_agent_prompt(self, agent_type: str, context: Dict[str, Any]) -> str:
        """Create Mesa agent-specific prompts"""
        
        base_context = f"""
MESA MULTI-AGENT SYSTEM CONTEXT:
You are a specialized {agent_type.upper()} AGENT in a Mesa-based multi-agent brewery simulation.

CURRENT MARKET SITUATION:
- Your Team's Inventory: {context.get('inventory', 100)} units
- Current Price: ${context.get('current_price', 10.0):.2f}
- Competitor Prices: {context.get('competitor_prices', [10.0])}
- Turn: {context.get('turn', 1)}
- Sales History: {context.get('sales_history', [])[-5:]}
- Production History: {context.get('production_history', [])[-5:]}
- Market Trend: {context.get('market_trend', 'stable')}

MESA AGENT CHARACTERISTICS:
- You are one agent in a coordinated multi-agent system
- Your decision will be combined with other specialized agents
- Focus on your specific domain expertise
- Consider emergent behaviors from agent interactions
- Make decisions that work well with other agents
"""

        agent_prompts = {
            "pricing": f"""
{base_context}

PRICING AGENT ROLE:
You are the PRICING SPECIALIST agent. Your job is to determine the optimal price.

PRICING CONSIDERATIONS:
- Inventory levels (high inventory = lower prices to clear stock)
- Competitor pricing (stay competitive but profitable)
- Market demand elasticity
- Profit margins (minimum viable price)
- Coordination with production and marketing agents

CONSTRAINTS:
- Price range: ${GameConfig.MIN_PRICE:.2f} - ${GameConfig.MAX_PRICE:.2f}
- Must maintain minimum profit margin of {GameConfig.PROFIT_MARGIN_TARGET*100}%
- Consider production costs: ${GameConfig.UNIT_PRODUCTION_COST}/unit

As a Mesa pricing agent, analyze the situation and recommend a price that:
1. Maximizes profit while staying competitive
2. Helps manage inventory levels appropriately  
3. Coordinates well with production and marketing agents

RESPOND IN JSON:
{{"price": 10.50, "reasoning": "Pricing agent rationale", "confidence": 0.85}}
""",

            "production": f"""
{base_context}

PRODUCTION AGENT ROLE:
You are the PRODUCTION SPECIALIST agent. Your job is to determine optimal production quantity.

PRODUCTION CONSIDERATIONS:
- Current inventory levels vs target inventory
- Expected demand based on pricing and marketing
- Production capacity constraints
- Inventory holding costs
- Coordination with pricing and marketing agents

CONSTRAINTS:
- Production range: 10-150 units
- Target inventory: {GameConfig.DESIRED_INVENTORY_WEEKS} weeks of sales
- Production cost: ${GameConfig.UNIT_PRODUCTION_COST}/unit
- Holding cost: ${GameConfig.UNIT_HOLDING_COST}/unit

As a Mesa production agent, analyze the situation and recommend production that:
1. Maintains optimal inventory levels
2. Responds to expected demand changes
3. Minimizes costs while avoiding stockouts

RESPOND IN JSON:
{{"production": 60, "reasoning": "Production agent rationale", "confidence": 0.85}}
""",

            "marketing": f"""
{base_context}

MARKETING AGENT ROLE:
You are the MARKETING SPECIALIST agent. Your job is to determine optimal marketing spend.

MARKETING CONSIDERATIONS:
- Inventory levels (high inventory = more marketing needed)
- Expected ROI from marketing spend
- Market competition and positioning
- Price sensitivity of demand
- Coordination with pricing and production agents

CONSTRAINTS:
- Marketing budget: $0 - $2,000
- Expected boost: ~$500 marketing = +15 demand units
- Diminishing returns at higher spend levels
- Consider profit margins from pricing agent

As a Mesa marketing agent, analyze the situation and recommend marketing spend that:
1. Maximizes demand generation ROI
2. Helps move inventory efficiently
3. Supports overall team profit objectives

RESPOND IN JSON:
{{"marketing": 800, "reasoning": "Marketing agent rationale", "confidence": 0.85}}
"""
        }
        
        return agent_prompts.get(agent_type, agent_prompts["pricing"])
    
    def _parse_mesa_response(self, response_text: str, agent_type: str) -> Dict[str, Any]:
        """Parse Mesa agent response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
            
            if agent_type == "pricing":
                price = max(GameConfig.MIN_PRICE, min(GameConfig.MAX_PRICE, 
                          float(decision_data.get('price', 10))))
                return {
                    'value': round(price, 2),
                    'reasoning': decision_data.get('reasoning', 'Mesa pricing decision'),
                    'confidence': decision_data.get('confidence', 0.8),
                    'agent_type': 'pricing',
                    'framework': 'mesa'
                }
            
            elif agent_type == "production":
                production = max(10, min(150, int(decision_data.get('production', 50))))
                return {
                    'value': production,
                    'reasoning': decision_data.get('reasoning', 'Mesa production decision'),
                    'confidence': decision_data.get('confidence', 0.8),
                    'agent_type': 'production',
                    'framework': 'mesa'
                }
            
            elif agent_type == "marketing":
                marketing = max(0, min(2000, int(decision_data.get('marketing', 500))))
                return {
                    'value': marketing,
                    'reasoning': decision_data.get('reasoning', 'Mesa marketing decision'),
                    'confidence': decision_data.get('confidence', 0.8),
                    'agent_type': 'marketing',
                    'framework': 'mesa'
                }
            
        except Exception as e:
            print(f"Failed to parse Mesa {agent_type} response: {e}")
            return self._fallback_decision(agent_type, {})
    
    def _fallback_decision(self, agent_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback decision when AI fails"""
        inventory = context.get('inventory', 100)
        inventory_ratio = inventory / 100.0
        
        if agent_type == "pricing":
            if inventory_ratio > 1.5:
                value = 9.5
            elif inventory_ratio < 0.5:
                value = 11.5
            else:
                value = 10.5
                
        elif agent_type == "production":
            if inventory_ratio > 1.5:
                value = 40
            elif inventory_ratio < 0.5:
                value = 100
            else:
                value = 60
                
        elif agent_type == "marketing":
            if inventory_ratio > 1.5:
                value = 1200
            elif inventory_ratio < 0.5:
                value = 400
            else:
                value = 800
        else:
            value = 50
        
        return {
            'value': value,
            'reasoning': f'Mesa {agent_type} agent: Fallback rule-based decision',
            'confidence': 0.6,
            'agent_type': agent_type,
            'framework': 'mesa',
            'ai_fallback': True
        }

class PricingAgent(Agent):
    """Mesa Pricing Agent with Gemini AI enhancement"""
    
    def __init__(self, unique_id, model, ai_helper):
        super().__init__(unique_id, model)
        self.agent_type = "pricing"
        self.ai_helper = ai_helper
        self.proposal = {}
        self.decision_history = []
    
    async def step_async(self, context: Dict[str, Any]):
        """Async step with AI enhancement"""
        decision = await self.ai_helper.get_agent_decision(self.agent_type, context)
        
        self.proposal = {
            "price": decision['value'],
            "reasoning": decision['reasoning'],
            "confidence": decision['confidence']
        }
        
        self.decision_history.append(decision)
        if len(self.decision_history) > 10:
            self.decision_history = self.decision_history[-10:]
    
    def step(self):
        """Synchronous step for compatibility"""
        # Basic fallback implementation
        self.proposal = {"price": 10.0, "reasoning": "Basic pricing", "confidence": 0.5}

class ProductionAgent(Agent):
    """Mesa Production Agent with Gemini AI enhancement"""
    
    def __init__(self, unique_id, model, ai_helper):
        super().__init__(unique_id, model)
        self.agent_type = "production"
        self.ai_helper = ai_helper
        self.proposal = {}
        self.decision_history = []
    
    async def step_async(self, context: Dict[str, Any]):
        """Async step with AI enhancement"""
        decision = await self.ai_helper.get_agent_decision(self.agent_type, context)
        
        self.proposal = {
            "target": decision['value'],
            "reasoning": decision['reasoning'],
            "confidence": decision['confidence']
        }
        
        self.decision_history.append(decision)
        if len(self.decision_history) > 10:
            self.decision_history = self.decision_history[-10:]
    
    def step(self):
        """Synchronous step for compatibility"""
        self.proposal = {"target": 50, "reasoning": "Basic production", "confidence": 0.5}

class MarketingAgent(Agent):
    """Mesa Marketing Agent with Gemini AI enhancement"""
    
    def __init__(self, unique_id, model, ai_helper):
        super().__init__(unique_id, model)
        self.agent_type = "marketing"
        self.ai_helper = ai_helper
        self.proposal = {}
        self.decision_history = []
    
    async def step_async(self, context: Dict[str, Any]):
        """Async step with AI enhancement"""
        decision = await self.ai_helper.get_agent_decision(self.agent_type, context)
        
        self.proposal = {
            "spend": decision['value'],
            "reasoning": decision['reasoning'],
            "confidence": decision['confidence']
        }
        
        self.decision_history.append(decision)
        if len(self.decision_history) > 10:
            self.decision_history = self.decision_history[-10:]
    
    def step(self):
        """Synchronous step for compatibility"""
        self.proposal = {"spend": 500, "reasoning": "Basic marketing", "confidence": 0.5}

class CEOAgent(Agent):
    """Mesa CEO Agent that coordinates all other agents"""
    
    def __init__(self, unique_id, model, ai_helper):
        super().__init__(unique_id, model)
        self.agent_type = "ceo"
        self.ai_helper = ai_helper
        self.final_decisions = {}
        self.coordination_history = []
    
    async def coordinate_async(self, proposals: Dict[str, Dict[str, Any]], context: Dict[str, Any]):
        """Coordinate agent proposals with AI-enhanced decision making"""
        
        # If Gemini AI is available, use it for coordination
        if self.ai_helper.enabled:
            coordination_decision = await self._ai_coordination(proposals, context)
        else:
            coordination_decision = self._rule_based_coordination(proposals, context)
        
        self.final_decisions = coordination_decision
        self.coordination_history.append(coordination_decision)
    
    async def _ai_coordination(self, proposals: Dict[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """AI-enhanced coordination of agent proposals"""
        
        coordination_prompt = f"""
MESA CEO AGENT - PROPOSAL COORDINATION

You are the CEO agent in a Mesa multi-agent system coordinating specialist agent proposals.

AGENT PROPOSALS:
- Pricing Agent: ${proposals.get('pricing', {}).get('price', 10):.2f} (Confidence: {proposals.get('pricing', {}).get('confidence', 0.5):.2f})
  Reasoning: {proposals.get('pricing', {}).get('reasoning', 'N/A')}

- Production Agent: {proposals.get('production', {}).get('target', 50)} units (Confidence: {proposals.get('production', {}).get('confidence', 0.5):.2f})
  Reasoning: {proposals.get('production', {}).get('reasoning', 'N/A')}

- Marketing Agent: ${proposals.get('marketing', {}).get('spend', 500)} (Confidence: {proposals.get('marketing', {}).get('confidence', 0.5):.2f})
  Reasoning: {proposals.get('marketing', {}).get('reasoning', 'N/A')}

MARKET CONTEXT:
- Current Inventory: {context.get('inventory', 100)} units
- Competitor Prices: {context.get('competitor_prices', [10])}
- Turn: {context.get('turn', 1)}

CEO COORDINATION ROLE:
As the Mesa CEO agent, your job is to:
1. Evaluate each specialist agent's proposal
2. Resolve any conflicts between proposals
3. Make strategic adjustments for global optimization
4. Ensure decisions work together as a cohesive strategy

Consider:
- Do the proposals work well together?
- Are there any strategic conflicts to resolve?
- Should any proposals be adjusted for better coordination?
- What's the overall strategic direction?

RESPOND IN JSON:
{{
    "price": 10.50,
    "production": 60,
    "marketing": 800,
    "coordination_strategy": "Brief description of coordination approach",
    "adjustments_made": ["list", "of", "adjustments"],
    "confidence": 0.85
}}
"""
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate_content(coordination_prompt)
            )
            
            return self._parse_coordination_response(response.text, proposals)
            
        except Exception as e:
            print(f"Mesa CEO AI coordination error: {e}")
            return self._rule_based_coordination(proposals, context)
    
    def _parse_coordination_response(self, response_text: str, proposals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Parse CEO coordination response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
            
            # Validate and constrain values
            price = max(GameConfig.MIN_PRICE, min(GameConfig.MAX_PRICE, 
                       float(decision_data.get('price', 10))))
            production = max(10, min(150, int(decision_data.get('production', 50))))
            marketing = max(0, min(2000, int(decision_data.get('marketing', 500))))
            
            return {
                'price': round(price, 2),
                'marketing_spend': marketing,
                'production_target': production,
                'reasoning': f"Mesa CEO: {decision_data.get('coordination_strategy', 'AI coordination')}",
                'adjustments': decision_data.get('adjustments_made', []),
                'confidence': decision_data.get('confidence', 0.8),
                'framework': 'mesa',
                'coordination_type': 'ai_enhanced'
            }
            
        except Exception as e:
            print(f"Failed to parse Mesa CEO response: {e}")
            return self._rule_based_coordination(proposals, {})
    
    def _rule_based_coordination(self, proposals: Dict[str, Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Rule-based coordination fallback"""
        
        # Extract proposals with defaults
        price_prop = proposals.get('pricing', {}).get('price', 10.0)
        prod_prop = proposals.get('production', {}).get('target', 50)
        mktg_prop = proposals.get('marketing', {}).get('spend', 500)
        
        # Simple coordination logic
        inventory = context.get('inventory', 100)
        inventory_ratio = inventory / 100.0
        
        # Strategic overrides based on inventory
        if inventory_ratio > 2.5:  # High inventory - clearance mode
            final_price = max(GameConfig.MIN_PRICE, price_prop - 0.5)
            final_production = max(10, int(prod_prop * 0.7))
            final_marketing = min(2000, int(mktg_prop * 1.3))
            strategy = "High inventory clearance"
            
        elif inventory_ratio < 0.8:  # Low inventory - conservative mode
            final_price = min(GameConfig.MAX_PRICE, price_prop + 0.3)
            final_production = int(prod_prop * 1.2)
            final_marketing = int(mktg_prop * 0.8)
            strategy = "Low inventory conservation"
            
        else:  # Normal operations
            final_price = price_prop
            final_production = prod_prop
            final_marketing = mktg_prop
            strategy = "Normal coordination"
        
        return {
            'price': round(final_price, 2),
            'marketing_spend': final_marketing,
            'production_target': final_production,
            'reasoning': f"Mesa CEO: {strategy} (rule-based)",
            'adjustments': [],
            'confidence': 0.7,
            'framework': 'mesa',
            'coordination_type': 'rule_based'
        }

class MesaBreweryModel(Model):
    """Mesa Model representing the brewery MAS with Gemini AI"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        super().__init__()
        
        self.schedule = RandomActivation(self)
        self.ai_helper = MesaGeminiHelper(gemini_api_key)
        self.turn = 0
        self.history = []
        
        # Create agents
        self.pricing_agent = PricingAgent(1, self, self.ai_helper)
        self.production_agent = ProductionAgent(2, self, self.ai_helper)
        self.marketing_agent = MarketingAgent(3, self, self.ai_helper)
        self.ceo_agent = CEOAgent(4, self, self.ai_helper)
        
        # Add agents to scheduler
        for agent in [self.pricing_agent, self.production_agent, self.marketing_agent]:
            self.schedule.add(agent)
        
        # Team state
        self.team_state = {
            'profit': 100000,
            'inventory': 100,
            'price': 10.0,
            'production': 50,
            'marketing': 500,
            'projected_demand': 50
        }
        
        print("✅ Mesa MAS Model initialized")
        print(f"🤖 Gemini AI: {'Enabled' if self.ai_helper.enabled else 'Disabled'}")
    
    async def step_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Async step function for the entire MAS"""
        
        self.turn += 1
        
        # Update context with current state
        mesa_context = context.copy()
        mesa_context.update({
            'inventory': self.team_state['inventory'],
            'current_price': self.team_state['price'],
            'sales_history': [h.get('sales', 50) for h in self.history[-10:]],
            'production_history': [h.get('production', 50) for h in self.history[-10:]],
            'turn': self.turn
        })
        
        # Run specialist agents asynchronously
        await asyncio.gather(
            self.pricing_agent.step_async(mesa_context),
            self.production_agent.step_async(mesa_context),
            self.marketing_agent.step_async(mesa_context)
        )
        
        # Collect proposals
        proposals = {
            'pricing': self.pricing_agent.proposal,
            'production': self.production_agent.proposal,
            'marketing': self.marketing_agent.proposal
        }
        
        # CEO coordination
        await self.ceo_agent.coordinate_async(proposals, mesa_context)
        
        # Get final decisions
        decisions = self.ceo_agent.final_decisions
        
        # Update team state
        self.team_state.update({
            'price': decisions['price'],
            'production': decisions['production_target'],
            'marketing': decisions['marketing_spend']
        })
        
        # Store history
        self.history.append({
            'turn': self.turn,
            'proposals': proposals,
            'final_decisions': decisions,
            'coordination_type': decisions.get('coordination_type', 'unknown')
        })
        
        return {
            'price': decisions['price'],
            'production': decisions['production_target'],
            'marketing': decisions['marketing_spend'],
            'reasoning': decisions['reasoning'],
            'framework': 'mesa',
            'agent_proposals': proposals,
            'coordination_confidence': decisions.get('confidence', 0.7),
            'ai_enhanced': self.ai_helper.enabled
        }
    
    def step(self):
        """Synchronous step for compatibility"""
        # Basic implementation for fallback
        self.schedule.step()
        return {
            'price': 10.0,
            'production': 50,
            'marketing': 500,
            'reasoning': 'Mesa MAS: Basic operation',
            'framework': 'mesa'
        }
    
    def get_agent_details(self) -> Dict[str, Any]:
        """Get detailed information about all agents"""
        return {
            'pricing_agent': {
                'type': self.pricing_agent.agent_type,
                'decision_history_count': len(self.pricing_agent.decision_history),
                'last_proposal': self.pricing_agent.proposal
            },
            'production_agent': {
                'type': self.production_agent.agent_type,
                'decision_history_count': len(self.production_agent.decision_history),
                'last_proposal': self.production_agent.proposal
            },
            'marketing_agent': {
                'type': self.marketing_agent.agent_type,
                'decision_history_count': len(self.marketing_agent.decision_history),
                'last_proposal': self.marketing_agent.proposal
            },
            'ceo_agent': {
                'type': self.ceo_agent.agent_type,
                'coordination_history_count': len(self.ceo_agent.coordination_history),
                'last_decisions': self.ceo_agent.final_decisions
            },
            'model_stats': {
                'total_turns': self.turn,
                'ai_enabled': self.ai_helper.enabled,
                'agent_count': len(self.schedule.agents)
            }
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_mesa_framework():
        print("🧪 Testing Mesa Framework with Gemini AI...")
        
        # Initialize Mesa model (set your API key here)
        GEMINI_API_KEY = None  # Replace with your actual API key
        mesa_model = MesaBreweryModel(GEMINI_API_KEY)
        
        # Test context
        test_context = {
            'inventory': 120,
            'current_price': 10.5,
            'competitor_prices': [9.8, 11.2, 10.0],
            'turn': 5,
            'market_trend': 'growing'
        }
        
        # Run async step
        decisions = await mesa_model.step_async(test_context)
        
        print("\n📊 Mesa Framework Results:")
        print(f"💰 Price: ${decisions['price']:.2f}")
        print(f"🏭 Production: {decisions['production']} units") 
        print(f"📢 Marketing: ${decisions['marketing']}")
        print(f"🤖 AI Enhanced: {decisions['ai_enhanced']}")
        print(f"📝 Reasoning: {decisions['reasoning']}")
        
        # Show agent details
        agent_details = mesa_model.get_agent_details()
        print(f"\n🤖 Agent Details:")
        print(f"- Pricing Agent: {agent_details['pricing_agent']['decision_history_count']} decisions")
        print(f"- Production Agent: {agent_details['production_agent']['decision_history_count']} decisions")
        print(f"- Marketing Agent: {agent_details['marketing_agent']['decision_history_count']} decisions")
        print(f"- CEO Coordination: {agent_details['ceo_agent']['coordination_history_count']} coordinations")
    
    # Run test
    asyncio.run(test_mesa_framework())