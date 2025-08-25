# # temporal_framework.py - Temporal Workflows with Gemini AI Integration

# import asyncio
# import json
# import random
# import numpy as np
# from typing import Dict, Any, List, Optional
# from dataclasses import dataclass
# from datetime import timedelta
# import google.generativeai as genai

# # Mock Temporal imports for demo (replace with real temporalio imports)
# try:
#     from temporalio import activity, workflow
#     from temporalio.client import Client
#     from temporalio.worker import Worker
#     from temporalio.common import RetryPolicy
#     TEMPORAL_AVAILABLE = True
#     print("✅ Temporal SDK available")
# except ImportError:
#     TEMPORAL_AVAILABLE = False
#     print("⚠️ Temporal SDK not available - using mock implementation")
    
#     # Mock decorators for demo
#     def workflow(cls):
#         cls._is_workflow = True
#         return cls
    
#     def activity(func):
#         func._is_activity = True
#         return func
    
#     class RetryPolicy:
#         def __init__(self, maximum_attempts=3):
#             self.maximum_attempts = maximum_attempts

# @dataclass
# class TemporalGameConfig:
#     """Temporal-specific game configuration"""
#     DESIRED_INVENTORY_WEEKS: float = 2.5
#     PROFIT_MARGIN_TARGET: float = 0.35
#     UNIT_PRODUCTION_COST: float = 3.0
#     UNIT_HOLDING_COST: float = 0.5
#     BASE_MARKET_PRICE: float = 10.0
#     MIN_PRICE: float = 8.0
#     MAX_PRICE: float = 15.0
    
#     # Temporal-specific configs
#     WORKFLOW_TIMEOUT: int = 30  # seconds
#     ACTIVITY_TIMEOUT: int = 10  # seconds
#     MAX_RETRIES: int = 3

# @dataclass
# class MarketContext:
#     """Context data for Temporal workflows"""
#     inventory: int
#     current_price: float
#     competitor_prices: List[float]
#     turn: int
#     sales_history: List[int]
#     production_history: List[int]
#     market_trend: str = "stable"

# @dataclass
# class WorkflowDecision:
#     """Decision output from Temporal workflow"""
#     price: float
#     production: int
#     marketing: int
#     reasoning: str
#     workflow_id: str
#     activities_executed: List[str]
#     confidence: float = 0.8

# class TemporalGeminiHelper:
#     """Gemini AI helper for Temporal workflows"""
    
#     def __init__(self, api_key: Optional[str] = None):
#         self.enabled = False
#         self.model = None
        
#         if api_key:
#             try:
#                 genai.configure(api_key=api_key)
#                 self.model = genai.GenerativeModel('gemini-pro')
#                 self.enabled = True
#                 print("✅ Temporal Gemini AI: Enabled")
#             except Exception as e:
#                 print(f"❌ Temporal Gemini AI initialization failed: {e}")
#         else:
#             print("⚠️ Temporal Gemini AI: Disabled (no API key)")
    
#     async def get_activity_decision(self, activity_type: str, context: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, Any]:
#         """Get AI-enhanced decision for a specific Temporal activity"""
#         if not self.enabled:
#             return self._fallback_activity_decision(activity_type, context)
        
#         try:
#             prompt = self._create_temporal_activity_prompt(activity_type, context, workflow_context)
            
#             response = await asyncio.get_event_loop().run_in_executor(
#                 None, lambda: self.model.generate_content(prompt)
#             )
            
#             return self._parse_temporal_response(response.text, activity_type)
            
#         except Exception as e:
#             print(f"Temporal Gemini error for {activity_type} activity: {e}")
#             return self._fallback_activity_decision(activity_type, context)
    
#     def _create_temporal_activity_prompt(self, activity_type: str, context: Dict[str, Any], workflow_context: Dict[str, Any]) -> str:
#         """Create Temporal activity-specific prompts"""
        
#         base_context = f"""
# TEMPORAL WORKFLOW SYSTEM CONTEXT:
# You are executing a specialized ACTIVITY within a Temporal workflow for brewery decision-making.

# WORKFLOW STATE:
# - Activity Type: {activity_type.upper()}
# - Workflow ID: {workflow_context.get('workflow_id', 'brew_decision_workflow')}
# - Execution Attempt: {workflow_context.get('attempt', 1)}
# - Previous Activities: {workflow_context.get('completed_activities', [])}

# CURRENT MARKET DATA:
# - Inventory Level: {context.get('inventory', 100)} units
# - Current Price: ${context.get('current_price', 10.0):.2f}
# - Competitor Prices: {context.get('competitor_prices', [10.0])}
# - Turn: {context.get('turn', 1)}
# - Sales History: {context.get('sales_history', [])[-5:]}
# - Market Trend: {context.get('market_trend', 'stable')}

# TEMPORAL WORKFLOW CHARACTERISTICS:
# - This is a durable, fault-tolerant workflow system
# - Each activity is independently executable and retryable
# - Activities coordinate to achieve overall business objectives
# - State is maintained across activity boundaries
# - Decisions must be deterministic for workflow replay
# """

#         activity_prompts = {
#             "market_analysis": f"""
# {base_context}

# MARKET ANALYSIS ACTIVITY:
# You are the MARKET ANALYSIS activity in the Temporal workflow.

# ACTIVITY RESPONSIBILITIES:
# - Analyze current market conditions and trends
# - Evaluate competitive landscape
# - Assess demand patterns and elasticity
# - Provide market insights for downstream activities
# - Generate actionable market intelligence

# ANALYSIS FOCUS:
# - Competitor pricing analysis: {context.get('competitor_prices', [])}
# - Historical demand patterns: {context.get('sales_history', [])[-10:]}
# - Market trend direction: {context.get('market_trend', 'stable')}
# - Inventory pressure analysis: {context.get('inventory', 100)} units

# As a Temporal market analysis activity, provide comprehensive market insights:
# 1. Market conditions assessment
# 2. Competitive position analysis
# 3. Demand forecast and trends
# 4. Strategic recommendations for pricing/production/marketing

# RESPOND IN JSON:
# {{
#     "market_conditions": "favorable/challenging/neutral",
#     "competitive_position": "strong/weak/average", 
#     "demand_forecast": 55,
#     "price_recommendation_range": [9.5, 11.5],
#     "market_insights": "Key insights about market state",
#     "confidence": 0.85,
#     "next_activity_guidance": "Guidance for pricing activity"
# }}
# """,

#             "pricing_strategy": f"""
# {base_context}

# PRICING STRATEGY ACTIVITY:
# You are the PRICING STRATEGY activity in the Temporal workflow.

# PREVIOUS ACTIVITY RESULTS:
# - Market Analysis: {workflow_context.get('market_analysis', 'Not available')}

# ACTIVITY RESPONSIBILITIES:
# - Determine optimal pricing strategy based on market analysis
# - Consider profit margins and competitive positioning
# - Factor in inventory levels and demand forecasts
# - Provide pricing decision for workflow execution

# PRICING CONSTRAINTS:
# - Price range: ${TemporalGameConfig.MIN_PRICE:.2f} - ${TemporalGameConfig.MAX_PRICE:.2f}
# - Target profit margin: {TemporalGameConfig.PROFIT_MARGIN_TARGET*100}%
# - Production cost: ${TemporalGameConfig.UNIT_PRODUCTION_COST}/unit

# As a Temporal pricing strategy activity, determine optimal pricing:
# 1. Analyze market analysis results
# 2. Calculate competitive pricing position
# 3. Factor in inventory and cost constraints
# 4. Recommend final price with justification

# RESPOND IN JSON:
# {{
#     "recommended_price": 10.50,
#     "pricing_strategy": "competitive/premium/discount",
#     "margin_analysis": "Expected margin percentage and rationale",
#     "risk_assessment": "Low/Medium/High risk level",
#     "confidence": 0.85,
#     "justification": "Detailed reasoning for price recommendation"
# }}
# """,

#             "production_planning": f"""
# {base_context}

# PRODUCTION PLANNING ACTIVITY:
# You are the PRODUCTION PLANNING activity in the Temporal workflow.

# PREVIOUS ACTIVITY RESULTS:
# - Market Analysis: {workflow_context.get('market_analysis', 'Not available')}
# - Pricing Strategy: {workflow_context.get('pricing_strategy', 'Not available')}

# ACTIVITY RESPONSIBILITIES:
# - Determine optimal production quantity
# - Balance inventory levels with expected demand
# - Consider production costs and capacity constraints
# - Coordinate with pricing and marketing strategies

# PRODUCTION CONSTRAINTS:
# - Production range: 10-150 units per turn
# - Target inventory: {TemporalGameConfig.DESIRED_INVENTORY_WEEKS} weeks of sales
# - Production cost: ${TemporalGameConfig.UNIT_PRODUCTION_COST}/unit
# - Current inventory: {context.get('inventory', 100)} units

# As a Temporal production planning activity, determine optimal production:
# 1. Analyze demand forecasts from market analysis
# 2. Consider pricing impact on demand
# 3. Calculate optimal inventory levels
# 4. Recommend production quantity with rationale

# RESPOND IN JSON:
# {{
#     "recommended_production": 65,
#     "production_strategy": "aggressive/conservative/balanced",
#     "inventory_analysis": "Current vs target inventory assessment",
#     "demand_response": "Expected demand based on pricing",
#     "confidence": 0.85,
#     "rationale": "Detailed reasoning for production recommendation"
# }}
# """,

#             "marketing_optimization": f"""
# {base_context}

# MARKETING OPTIMIZATION ACTIVITY:
# You are the MARKETING OPTIMIZATION activity in the Temporal workflow.

# PREVIOUS ACTIVITY RESULTS:
# - Market Analysis: {workflow_context.get('market_analysis', 'Not available')}
# - Pricing Strategy: {workflow_context.get('pricing_strategy', 'Not available')}
# - Production Planning: {workflow_context.get('production_planning', 'Not available')}

# ACTIVITY RESPONSIBILITIES:
# - Determine optimal marketing spend
# - Maximize ROI based on pricing and production decisions
# - Consider inventory levels and market conditions
# - Provide final marketing recommendation

# MARKETING CONSTRAINTS:
# - Marketing budget: $0 - $2,000
# - Expected impact: ~$500 = +15 demand units
# - Diminishing returns at higher spend levels
# - Must align with pricing and production strategy

# As a Temporal marketing optimization activity, determine optimal marketing:
# 1. Analyze previous activity recommendations
# 2. Calculate expected marketing ROI
# 3. Consider inventory movement needs
# 4. Recommend marketing spend with justification

# RESPOND IN JSON:
# {{
#     "recommended_marketing": 800,
#     "marketing_strategy": "aggressive/targeted/minimal",
#     "roi_analysis": "Expected return on marketing investment",
#     "inventory_support": "How marketing supports inventory movement",
#     "confidence": 0.85,
#     "justification": "Detailed reasoning for marketing recommendation"
# }}
# """,

#             "decision_coordination": f"""
# {base_context}

# DECISION COORDINATION ACTIVITY:
# You are the DECISION COORDINATION activity in the Temporal workflow.

# ALL PREVIOUS ACTIVITY RESULTS:
# - Market Analysis: {workflow_context.get('market_analysis', 'Not available')}
# - Pricing Strategy: {workflow_context.get('pricing_strategy', 'Not available')}
# - Production Planning: {workflow_context.get('production_planning', 'Not available')}
# - Marketing Optimization: {workflow_context.get('marketing_optimization', 'Not available')}

# ACTIVITY RESPONSIBILITIES:
# - Coordinate all activity recommendations
# - Resolve any conflicts between activities
# - Ensure decisions work together cohesively
# - Provide final integrated business decision

# COORDINATION CONSIDERATIONS:
# - Do pricing, production, and marketing align strategically?
# - Are there any resource or capacity conflicts?
# - Does the overall strategy make business sense?
# - What adjustments are needed for optimal coordination?

# As a Temporal decision coordination activity, provide final integrated decision:
# 1. Review all activity recommendations
# 2. Identify and resolve any strategic conflicts
# 3. Make final adjustments for coordination
# 4. Provide comprehensive final decision

# RESPOND IN JSON:
# {{
#     "final_price": 10.50,
#     "final_production": 65,
#     "final_marketing": 800,
#     "coordination_strategy": "How activities were coordinated",
#     "adjustments_made": ["list", "of", "coordination", "adjustments"],
#     "overall_confidence": 0.85,
#     "workflow_summary": "Summary of the complete workflow execution"
# }}
# """
#         }
        
#         return activity_prompts.get(activity_type, activity_prompts["market_analysis"])
    
#     def _parse_temporal_response(self, response_text: str, activity_type: str) -> Dict[str, Any]:
#         """Parse Temporal activity response"""
#         try:
#             import re
#             json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
#             if json_match:
#                 decision_data = json.loads(json_match.group())
#             else:
#                 raise ValueError("No JSON found")
            
#             # Add common fields
#             decision_data['activity_type'] = activity_type
#             decision_data['framework'] = 'temporal'
#             decision_data['timestamp'] = asyncio.get_event_loop().time()
            
#             return decision_data
            
#         except Exception as e:
#             print(f"Failed to parse Temporal {activity_type} response: {e}")
#             return self._fallback_activity_decision(activity_type, {})
    
#     def _fallback_activity_decision(self, activity_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
#         """Fallback decision when AI fails"""
#         inventory = context.get('inventory', 100)
        
#         fallback_decisions = {
#             "market_analysis": {
#                 "market_conditions": "neutral",
#                 "competitive_position": "average",
#                 "demand_forecast": 50,
#                 "price_recommendation_range": [9.0, 11.0],
#                 "market_insights": "Fallback market analysis",
#                 "confidence": 0.6
#             },
#             "pricing_strategy": {
#                 "recommended_price": 10.5 if inventory < 100 else 9.8,
#                 "pricing_strategy": "competitive",
#                 "confidence": 0.6
#             },
#             "production_planning": {
#                 "recommended_production": 80 if inventory < 50 else 40,
#                 "production_strategy": "balanced",
#                 "confidence": 0.6
#             },
#             "marketing_optimization": {
#                 "recommended_marketing": 1200 if inventory > 120 else 600,
#                 "marketing_strategy": "targeted",
#                 "confidence": 0.6
#             },
#             "decision_coordination": {
#                 "final_price": 10.0,
#                 "final_production": 50,
#                 "final_marketing": 700,
#                 "coordination_strategy": "fallback coordination",
#                 "confidence": 0.6
#             }
#         }
        
#         result = fallback_decisions.get(activity_type, fallback_decisions["market_analysis"])
#         result.update({
#             'activity_type': activity_type,
#             'framework': 'temporal',
#             'fallback': True
#         })
        
#         return result

# # Temporal Activities (decorated for real Temporal, mock for demo)
# @activity
# async def market_analysis_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
#     """Market analysis activity with Gemini AI"""
#     context_dict = {
#         'inventory': context.inventory,
#         'current_price': context.current_price,
#         'competitor_prices': context.competitor_prices,
#         'turn': context.turn,
#         'sales_history': context.sales_history,
#         'market_trend': context.market_trend
#     }
    
#     return await ai_helper.get_activity_decision("market_analysis", context_dict, workflow_context)

# @activity
# async def pricing_strategy_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
#     """Pricing strategy activity with Gemini AI"""
#     context_dict = {
#         'inventory': context.inventory,
#         'current_price': context.current_price,
#         'competitor_prices': context.competitor_prices,
#         'turn': context.turn
#     }
    
#     return await ai_helper.get_activity_decision("pricing_strategy", context_dict, workflow_context)

# @activity
# async def production_planning_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
#     """Production planning activity with Gemini AI"""
#     context_dict = {
#         'inventory': context.inventory,
#         'current_price': context.current_price,
#         'sales_history': context.sales_history,
#         'turn': context.turn
#     }
    
#     return await ai_helper.get_activity_decision("production_planning", context_dict, workflow_context)

# @activity
# async def marketing_optimization_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
#     """Marketing optimization activity with Gemini AI"""
#     context_dict = {
#         'inventory': context.inventory,
#         'current_price': context.current_price,
#         'competitor_prices': context.competitor_prices,
#         'turn': context.turn
#     }
    
#     return await ai_helper.get_activity_decision("marketing_optimization", context_dict, workflow_context)

# @activity
# async def decision_coordination_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
#     """Decision coordination activity with Gemini AI"""
#     context_dict = {
#         'inventory': context.inventory,
#         'current_price': context.current_price,
#         'turn': context.turn
#     }
    
#     return await ai_helper.get_activity_decision("decision_coordination", context_dict, workflow_context)

# @workflow
# class BreweryBusinessWorkflow:
#     """Main Temporal workflow for brewery business decisions"""
    
#     def __init__(self):
#         self.workflow_id = f"brewery_workflow_{random.randint(1000, 9999)}"
#         self.completed_activities = []
#         self.activity_results = {}
    
#     async def run(self, context: MarketContext, ai_helper: TemporalGeminiHelper) -> WorkflowDecision:
#         """Execute the complete business decision workflow"""
        
#         workflow_context = {
#             'workflow_id': self.workflow_id,
#             'attempt': 1,
#             'completed_activities': []
#         }
        
#         try:
#             # Activity 1: Market Analysis
#             market_result = await self._execute_activity(
#                 market_analysis_activity, context, workflow_context, ai_helper, "market_analysis"
#             )
            
#             # Activity 2: Pricing Strategy
#             workflow_context['market_analysis'] = market_result
#             pricing_result = await self._execute_activity(
#                 pricing_strategy_activity, context, workflow_context, ai_helper, "pricing_strategy"
#             )
            
#             # Activity 3: Production Planning
#             workflow_context['pricing_strategy'] = pricing_result
#             production_result = await self._execute_activity(
#                 production_planning_activity, context, workflow_context, ai_helper, "production_planning"
#             )
            
#             # Activity 4: Marketing Optimization
#             workflow_context['production_planning'] = production_result
#             marketing_result = await self._execute_activity(
#                 marketing_optimization_activity, context, workflow_context, ai_helper, "marketing_optimization"
#             )
            
#             # Activity 5: Decision Coordination
#             workflow_context.update({
#                 'marketing_optimization': marketing_result
#             })
#             coordination_result = await self._execute_activity(
#                 decision_coordination_activity, context, workflow_context, ai_helper, "decision_coordination"
#             )
            
#             # Create final workflow decision
#             return WorkflowDecision(
#                 price=float(coordination_result.get('final_price', pricing_result.get('recommended_price', 10.0))),
#                 production=int(coordination_result.get('final_production', production_result.get('recommended_production', 50))),
#                 marketing=int(coordination_result.get('final_marketing', marketing_result.get('recommended_marketing', 500))),
#                 reasoning=f"Temporal Workflow: {coordination_result.get('coordination_strategy', 'Workflow execution completed')}",
#                 workflow_id=self.workflow_id,
#                 activities_executed=self.completed_activities,
#                 confidence=float(coordination_result.get('overall_confidence', 0.8))
#             )
            
#         except Exception as e:
#             print(f"Workflow execution error: {e}")
#             return self._fallback_workflow_decision(context)
    
#     async def _execute_activity(self, activity_func, context: MarketContext, workflow_context: Dict[str, Any], 
#                               ai_helper: TemporalGeminiHelper, activity_name: str) -> Dict[str, Any]:
#         """Execute a single activity with retry logic"""
        
#         max_retries = TemporalGameConfig.MAX_RETRIES
#         for attempt in range(max_retries):
#             try:
#                 workflow_context['attempt'] = attempt + 1
                
#                 # Execute activity (with or without real Temporal)
#                 if TEMPORAL_AVAILABLE:
#                     # In real Temporal, this would use workflow.execute_activity
#                     result = await activity_func(context, workflow_context, ai_helper)
#                 else:
#                     # Mock execution
#                     result = await activity_func(context, workflow_context, ai_helper)
                
#                 # Track completion
#                 self.completed_activities.append(activity_name)
#                 self.activity_results[activity_name] = result
#                 workflow_context['completed_activities'] = self.completed_activities
                
#                 return result
                
#             except Exception as e:
#                 print(f"Activity {activity_name} attempt {attempt + 1} failed: {e}")
#                 if attempt == max_retries - 1:
#                     # Final fallback
#                     fallback_result = {
#                         'fallback': True,
#                         'activity_type': activity_name,
#                         'error': str(e)
#                     }
#                     self.completed_activities.append(f"{activity_name}_fallback")
#                     return fallback_result
                
#                 await asyncio.sleep(1)  # Brief retry delay
    
#     def _fallback_workflow_decision(self, context: MarketContext) -> WorkflowDecision:
#         """Fallback workflow decision when everything fails"""
#         inventory_ratio = context.inventory / 100.0
        
#         if inventory_ratio > 1.5:
#             price, production, marketing = 9.5, 40, 1200
#         elif inventory_ratio < 0.5:
#             price, production, marketing = 11.5, 100, 400
#         else:
#             price, production, marketing = 10.5, 60, 800
        
#         return WorkflowDecision(
#             price=price,
#             production=production,
#             marketing=marketing,
#             reasoning="Temporal Workflow: Complete fallback decision (all activities failed)",
#             workflow_id=f"fallback_{self.workflow_id}",
#             activities_executed=["fallback"],
#             confidence=0.5
#         )

# class TemporalBrewerySystem:
#     """Main Temporal brewery system integrating workflows with Gemini AI"""
    
#     def __init__(self, gemini_api_key: Optional[str] = None):
#         self.ai_helper = TemporalGeminiHelper(gemini_api_key)
#         self.workflow_history = []
#         self.team_state = {
#             'profit': 100000,
#             'inventory': 100,
#             'price': 10.0,
#             'production': 50,
#             'marketing': 500,
#             'projected_demand': 50
#         }
        
#         print("✅ Temporal Brewery System initialized")
#         print(f"🤖 Gemini AI: {'Enabled' if self.ai_helper.enabled else 'Disabled'}")
#         print(f"⚡ Temporal SDK: {'Available' if TEMPORAL_AVAILABLE else 'Mock implementation'}")
    
#     async def execute_business_workflow(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Execute the complete business decision workflow"""
        
#         # Create workflow context
#         context = MarketContext(
#             inventory=context_data.get('inventory', self.team_state['inventory']),
#             current_price=context_data.get('current_price', self.team_state['price']),
#             competitor_prices=context_data.get('competitor_prices', [10.0]),
#             turn=context_data.get('turn', 1),
#             sales_history=context_data.get('sales_history', []),
#             production_history=context_data.get('production_history', []),
#             market_trend=context_data.get('market_trend', 'stable')
#         )
        
#         # Create and execute workflow
#         workflow = BreweryBusinessWorkflow()
#         decision = await workflow.run(context, self.ai_helper)
        
#         # Update team state
#         self.team_state.update({
#             'price': decision.price,
#             'production': decision.production,
#             'marketing': decision.marketing
#         })
        
#         # Store workflow history
#         workflow_record = {
#             'workflow_id': decision.workflow_id,
#             'decision': decision,
#             'activities_executed': decision.activities_executed,
#             'context': context_data,
#             'timestamp': asyncio.get_event_loop().time()
#         }
#         self.workflow_history.append(workflow_record)
        
#         return {
#             'price': decision.price,
#             'production': decision.production,
#             'marketing': decision.marketing,
#             'reasoning': decision.reasoning,
#             'framework': 'temporal',
#             'workflow_id': decision.workflow_id,
#             'activities_executed': decision.activities_executed,
#             'confidence': decision.confidence,
#             'ai_enhanced': self.ai_helper.enabled
#         }
    
#     def get_workflow_details(self) -> Dict[str, Any]:
#         """Get detailed information about workflow executions"""
#         if not self.workflow_history:
#             return {'status': 'no_workflows_executed'}
        
#         latest_workflow = self.workflow_history[-1]
        
#         return {
#             'total_workflows': len(self.workflow_history),
#             'latest_workflow': {
#                 'id': latest_workflow['workflow_id'],
#                 'activities_count': len(latest_workflow['activities_executed']),
#                 'activities': latest_workflow['activities_executed'],
#                 'confidence': latest_workflow['decision'].confidence
#             },
#             'ai_enabled': self.ai_helper.enabled,
#             'temporal_available': TEMPORAL_AVAILABLE,
#             'average_confidence': np.mean([w['decision'].confidence for w in self.workflow_history])
#         }

# # Example usage and testing
# if __name__ == "__main__":
#     async def test_temporal_framework():
#         print("🧪 Testing Temporal Framework with Gemini AI...")
        
#         # Initialize Temporal system (set your API key here)
#         GEMINI_API_KEY = None  # Replace with your actual API key
#         temporal_system = TemporalBrewerySystem(GEMINI_API_KEY)
        
#         # Test context
#         test_context = {
#             'inventory': 85,
#             'current_price': 10.2,
#             'competitor_prices': [9.5, 11.0, 10.8],
#             'turn': 3,
#             'sales_history': [45, 52, 48],
#             'production_history': [60, 55, 50],
#             'market_trend': 'stable'
#         }
        
#         # Execute workflow
#         decisions = await temporal_system.execute_business_workflow(test_context)
        
#         print("\n📊 Temporal Framework Results:")
#         print(f"💰 Price: ${decisions['price']:.2f}")
#         print(f"🏭 Production: {decisions['production']} units")
#         print(f"📢 Marketing: ${decisions['marketing']}")
#         print(f"🤖 AI Enhanced: {decisions['ai_enhanced']}")
#         print(f"⚡ Workflow ID: {decisions['workflow_id']}")
#         print(f"📝 Reasoning: {decisions['reasoning']}")
#         print(f"🎯 Confidence: {decisions['confidence']:.2f}")
        
#         # Show workflow details
#         workflow_details = temporal_system.get_workflow_details()
#         print(f"\n⚡ Workflow Details:")
#         print(f"- Total Workflows: {workflow_details['total_workflows']}")
#         print(f"- Activities Executed: {workflow_details['latest_workflow']['activities']}")
#         print(f"- Average Confidence: {workflow_details.get('average_confidence', 0):.2f}")
#         print(f"- Temporal SDK: {'Available' if workflow_details['temporal_available'] else 'Mock'}")
    
#     # Run test
#     asyncio.run(test_temporal_framework())

# temporal_framework_fixed.py - Fixed Temporal Workflows with Gemini AI Integration

import asyncio
import json
import random
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import timedelta
import google.generativeai as genai

# Handle Temporal imports with proper fallbacks
TEMPORAL_AVAILABLE = False
workflow_decorator = None
activity_decorator = None

try:
    from temporalio import workflow as temporal_workflow
    from temporalio import activity as temporal_activity
    from temporalio.client import Client
    from temporalio.worker import Worker
    from temporalio.common import RetryPolicy
    
    workflow_decorator = temporal_workflow.defn
    activity_decorator = temporal_activity.defn
    TEMPORAL_AVAILABLE = True
    print("✅ Temporal SDK available")
    
except ImportError:
    print("⚠️ Temporal SDK not available - using mock implementation")
    
    # Create mock decorators that don't interfere
    def mock_workflow_decorator(cls):
        cls._is_workflow = True
        return cls
    
    def mock_activity_decorator(func):
        func._is_activity = True
        return func
    
    workflow_decorator = mock_workflow_decorator
    activity_decorator = mock_activity_decorator
    
    # Mock classes for compatibility
    class RetryPolicy:
        def __init__(self, maximum_attempts=3):
            self.maximum_attempts = maximum_attempts

@dataclass
class TemporalGameConfig:
    """Temporal-specific game configuration"""
    DESIRED_INVENTORY_WEEKS: float = 2.5
    PROFIT_MARGIN_TARGET: float = 0.35
    UNIT_PRODUCTION_COST: float = 3.0
    UNIT_HOLDING_COST: float = 0.5
    BASE_MARKET_PRICE: float = 10.0
    MIN_PRICE: float = 8.0
    MAX_PRICE: float = 15.0
    
    # Temporal-specific configs
    WORKFLOW_TIMEOUT: int = 30  # seconds
    ACTIVITY_TIMEOUT: int = 10  # seconds
    MAX_RETRIES: int = 3

@dataclass
class MarketContext:
    """Context data for Temporal workflows"""
    inventory: int
    current_price: float
    competitor_prices: List[float]
    turn: int
    sales_history: List[int]
    production_history: List[int]
    market_trend: str = "stable"

@dataclass
class WorkflowDecision:
    """Decision output from Temporal workflow"""
    price: float
    production: int
    marketing: int
    reasoning: str
    workflow_id: str
    activities_executed: List[str]
    confidence: float = 0.8

class TemporalGeminiHelper:
    """Gemini AI helper for Temporal workflows"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.enabled = False
        self.model = None
        
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel('gemini-pro')
                self.enabled = True
                print("✅ Temporal Gemini AI: Enabled")
            except Exception as e:
                print(f"❌ Temporal Gemini AI initialization failed: {e}")
        else:
            print("⚠️ Temporal Gemini AI: Disabled (no API key)")
    
    async def get_activity_decision(self, activity_type: str, context: Dict[str, Any], workflow_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-enhanced decision for a specific Temporal activity"""
        if not self.enabled:
            return self._fallback_activity_decision(activity_type, context)
        
        try:
            prompt = self._create_temporal_activity_prompt(activity_type, context, workflow_context)
            
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate_content(prompt)
            )
            
            return self._parse_temporal_response(response.text, activity_type)
            
        except Exception as e:
            print(f"Temporal Gemini error for {activity_type} activity: {e}")
            return self._fallback_activity_decision(activity_type, context)
    
    def _create_temporal_activity_prompt(self, activity_type: str, context: Dict[str, Any], workflow_context: Dict[str, Any]) -> str:
        """Create Temporal activity-specific prompts"""
        
        base_context = f"""
TEMPORAL WORKFLOW SYSTEM CONTEXT:
You are executing a specialized ACTIVITY within a Temporal workflow for brewery decision-making.

WORKFLOW STATE:
- Activity Type: {activity_type.upper()}
- Workflow ID: {workflow_context.get('workflow_id', 'brew_decision_workflow')}
- Execution Attempt: {workflow_context.get('attempt', 1)}
- Previous Activities: {workflow_context.get('completed_activities', [])}

CURRENT MARKET DATA:
- Inventory Level: {context.get('inventory', 100)} units
- Current Price: ${context.get('current_price', 10.0):.2f}
- Competitor Prices: {context.get('competitor_prices', [10.0])}
- Turn: {context.get('turn', 1)}
- Sales History: {context.get('sales_history', [])[-5:]}
- Market Trend: {context.get('market_trend', 'stable')}

TEMPORAL WORKFLOW CHARACTERISTICS:
- This is a durable, fault-tolerant workflow system
- Each activity is independently executable and retryable
- Activities coordinate to achieve overall business objectives
- State is maintained across activity boundaries
- Decisions must be deterministic for workflow replay
"""

        activity_prompts = {
            "market_analysis": f"""
{base_context}

MARKET ANALYSIS ACTIVITY:
You are the MARKET ANALYSIS activity in the Temporal workflow.

Provide comprehensive market insights and respond in JSON:
{{
    "market_conditions": "favorable/challenging/neutral",
    "competitive_position": "strong/weak/average", 
    "demand_forecast": 55,
    "price_recommendation_range": [9.5, 11.5],
    "market_insights": "Key insights about market state",
    "confidence": 0.85,
    "next_activity_guidance": "Guidance for pricing activity"
}}
""",

            "pricing_strategy": f"""
{base_context}

PRICING STRATEGY ACTIVITY:
As the Temporal pricing strategy activity, determine optimal pricing and respond in JSON:
{{
    "recommended_price": 10.50,
    "pricing_strategy": "competitive/premium/discount",
    "margin_analysis": "Expected margin percentage and rationale",
    "risk_assessment": "Low/Medium/High risk level",
    "confidence": 0.85,
    "justification": "Detailed reasoning for price recommendation"
}}
""",

            "production_planning": f"""
{base_context}

PRODUCTION PLANNING ACTIVITY:
As the Temporal production planning activity, determine optimal production and respond in JSON:
{{
    "recommended_production": 65,
    "production_strategy": "aggressive/conservative/balanced",
    "inventory_analysis": "Current vs target inventory assessment",
    "demand_response": "Expected demand based on pricing",
    "confidence": 0.85,
    "rationale": "Detailed reasoning for production recommendation"
}}
""",

            "marketing_optimization": f"""
{base_context}

MARKETING OPTIMIZATION ACTIVITY:
As the Temporal marketing optimization activity, determine optimal marketing and respond in JSON:
{{
    "recommended_marketing": 800,
    "marketing_strategy": "aggressive/targeted/minimal",
    "roi_analysis": "Expected return on marketing investment",
    "inventory_support": "How marketing supports inventory movement",
    "confidence": 0.85,
    "justification": "Detailed reasoning for marketing recommendation"
}}
""",

            "decision_coordination": f"""
{base_context}

DECISION COORDINATION ACTIVITY:
Provide final integrated business decision and respond in JSON:
{{
    "final_price": 10.50,
    "final_production": 65,
    "final_marketing": 800,
    "coordination_strategy": "How activities were coordinated",
    "adjustments_made": ["list", "of", "coordination", "adjustments"],
    "overall_confidence": 0.85,
    "workflow_summary": "Summary of the complete workflow execution"
}}
"""
        }
        
        return activity_prompts.get(activity_type, activity_prompts["market_analysis"])
    
    def _parse_temporal_response(self, response_text: str, activity_type: str) -> Dict[str, Any]:
        """Parse Temporal activity response"""
        try:
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                decision_data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found")
            
            # Add common fields
            decision_data['activity_type'] = activity_type
            decision_data['framework'] = 'temporal'
            decision_data['timestamp'] = asyncio.get_event_loop().time()
            
            return decision_data
            
        except Exception as e:
            print(f"Failed to parse Temporal {activity_type} response: {e}")
            return self._fallback_activity_decision(activity_type, {})
    
    def _fallback_activity_decision(self, activity_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback decision when AI fails"""
        inventory = context.get('inventory', 100)
        
        fallback_decisions = {
            "market_analysis": {
                "market_conditions": "neutral",
                "competitive_position": "average",
                "demand_forecast": 50,
                "price_recommendation_range": [9.0, 11.0],
                "market_insights": "Fallback market analysis",
                "confidence": 0.6
            },
            "pricing_strategy": {
                "recommended_price": 10.5 if inventory < 100 else 9.8,
                "pricing_strategy": "competitive",
                "confidence": 0.6
            },
            "production_planning": {
                "recommended_production": 80 if inventory < 50 else 40,
                "production_strategy": "balanced",
                "confidence": 0.6
            },
            "marketing_optimization": {
                "recommended_marketing": 1200 if inventory > 120 else 600,
                "marketing_strategy": "targeted",
                "confidence": 0.6
            },
            "decision_coordination": {
                "final_price": 10.0,
                "final_production": 50,
                "final_marketing": 700,
                "coordination_strategy": "fallback coordination",
                "confidence": 0.6
            }
        }
        
        result = fallback_decisions.get(activity_type, fallback_decisions["market_analysis"])
        result.update({
            'activity_type': activity_type,
            'framework': 'temporal',
            'fallback': True
        })
        
        return result

# Temporal Activity Functions (using proper decorators)
async def market_analysis_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
    """Market analysis activity with Gemini AI"""
    context_dict = {
        'inventory': context.inventory,
        'current_price': context.current_price,
        'competitor_prices': context.competitor_prices,
        'turn': context.turn,
        'sales_history': context.sales_history,
        'market_trend': context.market_trend
    }
    
    return await ai_helper.get_activity_decision("market_analysis", context_dict, workflow_context)

async def pricing_strategy_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
    """Pricing strategy activity with Gemini AI"""
    context_dict = {
        'inventory': context.inventory,
        'current_price': context.current_price,
        'competitor_prices': context.competitor_prices,
        'turn': context.turn
    }
    
    return await ai_helper.get_activity_decision("pricing_strategy", context_dict, workflow_context)

async def production_planning_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
    """Production planning activity with Gemini AI"""
    context_dict = {
        'inventory': context.inventory,
        'current_price': context.current_price,
        'sales_history': context.sales_history,
        'turn': context.turn
    }
    
    return await ai_helper.get_activity_decision("production_planning", context_dict, workflow_context)

async def marketing_optimization_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
    """Marketing optimization activity with Gemini AI"""
    context_dict = {
        'inventory': context.inventory,
        'current_price': context.current_price,
        'competitor_prices': context.competitor_prices,
        'turn': context.turn
    }
    
    return await ai_helper.get_activity_decision("marketing_optimization", context_dict, workflow_context)

async def decision_coordination_activity(context: MarketContext, workflow_context: Dict[str, Any], ai_helper: TemporalGeminiHelper) -> Dict[str, Any]:
    """Decision coordination activity with Gemini AI"""
    context_dict = {
        'inventory': context.inventory,
        'current_price': context.current_price,
        'turn': context.turn
    }
    
    return await ai_helper.get_activity_decision("decision_coordination", context_dict, workflow_context)

# Apply decorators if available (this fixes the TypeError)
if activity_decorator:
    market_analysis_activity = activity_decorator(market_analysis_activity)
    pricing_strategy_activity = activity_decorator(pricing_strategy_activity)
    production_planning_activity = activity_decorator(production_planning_activity)
    marketing_optimization_activity = activity_decorator(marketing_optimization_activity)
    decision_coordination_activity = activity_decorator(decision_coordination_activity)

class BreweryBusinessWorkflow:
    """Main Temporal workflow for brewery business decisions"""
    
    def __init__(self):
        self.workflow_id = f"brewery_workflow_{random.randint(1000, 9999)}"
        self.completed_activities = []
        self.activity_results = {}
    
    async def run(self, context: MarketContext, ai_helper: TemporalGeminiHelper) -> WorkflowDecision:
        """Execute the complete business decision workflow"""
        
        workflow_context = {
            'workflow_id': self.workflow_id,
            'attempt': 1,
            'completed_activities': []
        }
        
        try:
            # Activity 1: Market Analysis
            market_result = await self._execute_activity(
                market_analysis_activity, context, workflow_context, ai_helper, "market_analysis"
            )
            
            # Activity 2: Pricing Strategy
            workflow_context['market_analysis'] = market_result
            pricing_result = await self._execute_activity(
                pricing_strategy_activity, context, workflow_context, ai_helper, "pricing_strategy"
            )
            
            # Activity 3: Production Planning
            workflow_context['pricing_strategy'] = pricing_result
            production_result = await self._execute_activity(
                production_planning_activity, context, workflow_context, ai_helper, "production_planning"
            )
            
            # Activity 4: Marketing Optimization
            workflow_context['production_planning'] = production_result
            marketing_result = await self._execute_activity(
                marketing_optimization_activity, context, workflow_context, ai_helper, "marketing_optimization"
            )
            
            # Activity 5: Decision Coordination
            workflow_context.update({
                'marketing_optimization': marketing_result
            })
            coordination_result = await self._execute_activity(
                decision_coordination_activity, context, workflow_context, ai_helper, "decision_coordination"
            )
            
            # Create final workflow decision
            return WorkflowDecision(
                price=float(coordination_result.get('final_price', pricing_result.get('recommended_price', 10.0))),
                production=int(coordination_result.get('final_production', production_result.get('recommended_production', 50))),
                marketing=int(coordination_result.get('final_marketing', marketing_result.get('recommended_marketing', 500))),
                reasoning=f"Temporal Workflow: {coordination_result.get('coordination_strategy', 'Workflow execution completed')}",
                workflow_id=self.workflow_id,
                activities_executed=self.completed_activities,
                confidence=float(coordination_result.get('overall_confidence', 0.8))
            )
            
        except Exception as e:
            print(f"Workflow execution error: {e}")
            return self._fallback_workflow_decision(context)
    
    async def _execute_activity(self, activity_func, context: MarketContext, workflow_context: Dict[str, Any], 
                              ai_helper: TemporalGeminiHelper, activity_name: str) -> Dict[str, Any]:
        """Execute a single activity with retry logic"""
        
        max_retries = TemporalGameConfig.MAX_RETRIES
        for attempt in range(max_retries):
            try:
                workflow_context['attempt'] = attempt + 1
                
                # Execute activity
                result = await activity_func(context, workflow_context, ai_helper)
                
                # Track completion
                self.completed_activities.append(activity_name)
                self.activity_results[activity_name] = result
                workflow_context['completed_activities'] = self.completed_activities
                
                return result
                
            except Exception as e:
                print(f"Activity {activity_name} attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    # Final fallback
                    fallback_result = {
                        'fallback': True,
                        'activity_type': activity_name,
                        'error': str(e)
                    }
                    self.completed_activities.append(f"{activity_name}_fallback")
                    return fallback_result
                
                await asyncio.sleep(1)  # Brief retry delay
    
    def _fallback_workflow_decision(self, context: MarketContext) -> WorkflowDecision:
        """Fallback workflow decision when everything fails"""
        inventory_ratio = context.inventory / 100.0
        
        if inventory_ratio > 1.5:
            price, production, marketing = 9.5, 40, 1200
        elif inventory_ratio < 0.5:
            price, production, marketing = 11.5, 100, 400
        else:
            price, production, marketing = 10.5, 60, 800
        
        return WorkflowDecision(
            price=price,
            production=production,
            marketing=marketing,
            reasoning="Temporal Workflow: Complete fallback decision (all activities failed)",
            workflow_id=f"fallback_{self.workflow_id}",
            activities_executed=["fallback"],
            confidence=0.5
        )

# Apply workflow decorator if available
if workflow_decorator:
    BreweryBusinessWorkflow = workflow_decorator(BreweryBusinessWorkflow)

class TemporalBrewerySystem:
    """Main Temporal brewery system integrating workflows with Gemini AI"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.ai_helper = TemporalGeminiHelper(gemini_api_key)
        self.workflow_history = []
        self.team_state = {
            'profit': 100000,
            'inventory': 100,
            'price': 10.0,
            'production': 50,
            'marketing': 500,
            'projected_demand': 50
        }
        
        print("✅ Temporal Brewery System initialized")
        print(f"🤖 Gemini AI: {'Enabled' if self.ai_helper.enabled else 'Disabled'}")
        print(f"⚡ Temporal SDK: {'Available' if TEMPORAL_AVAILABLE else 'Mock implementation'}")
    
    async def execute_business_workflow(self, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete business decision workflow"""
        
        # Create workflow context
        context = MarketContext(
            inventory=context_data.get('inventory', self.team_state['inventory']),
            current_price=context_data.get('current_price', self.team_state['price']),
            competitor_prices=context_data.get('competitor_prices', [10.0]),
            turn=context_data.get('turn', 1),
            sales_history=context_data.get('sales_history', []),
            production_history=context_data.get('production_history', []),
            market_trend=context_data.get('market_trend', 'stable')
        )
        
        # Create and execute workflow
        workflow = BreweryBusinessWorkflow()
        decision = await workflow.run(context, self.ai_helper)
        
        # Update team state
        self.team_state.update({
            'price': decision.price,
            'production': decision.production,
            'marketing': decision.marketing
        })
        
        # Store workflow history
        workflow_record = {
            'workflow_id': decision.workflow_id,
            'decision': decision,
            'activities_executed': decision.activities_executed,
            'context': context_data,
            'timestamp': asyncio.get_event_loop().time()
        }
        self.workflow_history.append(workflow_record)
        
        return {
            'price': decision.price,
            'production': decision.production,
            'marketing': decision.marketing,
            'reasoning': decision.reasoning,
            'framework': 'temporal',
            'workflow_id': decision.workflow_id,
            'activities_executed': decision.activities_executed,
            'confidence': decision.confidence,
            'ai_enhanced': self.ai_helper.enabled
        }
    
    def get_workflow_details(self) -> Dict[str, Any]:
        """Get detailed information about workflow executions"""
        if not self.workflow_history:
            return {'status': 'no_workflows_executed'}
        
        latest_workflow = self.workflow_history[-1]
        
        return {
            'total_workflows': len(self.workflow_history),
            'latest_workflow': {
                'id': latest_workflow['workflow_id'],
                'activities_count': len(latest_workflow['activities_executed']),
                'activities': latest_workflow['activities_executed'],
                'confidence': latest_workflow['decision'].confidence
            },
            'ai_enabled': self.ai_helper.enabled,
            'temporal_available': TEMPORAL_AVAILABLE,
            'average_confidence': np.mean([w['decision'].confidence for w in self.workflow_history])
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_temporal_framework():
        print("🧪 Testing Fixed Temporal Framework with Gemini AI...")
        
        # Initialize Temporal system (set your API key here)
        GEMINI_API_KEY = None  # Replace with your actual API key
        temporal_system = TemporalBrewerySystem(GEMINI_API_KEY)
        
        # Test context
        test_context = {
            'inventory': 85,
            'current_price': 10.2,
            'competitor_prices': [9.5, 11.0, 10.8],
            'turn': 3,
            'sales_history': [45, 52, 48],
            'production_history': [60, 55, 50],
            'market_trend': 'stable'
        }
        
        # Execute workflow
        decisions = await temporal_system.execute_business_workflow(test_context)
        
        print("\n📊 Temporal Framework Results:")
        print(f"💰 Price: ${decisions['price']:.2f}")
        print(f"🏭 Production: {decisions['production']} units")
        print(f"📢 Marketing: ${decisions['marketing']}")
        print(f"🤖 AI Enhanced: {decisions['ai_enhanced']}")
        print(f"⚡ Workflow ID: {decisions['workflow_id']}")
        print(f"📝 Reasoning: {decisions['reasoning']}")
        print(f"🎯 Confidence: {decisions['confidence']:.2f}")
        
        # Show workflow details
        workflow_details = temporal_system.get_workflow_details()
        print(f"\n⚡ Workflow Details:")
        print(f"- Total Workflows: {workflow_details['total_workflows']}")
        print(f"- Activities Executed: {workflow_details['latest_workflow']['activities']}")
        print(f"- Average Confidence: {workflow_details.get('average_confidence', 0):.2f}")
        print(f"- Temporal SDK: {'Available' if workflow_details['temporal_available'] else 'Mock'}")
    
    # Run test
    asyncio.run(test_temporal_framework())