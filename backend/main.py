import asyncio
import json
import os
import random
import re
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("Gemini AI library available")
except ImportError:
    GEMINI_AVAILABLE = False
    print("Gemini AI not available - using rule-based decisions")

class AgentConfig:
    """Enhanced configuration with profit optimization focus"""
    MIN_PRICE = 8.0
    MAX_PRICE = 15.0
    MIN_PRODUCTION = 10
    MAX_PRODUCTION = 150
    MIN_MARKETING = 0
    MAX_MARKETING = 2000
    UNIT_PRODUCTION_COST = 3.5  # Actual production cost per unit
    UNIT_HOLDING_COST = 0.5     # Cost to hold inventory per unit per turn
    TARGET_PROFIT_MARGIN = 0.40  # Target 40% profit margin
    DESIRED_INVENTORY_WEEKS = 2.5
    MIN_PROFITABLE_PRICE = UNIT_PRODUCTION_COST / (1 - TARGET_PROFIT_MARGIN)  # ~$5.83

class HistoricalDataTracker:
    """Tracks historical data for all teams"""
    
    def __init__(self):
        self.team_histories = {
            'green': {'sales': [], 'production': [], 'prices': [], 'marketing': [], 'profits': [], 'inventory': []},
            'blue': {'sales': [], 'production': [], 'prices': [], 'marketing': [], 'profits': [], 'inventory': []},
            'purple': {'sales': [], 'production': [], 'prices': [], 'marketing': [], 'profits': [], 'inventory': []},
            'orange': {'sales': [], 'production': [], 'prices': [], 'marketing': [], 'profits': [], 'inventory': []}
        }
        self.market_history = []  # Overall market trends
    
    def add_turn_data(self, turn: int, teams: Dict[str, Dict[str, Any]], turn_results: Dict[str, Dict[str, Any]]):
        """Add data from completed turn"""
        for color, team in teams.items():
            if color in self.team_histories:
                self.team_histories[color]['sales'].append(turn_results[color]['sales'])
                self.team_histories[color]['production'].append(team['production'])
                self.team_histories[color]['prices'].append(team['price'])
                self.team_histories[color]['marketing'].append(team['marketing'])
                self.team_histories[color]['profits'].append(turn_results[color]['profit_this_turn'])
                self.team_histories[color]['inventory'].append(team['inventory'])
                
                # Keep only last 10 turns of history
                for key in self.team_histories[color]:
                    if len(self.team_histories[color][key]) > 10:
                        self.team_histories[color][key] = self.team_histories[color][key][-10:]
        
        # Track overall market data
        total_sales = sum(turn_results[color]['sales'] for color in teams.keys())
        avg_price = sum(team['price'] for team in teams.values()) / len(teams)
        
        self.market_history.append({
            'turn': turn,
            'total_market_sales': total_sales,
            'average_market_price': avg_price
        })
        
        if len(self.market_history) > 10:
            self.market_history = self.market_history[-10:]
    
    def get_team_history(self, team_color: str) -> Dict[str, List]:
        """Get historical data for a specific team"""
        return self.team_histories.get(team_color, {})
    
    def get_market_trends(self) -> Dict[str, Any]:
        """Get market trend analysis"""
        if len(self.market_history) < 3:
            return {'trend': 'insufficient_data'}
        
        recent_sales = [h['total_market_sales'] for h in self.market_history[-3:]]
        sales_trend = 'growing' if recent_sales[-1] > recent_sales[0] + 10 else 'declining' if recent_sales[-1] < recent_sales[0] - 10 else 'stable'
        
        recent_prices = [h['average_market_price'] for h in self.market_history[-3:]]
        price_trend = 'increasing' if recent_prices[-1] > recent_prices[0] + 0.5 else 'decreasing' if recent_prices[-1] < recent_prices[0] - 0.5 else 'stable'
        
        return {
            'sales_trend': sales_trend,
            'price_trend': price_trend,
            'market_volatility': self._calculate_volatility(recent_sales)
        }
    
    def _calculate_volatility(self, data: List[float]) -> str:
        """Calculate volatility level"""
        if len(data) < 2:
            return 'unknown'
        
        avg = sum(data) / len(data)
        variance = sum((x - avg) ** 2 for x in data) / len(data)
        std_dev = variance ** 0.5
        
        volatility_ratio = std_dev / avg if avg > 0 else 0
        
        if volatility_ratio > 0.3:
            return 'high'
        elif volatility_ratio > 0.15:
            return 'medium'
        else:
            return 'low'

class ProfitOptimizedGeminiHelper:
    """Enhanced Gemini helper focused on profitable decisions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.enabled = False
        self.model = None
        self.model_name = None
        
        if api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=api_key)
                models = ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-1.0-pro']
                
                for model_name in models:
                    try:
                        test_model = genai.GenerativeModel(model_name)
                        test_response = test_model.generate_content("Hello")
                        if test_response and test_response.text:
                            self.model = test_model
                            self.model_name = model_name
                            self.enabled = True
                            print(f"Profit-optimized Gemini AI enabled with {model_name}")
                            break
                    except Exception as e:
                        print(f"Model {model_name} failed: {e}")
                        continue
                        
            except Exception as e:
                print(f"Gemini initialization failed: {e}")
        
        if not self.enabled:
            print("Using profit-optimized rule-based decisions")

    async def get_profitable_decision(self, agent_type: str, context: Dict[str, Any], 
                                    team_history: Dict[str, List], market_trends: Dict[str, Any],
                                    framework: str) -> Dict[str, Any]:
        """Get profitable decision with historical context"""
        
        if self.enabled:
            try:
                return await self._ai_profitable_decision(agent_type, context, team_history, market_trends, framework)
            except Exception as e:
                print(f"AI decision failed for {agent_type}: {e}")
                return self._profitable_fallback(agent_type, context, team_history, market_trends)
        else:
            return self._profitable_fallback(agent_type, context, team_history, market_trends)
    
    async def _ai_profitable_decision(self, agent_type: str, context: Dict[str, Any], 
                                    team_history: Dict[str, List], market_trends: Dict[str, Any], 
                                    framework: str) -> Dict[str, Any]:
        """AI decision with profit optimization and historical data"""
        
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitors = context.get('competitor_prices', [10.0])
        
        # Calculate profit history trends
        profit_trend = "unknown"
        avg_recent_profit = 0
        if len(team_history.get('profits', [])) >= 3:
            recent_profits = team_history['profits'][-3:]
            avg_recent_profit = sum(recent_profits) / len(recent_profits)
            if recent_profits[-1] > recent_profits[0] + 100:
                profit_trend = "improving"
            elif recent_profits[-1] < recent_profits[0] - 100:
                profit_trend = "declining"
            else:
                profit_trend = "stable"
        
        prompt = f"""
You are a PROFIT-FOCUSED {agent_type.upper()} agent in a {framework.upper()} framework.

CRITICAL REQUIREMENT: Your decisions MUST be profitable. Avoid losses at all costs.

CURRENT SITUATION:
- Inventory: {inventory} units
- Current Price: ${current_price:.2f}
- Competitor Prices: {competitors}
- Turn: {context.get('turn', 1)}

HISTORICAL PERFORMANCE DATA:
- Sales History (last 5 turns): {team_history.get('sales', [])[-5:]}
- Price History: {team_history.get('prices', [])[-5:]}
- Marketing History: {team_history.get('marketing', [])[-5:]}
- Profit History: {team_history.get('profits', [])[-5:]}
- Inventory History: {team_history.get('inventory', [])[-5:]}
- Recent Profit Trend: {profit_trend}
- Average Recent Profit: ${avg_recent_profit:.0f}

MARKET TRENDS:
- Sales Trend: {market_trends.get('sales_trend', 'stable')}
- Price Trend: {market_trends.get('price_trend', 'stable')}
- Market Volatility: {market_trends.get('market_volatility', 'medium')}

PROFIT OPTIMIZATION RULES:
- Minimum Profitable Price: ${AgentConfig.MIN_PROFITABLE_PRICE:.2f} (covers costs + target margin)
- Production Cost: ${AgentConfig.UNIT_PRODUCTION_COST}/unit
- Holding Cost: ${AgentConfig.UNIT_HOLDING_COST}/unit/turn
- Target Profit Margin: {AgentConfig.TARGET_PROFIT_MARGIN*100}%

DECISION CONSTRAINTS:
- Price: ${AgentConfig.MIN_PRICE:.2f}-${AgentConfig.MAX_PRICE:.2f}
- Production: {AgentConfig.MIN_PRODUCTION}-{AgentConfig.MAX_PRODUCTION} units
- Marketing: ${AgentConfig.MIN_MARKETING}-${AgentConfig.MAX_MARKETING}

PROFIT ANALYSIS FROM HISTORY:
- If recent profits are negative, prioritize cost reduction and price optimization
- If recent profits are declining, analyze what changed and reverse the trend
- Use successful patterns from profitable periods
- Learn from competitor pricing and market response

STRATEGIC FOCUS BY AGENT:
- PRICING: Set prices that ensure profitability while remaining competitive
- PRODUCTION: Produce quantities that minimize costs while meeting demand
- MARKETING: Spend on marketing only when ROI is clearly positive
- CEO: Coordinate all decisions to maximize overall profitability

EXAMPLE PROFITABLE STRATEGIES:
- High inventory + losing money = Lower price slightly, reduce production significantly, increase marketing moderately
- Low inventory + profitable = Maintain or increase price, increase production, maintain marketing
- Competitors much cheaper + losing money = Find middle ground price, optimize production costs, targeted marketing

YOUR TASK:
Analyze the historical data and current situation to make a PROFITABLE decision.
Learn from past performance patterns and market trends.
Ensure positive profit contribution.

RESPOND IN JSON:
{{"price": 10.50, "production": 60, "marketing": 800, "reasoning": "Detailed profit-focused strategy based on historical analysis"}}
"""
        
        response = await asyncio.get_event_loop().run_in_executor(
            None, lambda: self.model.generate_content(prompt)
        )
        
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            
            # Ensure profitable pricing
            proposed_price = float(data.get('price', 10))
            min_profitable = max(AgentConfig.MIN_PROFITABLE_PRICE, AgentConfig.MIN_PRICE)
            price = max(min_profitable, min(AgentConfig.MAX_PRICE, proposed_price))
            
            production = max(AgentConfig.MIN_PRODUCTION, min(AgentConfig.MAX_PRODUCTION, int(data.get('production', 50))))
            marketing = max(AgentConfig.MIN_MARKETING, min(AgentConfig.MAX_MARKETING, int(data.get('marketing', 500))))
            
            return {
                'price': round(price, 2),
                'production': production,
                'marketing': marketing,
                'reasoning': f"{agent_type.upper()} AI: {data.get('reasoning', 'Profit-optimized decision')}",
                'agent_type': agent_type,
                'ai_enhanced': True,
                'profit_focused': True
            }
        else:
            raise ValueError("No JSON found in response")
    
    def _profitable_fallback(self, agent_type: str, context: Dict[str, Any], 
                           team_history: Dict[str, List], market_trends: Dict[str, Any]) -> Dict[str, Any]:
        """Profit-focused fallback decisions using historical data"""
        
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitors = context.get('competitor_prices', [10.0])
        avg_competitor = sum(competitors) / len(competitors)
        
        # Analyze profit history
        recent_profits = team_history.get('profits', [])
        recent_sales = team_history.get('sales', [])
        recent_prices = team_history.get('prices', [])
        
        # Determine if we've been profitable
        recently_profitable = True
        if len(recent_profits) >= 2:
            recently_profitable = sum(recent_profits[-2:]) > 0
        
        # Calculate minimum profitable price
        min_profitable_price = max(AgentConfig.MIN_PROFITABLE_PRICE, AgentConfig.MIN_PRICE)
        
        if agent_type == "pricing":
            # Pricing focused on profitability
            if not recently_profitable:
                # If losing money, increase price for better margins
                price = max(min_profitable_price, min(avg_competitor + 0.5, AgentConfig.MAX_PRICE))
                strategy = "Profit recovery: increased price to ensure margins"
            elif inventory > 150:
                # High inventory but need to stay profitable
                price = max(min_profitable_price, avg_competitor - 0.3)
                strategy = "Moderate discount while maintaining profitability"
            elif inventory < 50:
                # Low inventory - premium pricing opportunity
                price = min(AgentConfig.MAX_PRICE, avg_competitor + random.uniform(0.5, 1.2))
                strategy = "Premium pricing for scarcity value"
            else:
                # Normal conditions - competitive but profitable
                price = max(min_profitable_price, avg_competitor + random.uniform(-0.1, 0.3))
                strategy = "Competitive pricing with profit protection"
            
            return {
                'price': round(price, 2),
                'reasoning': f'Pricing Agent: {strategy}',
                'agent_type': 'pricing',
                'ai_enhanced': False,
                'profit_focused': True,
                'min_profitable_price': min_profitable_price
            }
        
        elif agent_type == "production":
            # Production focused on cost efficiency
            
            # Analyze sales trend from history
            expected_sales = 50  # default
            if len(recent_sales) >= 3:
                # Use weighted average of recent sales
                weights = [0.2, 0.3, 0.5]
                expected_sales = sum(sale * weight for sale, weight in zip(recent_sales[-3:], weights))
            elif recent_sales:
                expected_sales = sum(recent_sales) / len(recent_sales)
            
            # Calculate optimal inventory target
            target_inventory = expected_sales * AgentConfig.DESIRED_INVENTORY_WEEKS
            
            # Calculate production need
            inventory_gap = target_inventory - inventory
            base_production = max(0, inventory_gap) + expected_sales
            
            # Adjust production based on profit history
            if not recently_profitable:
                # If losing money, be more conservative with production
                production = max(AgentConfig.MIN_PRODUCTION, min(AgentConfig.MAX_PRODUCTION, int(base_production * 0.8)))
                strategy = "Conservative production to reduce costs during profit recovery"
            elif inventory > 150:
                # High inventory - minimize production
                production = max(AgentConfig.MIN_PRODUCTION, min(50, int(expected_sales * 0.7)))
                strategy = "Minimal production to reduce holding costs"
            else:
                # Normal production planning
                production = max(AgentConfig.MIN_PRODUCTION, min(AgentConfig.MAX_PRODUCTION, int(base_production)))
                strategy = "Demand-responsive production for optimal inventory"
            
            return {
                'production': production,
                'reasoning': f'Production Agent: {strategy}',
                'agent_type': 'production',
                'ai_enhanced': False,
                'profit_focused': True,
                'expected_sales': round(expected_sales, 1),
                'target_inventory': round(target_inventory, 1)
            }
        
        elif agent_type == "marketing":
            # Marketing focused on positive ROI
            
            # Analyze marketing effectiveness from history
            marketing_effectiveness = 1.0  # default
            if len(recent_sales) >= 2 and len(team_history.get('marketing', [])) >= 2:
                # Simple effectiveness calculation
                sales_change = recent_sales[-1] - recent_sales[-2] if len(recent_sales) >= 2 else 0
                marketing_last = team_history['marketing'][-1] if team_history.get('marketing') else 500
                if marketing_last > 0:
                    marketing_effectiveness = max(0.3, min(2.0, 1.0 + sales_change / (marketing_last / 100)))
            
            # Calculate ROI-based marketing spend
            if not recently_profitable:
                # If losing money, reduce marketing spend
                marketing = max(100, min(600, random.randint(200, 500)))
                strategy = "Reduced marketing spend during profit recovery"
            elif inventory > 150:
                # High inventory - invest in marketing but watch ROI
                expected_roi = (inventory * current_price * 0.1) / 1000  # Expected return per $1000 marketing
                if expected_roi > 1.2:  # Only if ROI looks good
                    marketing = random.randint(1000, 1500)
                    strategy = "Increased marketing with positive ROI projection"
                else:
                    marketing = random.randint(600, 1000)
                    strategy = "Moderate marketing due to uncertain ROI"
            elif inventory < 50:
                # Low inventory - minimal marketing
                marketing = random.randint(100, 400)
                strategy = "Minimal marketing due to low inventory"
            else:
                # Normal marketing with effectiveness adjustment
                base_marketing = 800
                adjusted_marketing = int(base_marketing * marketing_effectiveness)
                marketing = max(AgentConfig.MIN_MARKETING, min(AgentConfig.MAX_MARKETING, adjusted_marketing))
                strategy = f"ROI-optimized marketing (effectiveness: {marketing_effectiveness:.1f}x)"
            
            return {
                'marketing': marketing,
                'reasoning': f'Marketing Agent: {strategy}',
                'agent_type': 'marketing',
                'ai_enhanced': False,
                'profit_focused': True,
                'marketing_effectiveness': round(marketing_effectiveness, 2)
            }
        
        else:  # CEO
            # CEO makes profit-optimized coordinated decisions
            
            # Analyze overall profitability trend
            total_recent_profit = sum(recent_profits[-3:]) if len(recent_profits) >= 3 else 0
            
            if total_recent_profit < -500:  # Significant losses
                # Crisis mode: Focus on immediate profitability
                price = max(min_profitable_price + 0.5, min(AgentConfig.MAX_PRICE, avg_competitor + 0.3))
                production = max(AgentConfig.MIN_PRODUCTION, min(60, int(inventory * 0.3)))  # Reduce production
                marketing = max(100, min(500, int(inventory * 2)))  # Minimal marketing
                strategy = "Crisis management: Prioritize immediate profit recovery"
                
            elif total_recent_profit < 0:  # Recent losses
                # Recovery mode: Careful profit optimization
                price = max(min_profitable_price, avg_competitor)
                production = max(AgentConfig.MIN_PRODUCTION, min(80, int(expected_sales * 1.2)))
                marketing = max(200, min(800, random.randint(400, 700)))
                strategy = "Recovery mode: Balanced approach to restore profitability"
                
            elif inventory > 180:  # Profitable but excess inventory
                # Inventory management while maintaining profits
                price = max(min_profitable_price, avg_competitor - 0.2)
                production = AgentConfig.MIN_PRODUCTION
                marketing = min(AgentConfig.MAX_MARKETING, random.randint(1200, 1800))
                strategy = "Profitable inventory clearance: Maintain margins while moving stock"
                
            elif inventory < 40:  # Low inventory opportunity
                # Capitalize on scarcity
                price = min(AgentConfig.MAX_PRICE, avg_competitor + random.uniform(0.8, 1.5))
                production = min(AgentConfig.MAX_PRODUCTION, random.randint(100, 140))
                marketing = random.randint(200, 500)
                strategy = "Scarcity premium: Maximize margins with limited supply"
                
            else:  # Normal profitable operations
                # Optimize for sustained profitability
                price = max(min_profitable_price, avg_competitor + random.uniform(-0.1, 0.4))
                production = random.randint(50, 90)
                marketing = random.randint(600, 1100)
                strategy = "Sustainable profitability: Balanced optimization"
            
            # Calculate expected profit for validation
            expected_demand = 50 + (marketing / 500) * 15 + (10 - price) * 5
            expected_sales = min(expected_demand, inventory + production)
            expected_revenue = expected_sales * price
            expected_costs = production * AgentConfig.UNIT_PRODUCTION_COST + marketing + inventory * AgentConfig.UNIT_HOLDING_COST
            expected_profit = expected_revenue - expected_costs
            
            # If expected profit is negative, adjust strategy
            if expected_profit < 0:
                print(f"Adjusting {agent_type} decision - expected profit was negative")
                price = min(AgentConfig.MAX_PRICE, price + 0.5)  # Increase price
                production = max(AgentConfig.MIN_PRODUCTION, int(production * 0.8))  # Reduce production
                marketing = max(AgentConfig.MIN_MARKETING, int(marketing * 0.7))  # Reduce marketing
                strategy += " (Adjusted for positive profit)"
            
            return {
                'price': round(price, 2),
                'production': max(AgentConfig.MIN_PRODUCTION, min(AgentConfig.MAX_PRODUCTION, production)),
                'marketing': max(AgentConfig.MIN_MARKETING, min(AgentConfig.MAX_MARKETING, marketing)),
                'reasoning': f'{agent_type.upper()} Agent: {strategy}',
                'agent_type': agent_type,
                'ai_enhanced': False,
                'profit_focused': True,
                'expected_profit': round(expected_profit, 0),
                'profit_trend': profit_trend
            }

class CommonAgentManager:
    """Enhanced agent manager with historical data tracking"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.ai_helper = ProfitOptimizedGeminiHelper(gemini_api_key)
        self.history_tracker = HistoricalDataTracker()
        print("Enhanced Agent Manager with profit optimization initialized")
    
    async def get_framework_decision(self, framework: str, context: Dict[str, Any], 
                                   team_color: str) -> Dict[str, Any]:
        """Get framework decision with historical context"""
        
        # Get historical data for this team
        team_history = self.history_tracker.get_team_history(team_color)
        market_trends = self.history_tracker.get_market_trends()
        
        # Add historical context to decision context
        enhanced_context = context.copy()
        enhanced_context['team_history'] = team_history
        enhanced_context['market_trends'] = market_trends
        
        print(f"Getting {framework} decision with {len(team_history.get('profits', []))} turns of history...")
        
        if framework == "mesa":
            # Mesa: Parallel specialists + coordination
            pricing_task = asyncio.create_task(
                self.ai_helper.get_profitable_decision("pricing", enhanced_context, team_history, market_trends, framework)
            )
            production_task = asyncio.create_task(
                self.ai_helper.get_profitable_decision("production", enhanced_context, team_history, market_trends, framework)
            )
            marketing_task = asyncio.create_task(
                self.ai_helper.get_profitable_decision("marketing", enhanced_context, team_history, market_trends, framework)
            )
            
            pricing_dec, production_dec, marketing_dec = await asyncio.gather(pricing_task, production_task, marketing_task)
            
            # CEO coordinates with profit focus
            coord_context = enhanced_context.copy()
            coord_context['specialist_proposals'] = {
                'pricing': pricing_dec, 'production': production_dec, 'marketing': marketing_dec
            }
            
            final_decision = await self.ai_helper.get_profitable_decision("ceo", coord_context, team_history, market_trends, framework)
            
            return {
                'price': final_decision['price'],
                'production': final_decision['production'],
                'marketing': final_decision['marketing'],
                'reasoning': f"Mesa MAS: {final_decision['reasoning']}",
                'framework': 'mesa',
                'expected_profit': final_decision.get('expected_profit', 0)
            }
        
        elif framework == "temporal":
            # Temporal: Sequential workflow with profit optimization
            
            # Activity 1: Market analysis
            market_analysis = await self.ai_helper.get_profitable_decision(
                "pricing", enhanced_context, team_history, market_trends, framework
            )
            
            # Activity 2: Production planning
            prod_context = enhanced_context.copy()
            prod_context['market_analysis'] = market_analysis
            production_plan = await self.ai_helper.get_profitable_decision(
                "production", prod_context, team_history, market_trends, framework
            )
            
            # Activity 3: Marketing optimization
            mkt_context = enhanced_context.copy()
            mkt_context['production_plan'] = production_plan
            marketing_plan = await self.ai_helper.get_profitable_decision(
                "marketing", mkt_context, team_history, market_trends, framework
            )
            
            # Activity 4: Strategic coordination
            final_context = enhanced_context.copy()
            final_context['workflow_results'] = {
                'market_analysis': market_analysis,
                'production_plan': production_plan,
                'marketing_plan': marketing_plan
            }
            final_decision = await self.ai_helper.get_profitable_decision(
                "ceo", final_context, team_history, market_trends, framework
            )
            
            return {
                'price': final_decision['price'],
                'production': final_decision['production'],
                'marketing': final_decision['marketing'],
                'reasoning': f"Temporal Workflow: {final_decision['reasoning']}",
                'framework': 'temporal',
                'expected_profit': final_decision.get('expected_profit', 0)
            }
        
        else:  # google_adk
            # Google ADK: ML pipeline with profit optimization
            
            # ML Service 1: Vertex AI prediction
            vertex_context = enhanced_context.copy()
            vertex_context['ml_service'] = 'vertex_ai'
            vertex_pred = await self.ai_helper.get_profitable_decision(
                "pricing", vertex_context, team_history, market_trends, framework
            )
            
            # ML Service 2: AutoML optimization
            automl_context = enhanced_context.copy()
            automl_context['vertex_result'] = vertex_pred
            automl_context['ml_service'] = 'automl'
            automl_opt = await self.ai_helper.get_profitable_decision(
                "production", automl_context, team_history, market_trends, framework
            )
            
            # ML Service 3: BigQuery analytics
            bigquery_context = enhanced_context.copy()
            bigquery_context['ml_results'] = {'vertex': vertex_pred, 'automl': automl_opt}
            bigquery_context['ml_service'] = 'bigquery'
            bigquery_analytics = await self.ai_helper.get_profitable_decision(
                "marketing", bigquery_context, team_history, market_trends, framework
            )
            
            # ML Service 4: AI Platform ensemble
            ensemble_context = enhanced_context.copy()
            ensemble_context['ml_pipeline_complete'] = {
                'vertex': vertex_pred, 'automl': automl_opt, 'bigquery': bigquery_analytics
            }
            ensemble_context['ml_service'] = 'ensemble'
            ensemble_decision = await self.ai_helper.get_profitable_decision(
                "ceo", ensemble_context, team_history, market_trends, framework
            )
            
            return {
                'price': ensemble_decision['price'],
                'production': ensemble_decision['production'],
                'marketing': ensemble_decision['marketing'],
                'reasoning': f"Google ADK ML: {ensemble_decision['reasoning']}",
                'framework': 'google_adk',
                'expected_profit': ensemble_decision.get('expected_profit', 0)
            }

class BrewMastersGame:
    """Enhanced game with historical tracking and profit optimization"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.agent_manager = CommonAgentManager(gemini_api_key)
        self.turn = 0
        self.competition_mode = False
        
        self.teams = {
            'green': {'name': 'Human', 'profit': 100000, 'inventory': 100, 'price': 10.0, 'production': 50, 'marketing': 500, 'profit_this_turn': 0},
            'blue': {'name': 'Mesa MAS', 'profit': 100000, 'inventory': 100, 'price': 10.0, 'production': 50, 'marketing': 500, 'profit_this_turn': 0},
            'purple': {'name': 'Temporal Workflow', 'profit': 100000, 'inventory': 100, 'price': 10.0, 'production': 50, 'marketing': 500, 'profit_this_turn': 0},
            'orange': {'name': 'Google ADK ML', 'profit': 100000, 'inventory': 100, 'price': 10.0, 'production': 50, 'marketing': 500, 'profit_this_turn': 0}
        }
        
        ai_status = "Profit-Optimized Gemini" if self.agent_manager.ai_helper.enabled else "Profit-Optimized Rule-based"
        self.event_log = [
            "Multi-Framework BrewMasters with Profit Optimization Started!",
            f"AI Status: {ai_status}",
            "Enhanced Features: Historical data tracking + Profit focus",
            "All decisions optimized for profitability!"
        ]
        
        print(f"BrewMasters with profit optimization initialized")
    
    async def process_turn(self, human_decisions: Dict[str, Any]) -> Dict[str, Any]:
        self.turn += 1
        print(f"\n=== PROCESSING TURN {self.turn} WITH PROFIT OPTIMIZATION ===")
        
        # Human decisions
        human_team = self.teams['green']
        human_team.update({
            'price': float(human_decisions.get('price', 10)),
            'production': int(human_decisions.get('productionTarget', 50)),
            'marketing': int(human_decisions.get('marketingSpend', 500))
        })
        
        # AI framework decisions with historical context
        base_context = {
            'turn': self.turn,
            'competitor_prices': [team['price'] for team in self.teams.values()]
        }
        
        frameworks = [('blue', 'mesa'), ('purple', 'temporal'), ('orange', 'google_adk')]
        
        for color, framework in frameworks:
            context = base_context.copy()
            context.update({
                'inventory': self.teams[color]['inventory'],
                'current_price': self.teams[color]['price'],
                'profit': self.teams[color]['profit']
            })
            
            decision = await self.agent_manager.get_framework_decision(framework, context, color)
            
            self.teams[color].update({
                'price': decision['price'],
                'production': decision['production'],
                'marketing': decision['marketing']
            })
            
            expected_profit = decision.get('expected_profit', 'unknown')
            print(f"{framework}: ${decision['price']:.2f}, {decision['production']}u, ${decision['marketing']}m (Expected profit: {expected_profit})")
        
        # Calculate results with enhanced profit tracking
        turn_results = {}
        for color, team in self.teams.items():
            # Enhanced demand calculation
            demand = 50 + (team['marketing'] / 500) * 15 + (10 - team['price']) * 5 + random.randint(-8, 8)
            demand = max(10, min(120, int(demand)))
            
            sales = min(demand, team['inventory'])
            revenue = sales * team['price']
            
            # Detailed cost calculation
            production_costs = team['production'] * AgentConfig.UNIT_PRODUCTION_COST
            marketing_costs = team['marketing']
            holding_costs = team['inventory'] * AgentConfig.UNIT_HOLDING_COST
            total_costs = production_costs + marketing_costs + holding_costs
            
            profit_this_turn = revenue - total_costs
            
            # Update team state
            team['inventory'] = max(0, team['inventory'] - sales + team['production'])
            team['profit'] += profit_this_turn
            team['profit_this_turn'] = profit_this_turn
            
            turn_results[color] = {
                'sales': sales,
                'demand': demand,
                'revenue': revenue,
                'costs': total_costs,
                'profit_this_turn': profit_this_turn
            }
            
            # Log profit warnings
            if profit_this_turn < 0:
                print(f"WARNING: {team['name']} made a loss of ${profit_this_turn:.0f} this turn!")
        
        # Add turn data to historical tracker
        self.agent_manager.history_tracker.add_turn_data(self.turn, self.teams, turn_results)
        
        # Update event log with profit focus
        best_ai = max([
            ('Mesa', turn_results['blue']['profit_this_turn']),
            ('Temporal', turn_results['purple']['profit_this_turn']),
            ('Google ADK', turn_results['orange']['profit_this_turn'])
        ], key=lambda x: x[1])
        
        # Count profitable teams
        profitable_teams = len([result for result in turn_results.values() if result['profit_this_turn'] > 0])
        
        self.event_log = [
            f"Turn {self.turn} - Profit-Optimized Framework Battle",
            f"Human: ${human_team['price']:.2f} -> {turn_results['green']['sales']} sales, ${turn_results['green']['profit_this_turn']:+.0f} profit",
            f"Mesa: ${self.teams['blue']['price']:.2f} -> {turn_results['blue']['sales']} sales, ${turn_results['blue']['profit_this_turn']:+.0f} profit",
            f"Temporal: ${self.teams['purple']['price']:.2f} -> {turn_results['purple']['sales']} sales, ${turn_results['purple']['profit_this_turn']:+.0f} profit",
            f"Google ADK: ${self.teams['orange']['price']:.2f} -> {turn_results['orange']['sales']} sales, ${turn_results['orange']['profit_this_turn']:+.0f} profit",
            f"Best Framework: {best_ai[0]} (${best_ai[1]:+.0f} profit)",
            f"Profitable Teams: {profitable_teams}/4",
            f"Intelligence: {'Gemini AI' if self.agent_manager.ai_helper.enabled else 'Rule-based'} with profit optimization"
        ]
        
        print(f"Turn {self.turn} complete - {profitable_teams}/4 teams profitable")
        return self.get_game_state()
    
    def get_game_state(self) -> Dict[str, Any]:
        state = {
            'turn': self.turn,
            'competition_mode': self.competition_mode,
            'event_log': self.event_log
        }
        
        for color, team in self.teams.items():
            prefix = f"{color}_team"
            for key, value in team.items():
                if key != 'name':
                    state[f"{prefix}_{key}"] = value
        
        # Add historical data summary
        state['historical_summary'] = {
            'turns_played': self.turn,
            'market_trends': self.agent_manager.history_tracker.get_market_trends(),
            'profit_optimization_active': True
        }
        
        return state

# FastAPI setup
app = FastAPI(title="BrewMasters Profit-Optimized with Historical Data")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game: Optional[BrewMastersGame] = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    global game
    if game is None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        game = BrewMastersGame(gemini_api_key)
    
    try:
        initial_state = game.get_game_state()
        initial_state['server_info'] = {
            'version': '3.1.0-profit-optimized',
            'features': ['Historical Data Tracking', 'Profit Optimization', 'Loss Prevention'],
            'ai_enabled': game.agent_manager.ai_helper.enabled
        }
        await websocket.send_text(json.dumps(initial_state))
        print("Initial state sent")
        
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                
                if data.get('restart'):
                    game = BrewMastersGame(os.getenv("GEMINI_API_KEY"))
                    await websocket.send_text(json.dumps(game.get_game_state()))
                    continue
                
                updated_state = await game.process_turn(data)
                await websocket.send_text(json.dumps(updated_state))
                
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_text(json.dumps({"error": str(e)}))
    
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/")
async def read_root():
    return {
        "message": "BrewMasters Profit-Optimized with Historical Data",
        "version": "3.1.0-profit-optimized",
        "features": [
            "Historical data tracking for all teams",
            "Profit-optimized agent decisions", 
            "Loss prevention mechanisms",
            "Enhanced AI prompts with historical context"
        ],
        "ai_enabled": game.agent_manager.ai_helper.enabled if game else False
    }

if __name__ == "__main__":
    print("Starting BrewMasters Profit-Optimized System...")
    print("Features:")
    print("- Historical data tracking")
    print("- Profit optimization focus") 
    print("- Loss prevention")
    print("- Enhanced AI prompts")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        print(f"Gemini API key found - AI agents will be profit-optimized")
    else:
        print("No API key - using profit-optimized rule-based decisions")
    
    print("All agents now focus on profitability and use historical data!")
    uvicorn.run(app, host="0.0.0.0", port=8000)