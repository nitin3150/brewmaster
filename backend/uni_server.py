# import asyncio
# import json
# import random
# import numpy as np
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from typing import Dict, Any, Optional, List
# import uvicorn

# # Import Mesa components
# from model import BrewMastersModel
# from agent import SharedKnowledgeBase

# # Import Temporal components (assuming they're in a temporal_impl module)
# try:
#     from temporalio.client import Client
#     from temporalio.worker import Worker
#     from main import (GameSession as TemporalGameSession, 
#                      MASDecisionWorkflow, GameTurnWorkflow,
#                      create_shared_knowledge, pricing_agent_activity,
#                      marketing_agent_activity, production_agent_activity,
#                      ceo_decision_activity, calculate_game_outcome)
#     TEMPORAL_AVAILABLE = True
# except ImportError:
#     TEMPORAL_AVAILABLE = False
#     print("WARNING: Temporal not installed. Install with: pip install temporalio")

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class DemandSimulator:
#     """Simulates realistic market demand with randomness"""
    
#     def __init__(self):
#         self.base_demand = 50
#         self.market_sentiment = 0.0  # -1 to 1, affects overall market
#         self.trend_momentum = 0.0    # Carries momentum from previous turns
#         self.seasonal_factor = 1.0   # Could vary by "season"
#         self.competitor_effects = {}  # Track how competitors affect each other
        
#     def calculate_demand(self, price: float, marketing_spend: float, 
#                         competitor_prices: List[float], turn: int,
#                         previous_sales: List[int] = None) -> int:
#         """
#         Calculate demand with realistic randomness and market dynamics
#         """
#         # Update market sentiment randomly each turn
#         sentiment_change = random.gauss(0, 0.1)  # Small random walk
#         self.market_sentiment = max(-1, min(1, self.market_sentiment + sentiment_change))
        
#         # Base demand with market sentiment
#         base = self.base_demand * (1 + self.market_sentiment * 0.3)
        
#         # Price elasticity with some randomness
#         avg_competitor_price = np.mean(competitor_prices) if competitor_prices else price
#         price_competitiveness = (avg_competitor_price - price) / avg_competitor_price
        
#         # Non-linear price effect with noise
#         price_effect = price_competitiveness * random.uniform(15, 25)  # Was fixed at 20
        
#         # Marketing effectiveness with diminishing returns and randomness
#         marketing_effectiveness = random.uniform(0.02, 0.04)  # Was fixed at 0.03
#         marketing_boost = np.sqrt(marketing_spend) * marketing_effectiveness
        
#         # Trend momentum from previous sales
#         if previous_sales and len(previous_sales) >= 2:
#             recent_trend = previous_sales[-1] - previous_sales[-2]
#             self.trend_momentum = 0.7 * self.trend_momentum + 0.3 * (recent_trend / 10)
#             trend_effect = self.trend_momentum * random.uniform(3, 7)
#         else:
#             trend_effect = 0
        
#         # Random events (20% chance)
#         random_event = 0
#         if random.random() < 0.2:
#             # Random demand spike or drop
#             random_event = random.choice([-15, -10, -5, 5, 10, 15, 20])
        
#         # Weekend/weekday effect (simulate turns as days)
#         if turn % 7 in [5, 6]:  # "Weekend" turns
#             weekend_boost = random.uniform(5, 15)
#         else:
#             weekend_boost = 0
        
#         # Calculate total demand
#         total_demand = (
#             base + 
#             price_effect + 
#             marketing_boost + 
#             trend_effect + 
#             random_event + 
#             weekend_boost
#         )
        
#         # Add final random noise
#         noise = random.gauss(0, 5)
#         total_demand += noise
        
#         # Ensure reasonable bounds
#         return max(10, min(150, int(total_demand)))

# class UnifiedGameSession:
#     """Manages all three game implementations simultaneously"""
    
#     def __init__(self):
#         self.turn = 0
#         self.temporal_client: Optional[Client] = None
#         self.demand_simulator = DemandSimulator()
#         self.compete_mode = False
#         self.compete_turns_total = 0
#         self.compete_turns_done = 0
        
#         # Track sales history for each team
#         self.sales_history = {
#             'green': [],
#             'blue': [],
#             'purple': []
#         }
        
#         # Initialize game states for all three teams
#         self.game_state = {
#             "turn": 0,
#             # Human Team (Green)
#             "green_team_profit": 100000,
#             "green_team_inventory": 100,
#             "green_team_price": 10,
#             "green_team_projected_demand": 50,
#             "green_team_production_target": 50,
#             "green_team_marketing_spend": 500,
#             "green_team_profit_this_turn": 0,
#             # Mesa MAS Team (Blue)
#             "blue_team_profit": 100000,
#             "blue_team_inventory": 100,
#             "blue_team_price": 10,
#             "blue_team_projected_demand": 50,
#             "blue_team_production_target": 50,
#             "blue_team_marketing_spend": 500,
#             "blue_team_profit_this_turn": 0,
#             # Temporal MAS Team (Purple)
#             "purple_team_profit": 100000,
#             "purple_team_inventory": 100,
#             "purple_team_price": 10,
#             "purple_team_projected_demand": 50,
#             "purple_team_production_target": 50,
#             "purple_team_marketing_spend": 500,
#             "purple_team_profit_this_turn": 0,
#             # Event logs
#             "event_log": ["Game Started! Three-way competition between Human, Mesa MAS, and Temporal MAS"]
#         }
        
#         # Initialize Mesa model
#         self.mesa_model = BrewMastersModel()
        
#         # Initialize Temporal session (will be set up asynchronously)
#         self.temporal_session: Optional[TemporalGameSession] = None
        
#     async def initialize_temporal(self):
#         """Initialize Temporal connection"""
#         # For now, always use fallback AI to avoid connection issues
#         print("Using fallback AI for Temporal team (purple)")
#         return False
        
#         # Original Temporal connection code disabled
#         # Uncomment below if you want to try Temporal again later
#         """
#         if not TEMPORAL_AVAILABLE:
#             print("Temporal library not available")
#             return False
            
#         try:
#             # Try to connect to Temporal with a timeout
#             print("Attempting to connect to Temporal server at localhost:7233...")
#             import asyncio
            
#             async def try_connect():
#                 self.temporal_client = await Client.connect("localhost:7233")
#                 self.temporal_session = TemporalGameSession(self.temporal_client)
#                 return True
            
#             # Try to connect with a 3-second timeout
#             result = await asyncio.wait_for(try_connect(), timeout=3.0)
#             print("Successfully connected to Temporal!")
#             return result
            
#         except asyncio.TimeoutError:
#             print("Temporal connection timed out after 3 seconds - will use fallback AI")
#             print("Make sure Temporal server is running: temporal server start-dev")
#             return False
#         except Exception as e:
#             print(f"Failed to connect to Temporal: {e}")
#             print("Make sure Temporal server is running: temporal server start-dev")
#             return False
#         """
    
#     def calculate_outcome(self, team_prefix: str, price: float, marketing_spend: float, production_target: int) -> Dict[str, Any]:
#         """Calculate sales and profit for one team using random demand"""
#         # Get competitor prices
#         all_prices = [
#             self.game_state['green_team_price'],
#             self.game_state['blue_team_price'],
#             self.game_state['purple_team_price']
#         ]
        
#         # Map team prefix to color for sales history
#         team_color_map = {
#             'green_team': 'green',
#             'blue_team': 'blue', 
#             'purple_team': 'purple'
#         }
#         team_color = team_color_map.get(team_prefix, 'green')
        
#         # Remove this team's price from competitor list
#         competitor_prices = [p for p in all_prices if p != price]
        
#         # Calculate demand using the simulator
#         demand = self.demand_simulator.calculate_demand(
#             price=price,
#             marketing_spend=marketing_spend,
#             competitor_prices=competitor_prices,
#             turn=self.turn,
#             previous_sales=self.sales_history[team_color][-5:] if self.sales_history[team_color] else None
#         )
        
#         inventory_key = f"{team_prefix}_inventory"
#         profit_key = f"{team_prefix}_profit"
        
#         sales = min(demand, self.game_state[inventory_key])
#         revenue = sales * price
#         production_cost = production_target * 3
#         inventory_cost = self.game_state[inventory_key] * 0.5
#         profit_this_turn = revenue - production_cost - inventory_cost - marketing_spend
        
#         # Update state
#         self.game_state[profit_key] = float(self.game_state[profit_key] + profit_this_turn)
#         self.game_state[inventory_key] -= sales
#         self.game_state[inventory_key] += production_target
#         self.game_state[f"{team_prefix}_price"] = float(price)
#         self.game_state[f"{team_prefix}_profit_this_turn"] = float(profit_this_turn)
#         self.game_state[f"{team_prefix}_production_target"] = int(production_target)
#         self.game_state[f"{team_prefix}_marketing_spend"] = float(marketing_spend)
#         self.game_state[f"{team_prefix}_actual_demand"] = int(demand)
        
#         # Track sales history
#         self.sales_history[team_color].append(sales)
        
#         return {
#             'sales': sales,
#             'profit_this_turn': profit_this_turn,
#             'demand': demand
#         }
    
#     async def process_turn(self, human_decisions: Dict[str, Any]) -> Dict[str, Any]:
#         """Process one turn for all three teams"""
#         print(f"Starting turn processing for turn {self.turn + 1}")
        
#         self.turn += 1
#         self.game_state['turn'] = self.turn
#         self.game_state['event_log'] = [f"--- Turn {self.turn} ---"]
        
#         # Add market sentiment info to log
#         sentiment_str = "bearish" if self.demand_simulator.market_sentiment < -0.3 else \
#                        "bullish" if self.demand_simulator.market_sentiment > 0.3 else "neutral"
#         self.game_state['event_log'].append(f"Market sentiment: {sentiment_str}")
        
#         print(f"Processing human team...")
        
#         # 1. Process Human Team (Green)
#         human_outcome = self.calculate_outcome(
#             "green_team",
#             float(human_decisions.get('price', 10)),
#             float(human_decisions.get('marketingSpend', 500)),
#             int(human_decisions.get('productionTarget', 50))
#         )
#         self.game_state['event_log'].append(
#             f"Human: Sold {human_outcome['sales']} units (Demand: {human_outcome['demand']})"
#         )
        
#         print(f"Human outcome: {human_outcome}")
#         print(f"Processing Mesa MAS team...")
        
#         # 2. Process Mesa MAS Team (Blue) - Independent instance
#         # Create a complete game state for Mesa with all required fields
#         mesa_game_state = {
#             'turn': self.turn,
#             # Mesa team (Blue) data
#             'blue_team_profit': self.game_state['blue_team_profit'],
#             'blue_team_inventory': self.game_state['blue_team_inventory'],
#             'blue_team_price': self.game_state['blue_team_price'],
#             'blue_team_projected_demand': self.game_state['blue_team_projected_demand'],
#             'blue_team_production_target': self.game_state.get('blue_team_production_target', 50),
#             'blue_team_marketing_spend': self.game_state.get('blue_team_marketing_spend', 500),
#             # Human team (Green) data - Mesa needs this for competition analysis
#             'green_team_profit': self.game_state['green_team_profit'],
#             'green_team_inventory': self.game_state['green_team_inventory'],
#             'green_team_price': self.game_state['green_team_price'],
#             'green_team_projected_demand': self.game_state['green_team_projected_demand'],
#             'green_team_production_target': self.game_state.get('green_team_production_target', 50),
#             'green_team_marketing_spend': self.game_state.get('green_team_marketing_spend', 500),
#             'event_log': []
#         }
        
#         # Update Mesa model's state
#         self.mesa_model.game_state = mesa_game_state
#         self.mesa_model.turn = self.turn
        
#         # Run Mesa MAS decision making
#         mesa_dummy_human = {
#             'price': self.game_state['green_team_price'],
#             'marketingSpend': self.game_state['green_team_marketing_spend'],
#             'productionTarget': self.game_state['green_team_production_target']
#         }
        
#         # Save original green team values to prevent Mesa from modifying them
#         original_green_profit = mesa_game_state['green_team_profit']
#         original_green_inventory = mesa_game_state['green_team_inventory']
        
#         self.mesa_model.step(mesa_dummy_human)
        
#         # Restore green team values (Mesa should only modify blue team)
#         self.mesa_model.game_state['green_team_profit'] = original_green_profit
#         self.mesa_model.game_state['green_team_inventory'] = original_green_inventory
        
#         # Get Mesa's decisions
#         mesa_decisions = self.mesa_model.ceo_agent.final_decisions
        
#         # Calculate Mesa outcome
#         mesa_outcome = self.calculate_outcome(
#             "blue_team",
#             mesa_decisions['price'],
#             mesa_decisions['marketing_spend'],
#             mesa_decisions['production_target']
#         )
#         # Ensure all values are JSON serializable
#         mesa_outcome['profit_this_turn'] = float(mesa_outcome['profit_this_turn'])
        
#         self.game_state['event_log'].append(
#             f"Mesa MAS: Sold {mesa_outcome['sales']} units (Demand: {mesa_outcome['demand']})"
#         )
        
#         print(f"Mesa outcome: {mesa_outcome}")
#         print(f"Processing Temporal team...")
        
#         # 3. Process Temporal MAS Team (Purple)
#         temporal_outcome = None
#         if self.temporal_session and TEMPORAL_AVAILABLE:
#             try:
#                 print("Attempting to process Temporal turn...")
#                 # Add timeout to prevent hanging
#                 import asyncio
                
#                 # Create isolated state for Temporal (map purple to blue for compatibility)
#                 temporal_state = {
#                     'turn': self.turn,
#                     'blue_team_inventory': self.game_state['purple_team_inventory'],
#                     'blue_team_price': self.game_state['purple_team_price'], 
#                     'blue_team_profit': self.game_state['purple_team_profit'],
#                     'green_team_price': self.game_state['green_team_price'],
#                     'green_team_inventory': self.game_state['green_team_inventory'],
#                     'event_log': []
#                 }
                
#                 # Create a new game state for Temporal session
#                 self.temporal_session.game_state = temporal_state.copy()
#                 self.temporal_session.turn_history = []  # Reset turn history
                
#                 # Run Temporal workflow with dummy human decisions - WITH TIMEOUT
#                 try:
#                     temporal_result = await asyncio.wait_for(
#                         self.temporal_session.process_turn(mesa_dummy_human),
#                         timeout=30.0  # Increased from 5 to 30 seconds
#                     )
#                     print("Temporal processing completed")
#                     print(f"Temporal result keys: {list(temporal_result.keys())}")
                    
#                     # Extract decisions from the Temporal result
#                     # The result should have updated blue_team values
#                     if 'game_state' in temporal_result:
#                         result_state = temporal_result['game_state']
#                     else:
#                         result_state = temporal_result
                    
#                     # Extract the MAS decisions from the result
#                     temporal_price = result_state.get('blue_team_price', 10)
#                     temporal_marketing = 500  # Default
#                     temporal_production = 50  # Default
                    
#                     # Try to extract from turn history if available
#                     if 'turn_history' in temporal_result and temporal_result['turn_history']:
#                         last_turn = temporal_result['turn_history'][-1]
#                         temporal_price = last_turn.get('price', temporal_price)
#                         temporal_marketing = last_turn.get('marketing_spend', temporal_marketing)
#                         temporal_production = last_turn.get('production_target', temporal_production)
#                         print(f"Extracted from turn history: Price={temporal_price}, Marketing={temporal_marketing}, Production={temporal_production}")
#                     else:
#                         # Parse from event log as backup
#                         for log in result_state.get('event_log', []):
#                             if 'MAS Decision:' in log:
#                                 import re
#                                 price_match = re.search(r'Price \$(\d+\.?\d*)', log)
#                                 prod_match = re.search(r'Produce (\d+)', log)
#                                 market_match = re.search(r'Marketing \$(\d+)', log)
                                
#                                 if price_match:
#                                     temporal_price = float(price_match.group(1))
#                                 if prod_match:
#                                     temporal_production = int(prod_match.group(1))
#                                 if market_match:
#                                     temporal_marketing = int(market_match.group(1))
#                                 print(f"Extracted from event log: Price={temporal_price}, Marketing={temporal_marketing}, Production={temporal_production}")
                    
#                     # Calculate Temporal outcome
#                     temporal_outcome = self.calculate_outcome(
#                         "purple_team",
#                         temporal_price,
#                         temporal_marketing,
#                         temporal_production
#                     )
#                     self.game_state['event_log'].append(
#                         f"Temporal MAS: Sold {temporal_outcome['sales']} units (Demand: {temporal_outcome['demand']})"
#                     )
#                     print(f"Temporal outcome: {temporal_outcome}")
                    
#                 except asyncio.TimeoutError:
#                     print("Temporal processing timed out after 30 seconds - using fallback")
#                     temporal_outcome = self._run_fallback_ai("purple_team")
#                     # Don't add redundant message to event log
#                 except Exception as e:
#                     print(f"Temporal processing error: {type(e).__name__}: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     temporal_outcome = self._run_fallback_ai("purple_team")
                    
#             except Exception as e:
#                 print(f"Temporal error: {e}")
#                 self.game_state['event_log'].append(f"Temporal MAS Error: {str(e)}")
#                 # Fallback to simple AI
#                 temporal_outcome = self._run_fallback_ai("purple_team")
#         else:
#             # Temporal not available, use fallback AI
#             print("Temporal not available - using fallback AI")
#             temporal_outcome = self._run_fallback_ai("purple_team")
            
#         # Ensure temporal_outcome was processed
#         if temporal_outcome is None:
#             print("WARNING: Temporal outcome is None, using default fallback")
#             temporal_outcome = self._run_fallback_ai("purple_team")
            
#         print(f"Final temporal outcome: {temporal_outcome}")
        
#         # Update projected demands with randomness
#         self.update_projected_demands()
        
#         # Ensure the event log always indicates 3-team mode
#         if not any("Three-way competition" in log for log in self.game_state['event_log']):
#             self.game_state['event_log'].insert(0, "Three-way competition: Human vs Mesa MAS vs Temporal MAS")
        
#         print(f"Turn {self.turn} complete. Returning game state...")
#         print(f"Game state keys: {list(self.game_state.keys())}")
        
#         # Verify all three teams have data
#         for team in ['green', 'blue', 'purple']:
#             profit_key = f"{team}_team_profit"
#             if profit_key not in self.game_state:
#                 print(f"WARNING: Missing {profit_key} in game state!")
        
#         return self.game_state
    
#     def _run_fallback_ai(self, team_prefix: str) -> Dict[str, Any]:
#         """Volatile fallback AI that simulates reactive human-like decisions"""
#         # Get current state
#         current_inventory = self.game_state[f'{team_prefix}_inventory']
#         current_price = self.game_state[f'{team_prefix}_price']
        
#         # Get team color from prefix
#         team_color = team_prefix.replace('_team', '')
        
#         # Calculate average sales with proper error handling
#         recent_sales = []
#         if team_color in self.sales_history and len(self.sales_history[team_color]) > 0:
#             recent_sales = self.sales_history[team_color][-5:]
#             avg_sales = np.mean(recent_sales)
#             sales_volatility = np.std(recent_sales) if len(recent_sales) > 1 else 10
#         else:
#             avg_sales = 50
#             sales_volatility = 15
        
#         # Add randomness to perception (humans don't have perfect information)
#         perceived_inventory = current_inventory * random.uniform(0.85, 1.15)
#         perceived_sales = avg_sales * random.uniform(0.9, 1.1)
        
#         # Calculate inventory pressure with overreaction
#         inventory_ratio = perceived_inventory / (perceived_sales + 1)
        
#         # VOLATILE PRICING STRATEGY
#         if inventory_ratio > 3.5:
#             # Panic mode - heavy discounting
#             price = random.uniform(8.0, 9.0)
#             price_reason = "Panic discounting!"
#         elif inventory_ratio > 2.5:
#             # Worried about excess inventory
#             price = random.uniform(9.0, 10.0)
#             price_reason = "Clearance pricing"
#         elif inventory_ratio < 0.8:
#             # Low inventory - premium pricing
#             price = random.uniform(12.0, 14.0)
#             price_reason = "Premium - low stock!"
#         elif inventory_ratio < 1.5:
#             # Comfortable - optimize margin
#             price = random.uniform(10.5, 11.5)
#             price_reason = "Optimizing margins"
#         else:
#             # Normal operations with noise
#             price = current_price + random.uniform(-1.0, 1.0)
#             price = max(8.5, min(13.5, price))
#             price_reason = "Market pricing"
        
#         # Add occasional price shocks (15% chance)
#         if random.random() < 0.15:
#             shock = random.choice([-1.5, -1.0, 1.0, 1.5])
#             price += shock
#             price_reason += f" + shock {shock:+.1f}"
        
#         # VOLATILE MARKETING STRATEGY
#         marketing_base = 500
        
#         # Inventory-based marketing with overreaction
#         if inventory_ratio > 4:
#             marketing = random.randint(1500, 2000)
#             marketing_reason = "Maximum push!"
#         elif inventory_ratio > 3:
#             marketing = random.randint(1000, 1500)
#             marketing_reason = "Heavy promotion"
#         elif inventory_ratio > 2:
#             marketing = random.randint(700, 1200)
#             marketing_reason = "Increased marketing"
#         elif inventory_ratio < 1:
#             marketing = random.randint(100, 400)
#             marketing_reason = "Low stock, save $"
#         else:
#             marketing = random.randint(400, 800)
#             marketing_reason = "Normal marketing"
        
#         # Add volatility based on recent performance
#         if len(recent_sales) >= 2:
#             sales_trend = recent_sales[-1] - recent_sales[-2]
#             if sales_trend < -10:
#                 # Sales dropping - panic marketing
#                 marketing = int(marketing * random.uniform(1.3, 1.7))
#                 marketing_reason += " + panic boost!"
#             elif sales_trend > 10:
#                 # Sales rising - ride the wave
#                 marketing = int(marketing * random.uniform(0.8, 1.2))
#                 marketing_reason += " + momentum"
        
#         # Random marketing campaigns (20% chance)
#         if random.random() < 0.2:
#             campaign_multiplier = random.choice([0.5, 0.75, 1.5, 2.0])
#             marketing = int(marketing * campaign_multiplier)
#             marketing_reason = f"Special campaign x{campaign_multiplier}"
        
#         # Final bounds
#         marketing = max(0, min(2000, marketing))
#         marketing = round(marketing / 50) * 50  # Round to nearest 50
        
#         # VOLATILE PRODUCTION STRATEGY
#         # Base on perceived demand with overreaction
#         production_base = perceived_sales
        
#         # Inventory-based adjustments with overreaction
#         if inventory_ratio > 3:
#             production = random.randint(0, 30)
#             production_reason = "Halt production!"
#         elif inventory_ratio > 2:
#             production = int(production_base * random.uniform(0.3, 0.6))
#             production_reason = "Slow production"
#         elif inventory_ratio < 0.5:
#             production = int(production_base * random.uniform(1.5, 2.5))
#             production_reason = "Emergency production!"
#         elif inventory_ratio < 1:
#             production = int(production_base * random.uniform(1.2, 1.5))
#             production_reason = "Boost production"
#         else:
#             production = int(production_base * random.uniform(0.9, 1.3))
#             production_reason = "Normal production"
        
#         # Add production volatility based on market sentiment
#         if self.demand_simulator.market_sentiment > 0.3:
#             production = int(production * random.uniform(1.1, 1.3))
#             production_reason += " + bullish"
#         elif self.demand_simulator.market_sentiment < -0.3:
#             production = int(production * random.uniform(0.7, 0.9))
#             production_reason += " + bearish"
        
#         # Random production decisions (15% chance)
#         if random.random() < 0.15:
#             production_shock = random.choice([0.5, 0.7, 1.3, 1.5])
#             production = int(production * production_shock)
#             production_reason += f" x{production_shock}"
        
#         # Ensure bounds
#         production = max(0, min(200, production))
#         price = max(8.0, min(14.0, price))
        
#         # Log the volatile decision with actual values
#         self.game_state['event_log'].append(
#             f"Temporal AI: Price ${price:.1f} ({price_reason}), Marketing ${marketing} ({marketing_reason}), Prod {production} ({production_reason})"
#         )
        
#         print(f"Temporal Decision: Inventory={current_inventory}, Ratio={inventory_ratio:.2f}, Price=${price:.2f}, Marketing=${marketing}, Production={production}")
        
#         outcome = self.calculate_outcome(team_prefix, price, marketing, production)
#         return outcome
    
#     def update_projected_demands(self):
#         """Update projected demands with randomness"""
#         # Add some noise to projections
#         for team in ['green_team', 'blue_team', 'purple_team']:
#             base_projection = 50
#             if team.replace('_team', '') in self.sales_history:
#                 recent_sales = self.sales_history[team.replace('_team', '')][-3:]
#                 if recent_sales:
#                     base_projection = int(np.mean(recent_sales))
            
#             # Add random variation to projection
#             noise = random.randint(-10, 10)
#             self.game_state[f'{team}_projected_demand'] = max(20, base_projection + noise)
    
#     async def run_compete_mode(self, websocket: WebSocket, turns: int):
#         """Run AI competition mode for specified turns"""
#         self.compete_mode = True
#         self.compete_turns_total = turns
#         self.compete_turns_done = 0
        
#         print(f"Starting AI competition mode for {turns} turns")
        
#         for i in range(turns):
#             if not self.compete_mode:  # Check if stopped
#                 break
                
#             # Run AI vs AI turn
#             # Human team uses simple AI in compete mode
#             human_ai_decisions = self._get_simple_ai_decisions("green_team")
            
#             # Process turn with AI decisions
#             updated_state = await self.process_turn(human_ai_decisions)
            
#             # Update progress
#             self.compete_turns_done = i + 1
#             progress = (self.compete_turns_done / self.compete_turns_total) * 100
#             updated_state['compete_progress'] = progress
#             updated_state['compete_complete'] = (i == turns - 1)
            
#             # Send update
#             await websocket.send_text(json.dumps(updated_state))
            
#             # Small delay to make it viewable
#             await asyncio.sleep(0.5)
        
#         self.compete_mode = False
#         print(f"Competition complete after {self.compete_turns_done} turns")
        
#         # Send final summary
#         final_state = self.game_state.copy()
#         final_state['event_log'] = [
#             "🏆 AI COMPETITION COMPLETE! 🏆",
#             f"Total Turns: {self.compete_turns_done}",
#             f"Final Profits:",
#             f"  Green (Simple AI): ${self.game_state['green_team_profit']:,.0f}",
#             f"  Blue (Mesa MAS): ${self.game_state['blue_team_profit']:,.0f}",
#             f"  Purple (Temporal/Volatile AI): ${self.game_state['purple_team_profit']:,.0f}",
#             "",
#             f"🥇 Winner: {self._get_winner()}!"
#         ]
#         final_state['compete_progress'] = 100
#         final_state['compete_complete'] = True
#         await websocket.send_text(json.dumps(final_state))
    
#     def _get_simple_ai_decisions(self, team_prefix: str) -> Dict[str, Any]:
#         """Simple AI for human team during compete mode"""
#         current_inventory = self.game_state[f'{team_prefix}_inventory']
        
#         # Simple inventory-based decisions
#         if current_inventory > 150:
#             price = 9.0
#             marketing = 1200
#             production = 30
#         elif current_inventory > 100:
#             price = 9.5
#             marketing = 800
#             production = 40
#         elif current_inventory < 50:
#             price = 11.0
#             marketing = 400
#             production = 80
#         else:
#             price = 10.0
#             marketing = 600
#             production = 60
        
#         # Add some randomness
#         price += random.uniform(-0.5, 0.5)
#         marketing = marketing + random.randint(-200, 200)
#         production = production + random.randint(-10, 10)
        
#         # Bounds
#         price = max(8, min(14, price))
#         marketing = max(0, min(2000, marketing))
#         production = max(0, min(200, production))
        
#         return {
#             'price': price,
#             'marketingSpend': marketing,
#             'productionTarget': production
#         }
    
#     def _get_winner(self) -> str:
#         """Determine competition winner"""
#         profits = {
#             'Green (Simple AI)': self.game_state['green_team_profit'],
#             'Blue (Mesa MAS)': self.game_state['blue_team_profit'],
#             'Purple (Temporal/Volatile AI)': self.game_state['purple_team_profit']
#         }
#         winner = max(profits.items(), key=lambda x: x[1])
#         return winner[0]

# # Store game sessions
# game_sessions = {}

# async def run_temporal_worker():
#     """Run Temporal worker for processing workflows"""
#     if not TEMPORAL_AVAILABLE:
#         print("Temporal not available, worker not started")
#         return
        
#     try:
#         print("Connecting Temporal worker to localhost:7233...")
#         client = await Client.connect("localhost:7233")
        
#         worker = Worker(
#             client,
#             task_queue="brewmasters-queue",
#             workflows=[MASDecisionWorkflow, GameTurnWorkflow],
#             activities=[
#                 create_shared_knowledge,
#                 pricing_agent_activity,
#                 marketing_agent_activity,
#                 production_agent_activity,
#                 ceo_decision_activity,
#                 calculate_game_outcome
#             ]
#         )
        
#         print("Temporal worker started successfully!")
#         await worker.run()
#     except Exception as e:
#         print(f"Temporal worker error: {e}")
#         print("Continuing without Temporal worker - will use fallback AI")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("Client connected for three-way comparison")
    
#     connection_id = id(websocket)
#     session = UnifiedGameSession()
#     game_sessions[connection_id] = session
    
#     # Try to initialize Temporal
#     temporal_connected = await session.initialize_temporal()
#     if temporal_connected:
#         print("Temporal connected successfully")
#     else:
#         print("Running without Temporal (will use fallback AI)")
    
#     try:
#         # Send initial game state with clear 3-team indication
#         initial_state = session.game_state.copy()
#         # Ensure the initial message clearly indicates 3-team mode
#         if "Three-way competition" not in initial_state['event_log'][0]:
#             initial_state['event_log'].insert(0, "Three-way competition between Human, Mesa MAS, and Temporal MAS")
        
#         await websocket.send_text(json.dumps(initial_state))
        
#         while True:
#             message = await websocket.receive_text()
            
#             # Check for restart
#             if message == '{"restart": true}':
#                 # Create new session
#                 session = UnifiedGameSession()
#                 game_sessions[connection_id] = session
#                 await session.initialize_temporal()
#                 await websocket.send_text(json.dumps(session.game_state))
#                 continue
            
#             # Check for compete mode
#             data = json.loads(message)
#             if data.get('compete'):
#                 turns = data.get('turns', 10)
#                 asyncio.create_task(session.run_compete_mode(websocket, turns))
#                 continue
            
#             if data.get('stopCompete'):
#                 session.compete_mode = False
#                 continue
            
#             # Process normal turn (only if not in compete mode)
#             if not session.compete_mode:
#                 human_decisions = data
#                 print(f"Received human decisions: {human_decisions}")
                
#                 try:
#                     # Process all three teams
#                     updated_state = await session.process_turn(human_decisions)
                    
#                     # Send updated state
#                     response_json = json.dumps(updated_state)
#                     print(f"Sending response with {len(response_json)} chars...")
#                     await websocket.send_text(response_json)
#                     print("Response sent successfully")
#                 except Exception as e:
#                     print(f"Error processing turn: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     error_response = {
#                         "error": str(e),
#                         "turn": session.game_state['turn'],
#                         "event_log": [f"Error: {str(e)}"]
#                     }
#                     await websocket.send_text(json.dumps(error_response))
    
#     except WebSocketDisconnect:
#         print("Client disconnected")
#         if connection_id in game_sessions:
#             del game_sessions[connection_id]
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
#         if connection_id in game_sessions:
#             del game_sessions[connection_id]

# async def main():
#     """Run the unified server"""
#     # Start Temporal worker if available
#     worker_task = None
#     if TEMPORAL_AVAILABLE:
#         print("Starting Temporal worker...")
#         worker_task = asyncio.create_task(run_temporal_worker())
#         # Give worker time to start
#         await asyncio.sleep(1)
    
#     # Run the web server
#     config = uvicorn.Config(app, host="0.0.0.0", port=8000)
#     server = uvicorn.Server(config)
    
#     try:
#         await server.serve()
#     finally:
#         if worker_task:
#             worker_task.cancel()
#             try:
#                 await worker_task
#             except asyncio.CancelledError:
#                 pass

# if __name__ == "__main__":
#     print("Starting Unified BrewMasters Server...")
#     print("This server runs Human vs Mesa MAS vs Temporal MAS")
    
#     # Check if we have all Temporal imports
#     if TEMPORAL_AVAILABLE:
#         print("Temporal libraries found")
#         try:
#             from temporalio.worker import Worker
#             print("✓ Worker imported successfully")
#         except ImportError as e:
#             print(f"✗ Failed to import Worker: {e}")
            
#         try:
#             # Test if we can import the workflows
#             print(f"✓ Workflows available: MASDecisionWorkflow={MASDecisionWorkflow}, GameTurnWorkflow={GameTurnWorkflow}")
#         except Exception as e:
#             print(f"✗ Workflows not available: {e}")
#     else:
#         print("Temporal not available - will use fallback AI")
    
#     asyncio.run(main())

# debug_server.py - Minimal working version for debugging
import asyncio
import json
import random
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleGameSession:
    def __init__(self):
        self.turn = 0
        self.competition_mode = False
        
        # Simple game state
        self.game_state = {
            'turn': 0,
            'competition_mode': False,
            # Human Team (Green)
            'green_team_profit': 100000,
            'green_team_inventory': 100,
            'green_team_price': 10,
            'green_team_production': 50,
            'green_team_marketing': 500,
            'green_team_projected_demand': 50,
            'green_team_profit_this_turn': 0,
            # Mesa Team (Blue)
            'blue_team_profit': 100000,
            'blue_team_inventory': 100,
            'blue_team_price': 10,
            'blue_team_production': 50,
            'blue_team_marketing': 500,
            'blue_team_projected_demand': 50,
            'blue_team_profit_this_turn': 0,
            # Temporal Team (Purple)
            'purple_team_profit': 100000,
            'purple_team_inventory': 100,
            'purple_team_price': 10,
            'purple_team_production': 50,
            'purple_team_marketing': 500,
            'purple_team_projected_demand': 50,
            'purple_team_profit_this_turn': 0,
            # Event log
            'event_log': ["🎮 Debug Server Started!", "Three-way competition: Human vs Mesa vs Temporal"],
            # History
            'sales_history': {'green': [], 'blue': [], 'purple': []},
            'production_history': {'green': [], 'blue': [], 'purple': []},
            'price_history': {'green': [], 'blue': [], 'purple': []}
        }
    
    def make_ai_decision(self, team_name: str, current_state: dict) -> dict:
        """Advanced AI decision making with varying strategies"""
        inventory = current_state['inventory']
        profit = current_state.get('profit', 100000)
        turn = current_state.get('turn', 1)
        competitor_prices = current_state.get('competitor_prices', [10])
        
        # Calculate average competitor price for competitive analysis
        avg_competitor_price = sum(competitor_prices) / len(competitor_prices) if competitor_prices else 10
        
        # Different AI personalities
        if team_name == "Mesa":
            # Mesa AI: Analytical and reactive to inventory levels
            if inventory > 150:
                # High inventory - aggressive clearance
                price = round(max(8.0, avg_competitor_price - random.uniform(1.0, 2.0)), 2)
                marketing = random.randint(1500, 2000)  # Heavy marketing
                production = random.randint(10, 30)     # Reduce production
                strategy = "Clearance mode - high inventory"
                
            elif inventory > 100:
                # Medium-high inventory - competitive pricing
                price = round(avg_competitor_price - random.uniform(0.2, 0.8), 2)
                marketing = random.randint(800, 1200)
                production = random.randint(40, 60)
                strategy = "Competitive pricing"
                
            elif inventory < 50:
                # Low inventory - premium pricing
                price = round(min(14.0, avg_competitor_price + random.uniform(0.5, 1.5)), 2)
                marketing = random.randint(200, 500)    # Low marketing to preserve stock
                production = random.randint(80, 120)    # Increase production
                strategy = "Premium pricing - low stock"
                
            else:
                # Normal inventory - balanced approach
                price = round(avg_competitor_price + random.uniform(-0.3, 0.3), 2)
                marketing = random.randint(600, 1000)
                production = random.randint(45, 75)
                strategy = "Balanced strategy"
                
        elif team_name == "Temporal":
            # Temporal AI: More aggressive and trend-following
            inventory_ratio = inventory / 100  # Relative to starting inventory
            
            if inventory_ratio > 2.0:
                # Panic mode - slash prices and boost marketing
                price = round(random.uniform(8.0, 9.5), 2)
                marketing = random.randint(1800, 2000)  # Maximum marketing
                production = random.randint(5, 25)      # Minimal production
                strategy = "Panic liquidation"
                
            elif inventory_ratio > 1.5:
                # Worried - moderate discounting
                price = round(avg_competitor_price - random.uniform(0.5, 1.2), 2)
                marketing = random.randint(1200, 1600)
                production = random.randint(30, 50)
                strategy = "Moderate discounting"
                
            elif inventory_ratio < 0.3:
                # Very low stock - exploit scarcity
                price = round(min(15.0, avg_competitor_price + random.uniform(1.0, 2.5)), 2)
                marketing = random.randint(100, 300)    # Minimal marketing
                production = random.randint(100, 150)   # Maximum production
                strategy = "Scarcity premium"
                
            elif inventory_ratio < 0.7:
                # Low stock - premium approach
                price = round(avg_competitor_price + random.uniform(0.3, 1.0), 2)
                marketing = random.randint(400, 700)
                production = random.randint(70, 100)
                strategy = "Premium approach"
                
            else:
                # Normal operations with trend following
                trend_adjustment = random.uniform(-0.5, 0.5)
                price = round(avg_competitor_price + trend_adjustment, 2)
                marketing = random.randint(500, 1000)
                production = random.randint(40, 80)
                strategy = "Trend following"
        
        else:
            # Fallback for other teams
            price = round(random.uniform(9.0, 12.0), 2)
            marketing = random.randint(400, 800)
            production = random.randint(40, 70)
            strategy = "Basic strategy"
        
        # Add some random events (10% chance for each AI)
        if random.random() < 0.1:
            event_type = random.choice(['aggressive', 'conservative', 'experimental'])
            if event_type == 'aggressive':
                price *= 0.9  # Lower price
                marketing = int(marketing * 1.5)  # Boost marketing
                strategy += " + aggressive campaign"
            elif event_type == 'conservative':
                price *= 1.1  # Higher price
                marketing = int(marketing * 0.7)  # Reduce marketing
                strategy += " + conservative approach"
            elif event_type == 'experimental':
                production = int(production * random.uniform(0.5, 1.8))
                strategy += " + experimental production"
        
        # Ensure bounds
        price = round(max(8.0, min(15.0, price)), 2)
        marketing = max(0, min(2000, marketing))
        production = max(10, min(150, production))
        
        return {
            'price': price,
            'production': production,
            'marketing': marketing,
            'reasoning': f"{team_name}: {strategy} → ${price:.2f}, {production}u, ${marketing}m",
            'strategy': strategy
        }
    
    async def process_turn(self, human_decisions: dict) -> dict:
        """Process one turn"""
        print(f"🎯 Processing turn {self.turn + 1}")
        print(f"📥 Human decisions: {human_decisions}")
        
        self.turn += 1
        self.game_state['turn'] = self.turn
        
        # Process Human (Green) team
        human_price = float(human_decisions.get('price', 10))
        human_production = int(human_decisions.get('productionTarget', 50))
        human_marketing = int(human_decisions.get('marketingSpend', 500))
        
        self.game_state['green_team_price'] = human_price
        self.game_state['green_team_production'] = human_production
        self.game_state['green_team_marketing'] = human_marketing
        
        # Process Mesa (Blue) team
        mesa_state = {
            'inventory': self.game_state['blue_team_inventory'],
            'profit': self.game_state['blue_team_profit'],
            'turn': self.turn,
            'competitor_prices': [human_price, self.game_state.get('purple_team_price', 10)]
        }
        mesa_decisions = self.make_ai_decision("Mesa", mesa_state)
        
        self.game_state['blue_team_price'] = mesa_decisions['price']
        self.game_state['blue_team_production'] = mesa_decisions['production']
        self.game_state['blue_team_marketing'] = mesa_decisions['marketing']
        
        # Process Temporal (Purple) team
        temporal_state = {
            'inventory': self.game_state['purple_team_inventory'],
            'profit': self.game_state['purple_team_profit'],
            'turn': self.turn,
            'competitor_prices': [human_price, mesa_decisions['price']]
        }
        temporal_decisions = self.make_ai_decision("Temporal", temporal_state)
        
        self.game_state['purple_team_price'] = temporal_decisions['price']
        self.game_state['purple_team_production'] = temporal_decisions['production']
        self.game_state['purple_team_marketing'] = temporal_decisions['marketing']
        
        # Simple sales calculation for all teams
        teams = [
            ('green_team', 'green'),
            ('blue_team', 'blue'), 
            ('purple_team', 'purple')
        ]
        
        for team_prefix, color in teams:
            # Simple demand calculation
            price = self.game_state[f'{team_prefix}_price']
            marketing = self.game_state[f'{team_prefix}_marketing']
            base_demand = 50
            
            # Price effect (lower price = higher demand)
            price_effect = (12 - price) * 5
            # Marketing effect
            marketing_effect = marketing * 0.02
            # Random variation
            random_effect = random.randint(-15, 15)
            
            demand = max(10, min(120, int(base_demand + price_effect + marketing_effect + random_effect)))
            
            # Sales = min(demand, inventory)
            inventory = self.game_state[f'{team_prefix}_inventory']
            sales = min(demand, inventory)
            
            # Update inventory
            production = self.game_state[f'{team_prefix}_production']
            new_inventory = inventory - sales + production
            self.game_state[f'{team_prefix}_inventory'] = max(0, new_inventory)
            
            # Calculate profit
            revenue = sales * price
            costs = production * 3.5 + marketing
            profit_change = revenue - costs
            
            self.game_state[f'{team_prefix}_profit'] += profit_change
            self.game_state[f'{team_prefix}_profit_this_turn'] = profit_change
            self.game_state[f'{team_prefix}_projected_demand'] = demand
            
            # Update history
            self.game_state['sales_history'][color].append(sales)
            self.game_state['production_history'][color].append(production)
            self.game_state['price_history'][color].append(price)
            
            # Keep history limited
            for hist_key in ['sales_history', 'production_history', 'price_history']:
                if len(self.game_state[hist_key][color]) > 20:
                    self.game_state[hist_key][color] = self.game_state[hist_key][color][-20:]
        
        # Update event log
        self.game_state['event_log'] = [
            f"📊 Turn {self.turn} Complete",
            f"👤 Human: Price ${human_price:.2f}, Prod {human_production}, Mkt ${human_marketing}",
            f"🤖 Mesa: {mesa_decisions['reasoning']}",
            f"⚡ Temporal: {temporal_decisions['reasoning']}",
            f"Three-way competition active"
        ]
        
        print(f"✅ Turn {self.turn} processed successfully")
        print(f"📊 Final game state production values:")
        print(f"  Green: {self.game_state['green_team_production']}")
        print(f"  Blue: {self.game_state['blue_team_production']}")
        print(f"  Purple: {self.game_state['purple_team_production']}")
        print(f"📊 Final game state marketing values:")
        print(f"  Green: {self.game_state['green_team_marketing']}")
        print(f"  Blue: {self.game_state['blue_team_marketing']}")
        print(f"  Purple: {self.game_state['purple_team_marketing']}")
        
        return self.game_state
    
    async def run_competition(self, websocket: WebSocket, turns: int):
        """Run Mesa vs Temporal competition"""
        self.competition_mode = True
        print(f"🏆 Starting competition: {turns} turns")
        
        for i in range(turns):
            if not self.competition_mode:
                break
            
            self.turn += 1
            self.game_state['turn'] = self.turn
            
            # Only Mesa and Temporal compete (Human frozen)
            mesa_state = {
                'inventory': self.game_state['blue_team_inventory'],
                'profit': self.game_state['blue_team_profit'],
                'turn': self.turn,
                'competitor_prices': [self.game_state.get('purple_team_price', 10)]
            }
            mesa_decisions = self.make_ai_decision("Mesa", mesa_state)
            
            temporal_state = {
                'inventory': self.game_state['purple_team_inventory'],
                'profit': self.game_state['purple_team_profit'],
                'turn': self.turn,
                'competitor_prices': [mesa_decisions['price']]
            }
            temporal_decisions = self.make_ai_decision("Temporal", temporal_state)
            
            # Update only Mesa and Temporal
            self.game_state['blue_team_price'] = mesa_decisions['price']
            self.game_state['blue_team_production'] = mesa_decisions['production']
            self.game_state['blue_team_marketing'] = mesa_decisions['marketing']
            
            self.game_state['purple_team_price'] = temporal_decisions['price']
            self.game_state['purple_team_production'] = temporal_decisions['production']
            self.game_state['purple_team_marketing'] = temporal_decisions['marketing']
            
            # Calculate results for Mesa and Temporal only
            for team_prefix, color in [('blue_team', 'blue'), ('purple_team', 'purple')]:
                inventory = self.game_state[f'{team_prefix}_inventory']
                production = self.game_state[f'{team_prefix}_production']
                price = self.game_state[f'{team_prefix}_price']
                marketing = self.game_state[f'{team_prefix}_marketing']
                
                demand = random.randint(30, 100)
                sales = min(demand, inventory)
                
                self.game_state[f'{team_prefix}_inventory'] = max(0, inventory - sales + production)
                
                revenue = sales * price
                costs = production * 3.5 + marketing
                profit_change = revenue - costs
                
                self.game_state[f'{team_prefix}_profit'] += profit_change
            
            # Update event log
            self.game_state['event_log'] = [
                f"🏆 COMPETITION TURN {self.turn}",
                f"🤖 Mesa: {mesa_decisions['reasoning']}",
                f"⚡ Temporal: {temporal_decisions['reasoning']}",
                f"👤 Human: FROZEN (not competing)"
            ]
            
            # Send progress
            progress = ((i + 1) / turns) * 100
            response = self.game_state.copy()
            response['compete_progress'] = progress
            response['compete_complete'] = (i + 1 == turns)
            
            await websocket.send_text(json.dumps(response))
            await asyncio.sleep(1.0)
        
        self.competition_mode = False
        
        # Final results
        winner = "Mesa" if self.game_state['blue_team_profit'] > self.game_state['purple_team_profit'] else "Temporal"
        if self.game_state['blue_team_profit'] == self.game_state['purple_team_profit']:
            winner = "TIE"
        
        final_response = self.game_state.copy()
        final_response['event_log'] = [
            "🏆 COMPETITION COMPLETE!",
            f"Winner: {winner}",
            f"Mesa: ${self.game_state['blue_team_profit']:,.0f}",
            f"Temporal: ${self.game_state['purple_team_profit']:,.0f}"
        ]
        final_response['compete_progress'] = 100
        final_response['compete_complete'] = True
        
        await websocket.send_text(json.dumps(final_response))

# Global session
game_session = SimpleGameSession()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")
    
    try:
        # Send initial state
        await websocket.send_text(json.dumps(game_session.game_state))
        print("📤 Initial state sent")
        
        async for message in websocket.iter_text():
            print(f"📨 Received message: {message}")
            
            try:
                data = json.loads(message)
                print(f"📋 Parsed data: {data}")
                
                # Handle restart
                if data.get('restart'):
                    print("🔄 Restarting game")
                    game_session.__init__()
                    await websocket.send_text(json.dumps(game_session.game_state))
                    continue
                
                # Handle competition mode
                if data.get('compete'):
                    turns = data.get('turns', 10)
                    print(f"🏆 Starting competition: {turns} turns")
                    asyncio.create_task(game_session.run_competition(websocket, turns))
                    continue
                
                if data.get('stopCompete'):
                    print("⏹️ Stopping competition")
                    game_session.competition_mode = False
                    continue
                
                # Handle normal turn (only if not in competition mode)
                if not game_session.competition_mode:
                    print("🎮 Processing normal turn")
                    updated_state = await game_session.process_turn(data)
                    await websocket.send_text(json.dumps(updated_state))
                    print("📤 Response sent")
                else:
                    print("⚠️ Turn ignored - competition mode active")
                
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                await websocket.send_text(json.dumps({"error": f"Invalid JSON: {e}"}))
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_text(json.dumps({"error": str(e)}))
    
    except WebSocketDisconnect:
        print("🔌 Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")

@app.get("/")
async def read_root():
    return {"message": "🎮 BrewMasters Debug Server Running", "status": "OK"}

@app.get("/status")
async def get_status():
    return {
        "server": "debug",
        "turn": game_session.turn,
        "competition_mode": game_session.competition_mode,
        "teams": 3
    }

if __name__ == "__main__":
    print("🚀 Starting BrewMasters Debug Server...")
    print("🔍 This server has detailed logging to help debug issues")
    print("🌐 Server will run on http://localhost:8000")
    print("🔌 WebSocket endpoint: ws://localhost:8000/ws")
    uvicorn.run(app, host="0.0.0.0", port=8000)