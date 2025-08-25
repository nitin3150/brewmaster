# import asyncio
# import json
# import os
# import random
# from typing import Dict, Any, Optional
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# import uvicorn
# from dotenv import load_dotenv

# load_dotenv()

# try:
#     import google.generativeai as genai
#     GEMINI_AVAILABLE = True
#     print("✅ Gemini AI available")
# except ImportError:
#     GEMINI_AVAILABLE = False
#     print("⚠️ Gemini AI not available - using fallbacks")

# class SimplifiedGeminiHelper:
#     """Simplified Gemini helper that just works"""
    
#     def __init__(self, api_key: Optional[str] = None):
#         self.enabled = False
#         self.model = None
        
#         if api_key and GEMINI_AVAILABLE:
#             try:
#                 genai.configure(api_key=api_key)
#                 self.model = genai.GenerativeModel('gemini-1.5-flash')
#                 self.enabled = True
#                 print("✅ Gemini AI initialized successfully")
#             except Exception as e:
#                 print(f"❌ Gemini AI failed: {e}")
#                 self.enabled = False
#         else:
#             print("⚠️ Gemini AI disabled (no API key or SDK unavailable)")
    
#     async def make_decision(self, framework: str, context: Dict[str, Any]) -> Dict[str, Any]:
#         """Make AI-enhanced or fallback decision"""
#         if self.enabled:
#             try:
#                 return await self._ai_decision(framework, context)
#             except Exception as e:
#                 print(f"AI decision failed for {framework}: {e}")
#                 return self._fallback_decision(framework, context)
#         else:
#             return self._fallback_decision(framework, context)
    
#     async def _ai_decision(self, framework: str, context: Dict[str, Any]) -> Dict[str, Any]:
#         """Get AI decision from Gemini"""
#         prompt = f"""
# You are a {framework.upper()} AI system making brewery business decisions.

# Current Situation:
# - Inventory: {context.get('inventory', 100)} units  
# - Current Price: ${context.get('current_price', 10.0):.2f}
# - Competitor Prices: {context.get('competitor_prices', [10.0])}
# - Turn: {context.get('turn', 1)}

# Make strategic decisions for price (8.0-15.0), production (10-150), and marketing (0-2000).

# Framework approach:
# - Mesa: Multi-agent coordination with specialized agents
# - Temporal: Workflow-based sequential decision making  
# - Google ADK: ML-driven optimization with advanced analytics

# Respond only in JSON:
# {{"price": 10.50, "production": 60, "marketing": 800, "reasoning": "Strategy explanation"}}
# """
        
#         response = await asyncio.get_event_loop().run_in_executor(
#             None, lambda: self.model.generate_content(prompt)
#         )
        
#         # Parse response
#         import re
#         json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
#         if json_match:
#             data = json.loads(json_match.group())
            
#             # Validate and constrain
#             price = max(8.0, min(15.0, float(data.get('price', 10))))
#             production = max(10, min(150, int(data.get('production', 50))))
#             marketing = max(0, min(2000, int(data.get('marketing', 500))))
            
#             return {
#                 'price': round(price, 2),
#                 'production': production,
#                 'marketing': marketing,
#                 'reasoning': f"{framework.upper()} AI: {data.get('reasoning', 'Strategic decision')}",
#                 'ai_enhanced': True,
#                 'framework': framework
#             }
#         else:
#             raise ValueError("No valid JSON in response")
    
#     def _fallback_decision(self, framework: str, context: Dict[str, Any]) -> Dict[str, Any]:
#         """Smart fallback decisions based on framework characteristics"""
#         inventory = context.get('inventory', 100)
#         competitors = context.get('competitor_prices', [10.0])
#         avg_competitor = sum(competitors) / len(competitors)
        
#         # Framework-specific strategic approaches
#         if framework == "mesa":
#             # Mesa: Agent-based coordination with specialization
#             if inventory > 150:
#                 # High inventory - pricing agent suggests discount, marketing agent boosts spend
#                 price = max(8.0, avg_competitor - random.uniform(0.8, 1.2))
#                 production = random.randint(20, 40)
#                 marketing = random.randint(1200, 1800)
#                 strategy = "Mesa agents coordinate: discount pricing + heavy marketing"
#             elif inventory < 50:
#                 # Low inventory - pricing agent suggests premium, production agent increases
#                 price = min(15.0, avg_competitor + random.uniform(0.5, 1.0))
#                 production = random.randint(100, 140)
#                 marketing = random.randint(200, 500)
#                 strategy = "Mesa agents coordinate: premium pricing + boost production"
#             else:
#                 # Balanced - agents find middle ground
#                 price = avg_competitor + random.uniform(-0.3, 0.3)
#                 production = random.randint(50, 80)
#                 marketing = random.randint(600, 1000)
#                 strategy = "Mesa agents coordinate: balanced approach"
                
#         elif framework == "temporal":
#             # Temporal: Sequential workflow activities
#             if inventory > 150:
#                 # Workflow: Analysis → Clearance Strategy → Execution
#                 price = max(8.0, avg_competitor - random.uniform(1.0, 1.5))
#                 production = random.randint(15, 35)
#                 marketing = random.randint(1400, 2000)
#                 strategy = "Temporal workflow: Analysis→Clearance→Execute"
#             elif inventory < 50:
#                 # Workflow: Analysis → Growth Strategy → Execution  
#                 price = min(15.0, avg_competitor + random.uniform(0.8, 1.3))
#                 production = random.randint(110, 150)
#                 marketing = random.randint(150, 400)
#                 strategy = "Temporal workflow: Analysis→Growth→Execute"
#             else:
#                 # Workflow: Analysis → Optimization → Execution
#                 price = avg_competitor + random.uniform(-0.4, 0.4)
#                 production = random.randint(60, 90)
#                 marketing = random.randint(700, 1100)
#                 strategy = "Temporal workflow: Analysis→Optimize→Execute"
                
#         elif framework == "google_adk":
#             # Google ADK: ML/Cloud optimization
#             if inventory > 150:
#                 # ML models predict optimal clearance strategy
#                 price = max(8.0, avg_competitor - random.uniform(1.2, 1.8))
#                 production = random.randint(10, 30)
#                 marketing = random.randint(1500, 2000)
#                 strategy = "ML models: Predictive clearance optimization"
#             elif inventory < 50:
#                 # ML models predict growth opportunity
#                 price = min(15.0, avg_competitor + random.uniform(1.0, 1.5))
#                 production = random.randint(120, 150)
#                 marketing = random.randint(100, 350)
#                 strategy = "ML models: Predictive growth optimization"
#             else:
#                 # ML ensemble optimization
#                 price_variance = random.uniform(0.7, 1.3)
#                 price = avg_competitor * price_variance
#                 production = random.randint(70, 100)
#                 marketing = random.randint(800, 1200)
#                 strategy = "ML ensemble: Multi-objective optimization"
#         else:
#             # Generic fallback
#             price, production, marketing = 10.0, 50, 500
#             strategy = "Basic fallback strategy"
        
#         return {
#             'price': round(max(8.0, min(15.0, price)), 2),
#             'production': max(10, min(150, production)),
#             'marketing': max(0, min(2000, marketing)),
#             'reasoning': f"{framework.upper()}: {strategy} (fallback)",
#             'ai_enhanced': False,
#             'framework': framework
#         }

# class BrewMastersCoordinator:
#     """Main coordinator for all three frameworks"""
    
#     def __init__(self, gemini_api_key: Optional[str] = None):
#         self.ai_helper = SimplifiedGeminiHelper(gemini_api_key)
#         self.turn = 0
#         self.competition_mode = False
        
#         # Initialize all teams
#         self.teams = {
#             'green': {
#                 'name': 'Human Player',
#                 'profit': 100000,
#                 'inventory': 100,
#                 'price': 10.0,
#                 'production': 50,
#                 'marketing': 500,
#                 'profit_this_turn': 0
#             },
#             'blue': {
#                 'name': 'Mesa Multi-Agent System',
#                 'profit': 100000,
#                 'inventory': 100,
#                 'price': 10.0,
#                 'production': 50,
#                 'marketing': 500,
#                 'profit_this_turn': 0
#             },
#             'purple': {
#                 'name': 'Temporal Workflow System',
#                 'profit': 100000,
#                 'inventory': 100,
#                 'price': 10.0,
#                 'production': 50,
#                 'marketing': 500,
#                 'profit_this_turn': 0
#             },
#             'orange': {
#                 'name': 'Google ADK ML System',
#                 'profit': 100000,
#                 'inventory': 100,
#                 'price': 10.0,
#                 'production': 50,
#                 'marketing': 500,
#                 'profit_this_turn': 0
#             }
#         }
        
#         self.event_log = [
#             "🎮 Multi-Framework BrewMasters Started!",
#             f"🤖 AI Status: {'Enhanced with Gemini' if self.ai_helper.enabled else 'Fallback Mode'}",
#             "🔧 Frameworks Active: Mesa MAS + Temporal Workflows + Google ADK ML",
#             "⚔️ Ready for epic framework battles!"
#         ]
        
#         print("✅ BrewMasters Coordinator initialized successfully")
    
#     async def process_turn(self, human_decisions: Dict[str, Any]) -> Dict[str, Any]:
#         """Process complete turn for all frameworks"""
#         self.turn += 1
        
#         # Process human team
#         human_team = self.teams['green']
#         human_team.update({
#             'price': float(human_decisions.get('price', 10)),
#             'production': int(human_decisions.get('productionTarget', 50)),
#             'marketing': int(human_decisions.get('marketingSpend', 500))
#         })
        
#         # Prepare context for AI frameworks
#         base_context = {
#             'turn': self.turn,
#             'competitor_prices': [team['price'] for team in self.teams.values()],
#         }
        
#         # Process each AI framework
#         frameworks = [
#             ('blue', 'mesa'),
#             ('purple', 'temporal'),
#             ('orange', 'google_adk')
#         ]
        
#         framework_decisions = {}
#         for color, framework_name in frameworks:
#             context = base_context.copy()
#             context.update({
#                 'inventory': self.teams[color]['inventory'],
#                 'current_price': self.teams[color]['price']
#             })
            
#             # Get AI decision
#             decision = await self.ai_helper.make_decision(framework_name, context)
#             framework_decisions[framework_name] = decision
            
#             # Update team
#             self.teams[color].update({
#                 'price': decision['price'],
#                 'production': decision['production'],
#                 'marketing': decision['marketing']
#             })
        
#         # Calculate results for all teams
#         turn_results = {}
#         for color, team in self.teams.items():
#             demand = self._calculate_demand(team['price'], team['marketing'])
#             sales = min(demand, team['inventory'])
#             revenue = sales * team['price']
#             costs = (team['production'] * 3.5 + 
#                     team['marketing'] + 
#                     team['inventory'] * 0.5)
#             profit_this_turn = revenue - costs
            
#             # Update team
#             team['inventory'] = max(0, team['inventory'] - sales + team['production'])
#             team['profit'] += profit_this_turn
#             team['profit_this_turn'] = profit_this_turn
            
#             turn_results[color] = {
#                 'sales': sales,
#                 'demand': demand,
#                 'profit_this_turn': profit_this_turn
#             }
        
#         # Determine best performer
#         ai_performances = [
#             ('Mesa', turn_results['blue']['profit_this_turn']),
#             ('Temporal', turn_results['purple']['profit_this_turn']),
#             ('Google ADK', turn_results['orange']['profit_this_turn'])
#         ]
#         best_ai = max(ai_performances, key=lambda x: x[1])
        
#         # Update event log
#         self.event_log = [
#             f"📊 Turn {self.turn} - Framework Battle Results",
#             f"👤 Human: ${human_team['price']:.2f} → {turn_results['green']['sales']} sales, ${turn_results['green']['profit_this_turn']:+.0f} profit",
#             f"🤖 Mesa: ${self.teams['blue']['price']:.2f} → {turn_results['blue']['sales']} sales, ${turn_results['blue']['profit_this_turn']:+.0f} profit",
#             f"⚡ Temporal: ${self.teams['purple']['price']:.2f} → {turn_results['purple']['sales']} sales, ${turn_results['purple']['profit_this_turn']:+.0f} profit", 
#             f"🧠 Google ADK: ${self.teams['orange']['price']:.2f} → {turn_results['orange']['sales']} sales, ${turn_results['orange']['profit_this_turn']:+.0f} profit",
#             f"🏆 Best AI This Turn: {best_ai[0]} (${best_ai[1]:+.0f})",
#             f"💡 AI Mode: {'Gemini Enhanced' if self.ai_helper.enabled else 'Smart Fallbacks'}"
#         ]
        
#         return self.get_game_state()
    
#     def _calculate_demand(self, price: float, marketing: int) -> int:
#         """Calculate market demand"""
#         base_demand = 50
#         marketing_boost = (marketing / 500) * 15
#         price_effect = (10 - price) * 5
#         random_factor = random.randint(-10, 10)
        
#         demand = int(base_demand + marketing_boost + price_effect + random_factor)
#         return max(10, min(120, demand))
    
#     def get_game_state(self) -> Dict[str, Any]:
#         """Get current game state in expected format"""
#         state = {
#             'turn': self.turn,
#             'competition_mode': self.competition_mode,
#             'event_log': self.event_log
#         }
        
#         # Add team data in expected format
#         for color, team in self.teams.items():
#             prefix = f"{color}_team"
#             for key, value in team.items():
#                 if key != 'name':  # Exclude name from state
#                     state[f"{prefix}_{key}"] = value
        
#         return state
    
#     async def run_competition(self, websocket: WebSocket, turns: int):
#         """Run AI framework competition"""
#         self.competition_mode = True
        
#         for turn_num in range(turns):
#             if not self.competition_mode:
#                 break
            
#             # Human team frozen during competition
#             mock_human_decisions = {
#                 'price': 10.0,
#                 'productionTarget': 50,
#                 'marketingSpend': 500
#             }
            
#             # Process turn
#             updated_state = await self.process_turn(mock_human_decisions)
            
#             # Send progress update
#             progress = ((turn_num + 1) / turns) * 100
#             response = updated_state.copy()
#             response.update({
#                 'compete_progress': progress,
#                 'compete_complete': (turn_num + 1 == turns),
#                 'competition_turn': turn_num + 1
#             })
            
#             await websocket.send_text(json.dumps(response))
#             await asyncio.sleep(0.8)  # Pause for visualization
        
#         self.competition_mode = False
        
#         # Send final results
#         final_profits = {
#             'Mesa': self.teams['blue']['profit'],
#             'Temporal': self.teams['purple']['profit'],
#             'Google ADK': self.teams['orange']['profit']
#         }
#         winner = max(final_profits.items(), key=lambda x: x[1])
        
#         final_response = self.get_game_state()
#         final_response.update({
#             'event_log': [
#                 "🏆 FRAMEWORK COMPETITION COMPLETE!",
#                 f"🥇 Champion: {winner[0]} with ${winner[1]:,.0f} total profit!",
#                 "",
#                 "📊 Final Standings:",
#                 *[f"  {name}: ${profit:,.0f}" for name, profit in 
#                   sorted(final_profits.items(), key=lambda x: x[1], reverse=True)],
#                 "",
#                 f"⚔️ Battle Duration: {turns} turns",
#                 f"🤖 AI Mode: {'Gemini Enhanced' if self.ai_helper.enabled else 'Smart Fallbacks'}"
#             ],
#             'compete_progress': 100,
#             'compete_complete': True,
#             'competition_results': final_profits,
#             'winner': winner[0]
#         })
        
#         await websocket.send_text(json.dumps(final_response))

# # FastAPI Application
# app = FastAPI(title="BrewMasters Multi-Framework Battle")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Global coordinator
# coordinator: Optional[BrewMastersCoordinator] = None

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("🔌 Client connected")
    
#     global coordinator
#     if coordinator is None:
#         gemini_api_key = os.getenv("GEMINI_API_KEY")
#         coordinator = BrewMastersCoordinator(gemini_api_key)
    
#     try:
#         # Send initial state
#         initial_state = coordinator.get_game_state()
#         initial_state['server_info'] = {
#             'version': '2.1.0',
#             'frameworks': ['Mesa MAS', 'Temporal Workflows', 'Google ADK ML'],
#             'ai_enabled': coordinator.ai_helper.enabled
#         }
#         await websocket.send_text(json.dumps(initial_state))
        
#         async for message in websocket.iter_text():
#             try:
#                 data = json.loads(message)
                
#                 # Handle commands
#                 if data.get('restart'):
#                     coordinator = BrewMastersCoordinator(coordinator.ai_helper.model is not None)
#                     await websocket.send_text(json.dumps(coordinator.get_game_state()))
#                     continue
                
#                 if data.get('compete'):
#                     turns = data.get('turns', 10)
#                     asyncio.create_task(coordinator.run_competition(websocket, turns))
#                     continue
                
#                 if data.get('stopCompete'):
#                     coordinator.competition_mode = False
#                     continue
                
#                 # Normal turn processing
#                 if not coordinator.competition_mode:
#                     updated_state = await coordinator.process_turn(data)
#                     await websocket.send_text(json.dumps(updated_state))
#                 else:
#                     await websocket.send_text(json.dumps({
#                         "message": "Turn ignored - competition in progress"
#                     }))
                
#             except json.JSONDecodeError as e:
#                 await websocket.send_text(json.dumps({"error": f"Invalid JSON: {e}"}))
#             except Exception as e:
#                 print(f"❌ Error: {e}")
#                 await websocket.send_text(json.dumps({"error": str(e)}))
    
#     except WebSocketDisconnect:
#         print("🔌 Client disconnected")
#     except Exception as e:
#         print(f"❌ WebSocket error: {e}")

# @app.get("/")
# async def read_root():
#     return {
#         "message": "🎮 BrewMasters Multi-Framework Battle Arena",
#         "version": "2.1.0",
#         "status": "Ready for battle!",
#         "frameworks": {
#             "mesa": "Multi-Agent System with specialized coordination",
#             "temporal": "Workflow orchestration with sequential activities",
#             "google_adk": "ML/Cloud optimization with predictive analytics"
#         },
#         "ai_status": coordinator.ai_helper.enabled if coordinator else "Not initialized"
#     }

# @app.get("/status")
# async def get_status():
#     if not coordinator:
#         return {"status": "not_initialized"}
    
#     return {
#         "status": "ready",
#         "turn": coordinator.turn,
#         "competition_mode": coordinator.competition_mode,
#         "ai_enabled": coordinator.ai_helper.enabled,
#         "teams": {
#             color: {
#                 "name": team["name"],
#                 "profit": team["profit"],
#                 "inventory": team["inventory"]
#             }
#             for color, team in coordinator.teams.items()
#         }
#     }

# if __name__ == "__main__":
#     print("🚀 Starting BrewMasters Multi-Framework Battle Arena...")
#     print("🔧 Frameworks: Mesa MAS + Temporal Workflows + Google ADK ML")
#     print("🤖 Gemini AI integration with smart fallbacks")
#     print("⚔️ Ready for epic framework battles!")
    
#     # Check for Gemini API key
#     gemini_key = os.getenv("GEMINI_API_KEY")
#     if gemini_key and gemini_key != "your_gemini_api_key_here":
#         print("✅ Gemini API key detected - AI features enabled")
#     else:
#         print("⚠️ No Gemini API key - using smart fallback strategies")
#         print("💡 Set GEMINI_API_KEY environment variable for AI enhancement")
    
#     print("🌐 Starting server on http://localhost:8000")
#     uvicorn.run(app, host="0.0.0.0", port=8000)

import asyncio
import json
import os
import random
import sys
from typing import Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Better Gemini import detection
GEMINI_AVAILABLE = False
genai = None

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    print("✅ Gemini AI library imported successfully")
except ImportError as e:
    print(f"⚠️ Gemini AI not available: {e}")
    print("💡 Install with: pip install google-generativeai")
except Exception as e:
    print(f"❌ Unexpected error importing Gemini: {e}")

class EnhancedGeminiHelper:
    """Enhanced Gemini helper with better error handling and debugging"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.enabled = False
        self.model = None
        self.model_name = None
        
        print(f"🔧 Initializing Gemini Helper...")
        print(f"📦 Gemini library available: {GEMINI_AVAILABLE}")
        print(f"🔑 API key provided: {'Yes' if api_key else 'No'}")
        
        if not GEMINI_AVAILABLE:
            print("❌ Cannot initialize Gemini - library not available")
            return
            
        if not api_key:
            print("❌ Cannot initialize Gemini - no API key provided")
            print("💡 Set GEMINI_API_KEY environment variable")
            return
        
        if api_key == "your_gemini_api_key_here":
            print("❌ Please set a real API key (not the placeholder)")
            return
        
        # Try to configure and initialize Gemini
        try:
            print(f"🔄 Configuring Gemini with API key: {api_key[:10]}...")
            genai.configure(api_key=api_key)
            
            # Try different models in order of preference
            models_to_try = [
                'gemini-1.5-flash',
                'gemini-1.5-pro',
                'gemini-pro-latest',
                'gemini-1.0-pro'
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"🧪 Testing model: {model_name}")
                    test_model = genai.GenerativeModel(model_name)
                    
                    # Quick test
                    test_response = test_model.generate_content("Hello")
                    if test_response and test_response.text:
                        print(f"✅ Model {model_name} works!")
                        self.model = test_model
                        self.model_name = model_name
                        self.enabled = True
                        break
                    else:
                        print(f"❌ Model {model_name} returned empty response")
                        
                except Exception as model_error:
                    print(f"❌ Model {model_name} failed: {model_error}")
                    continue
            
            if self.enabled:
                print(f"🎯 Successfully initialized with model: {self.model_name}")
            else:
                print("❌ No working models found")
                
        except Exception as e:
            print(f"❌ Gemini initialization failed: {e}")
            self.enabled = False
    
    async def make_decision(self, framework: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Make AI-enhanced or fallback decision"""
        if self.enabled:
            try:
                print(f"🤖 Making AI decision for {framework}...")
                decision = await self._ai_decision(framework, context)
                print(f"✅ AI decision successful for {framework}")
                return decision
            except Exception as e:
                print(f"❌ AI decision failed for {framework}: {e}")
                print("🔄 Falling back to rule-based decision")
                return self._fallback_decision(framework, context)
        else:
            print(f"🔄 Using fallback decision for {framework} (AI disabled)")
            return self._fallback_decision(framework, context)
    
    async def _ai_decision(self, framework: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI decision from Gemini"""
        prompt = f"""
You are a {framework.upper()} AI system making brewery business decisions.

Current Situation:
- Inventory: {context.get('inventory', 100)} units  
- Current Price: ${context.get('current_price', 10.0):.2f}
- Competitor Prices: {context.get('competitor_prices', [10.0])}
- Turn: {context.get('turn', 1)}

Make strategic decisions for price (8.0-15.0), production (10-150), and marketing (0-2000).

Framework approach:
- Mesa: Multi-agent coordination with specialized agents
- Temporal: Workflow-based sequential decision making  
- Google ADK: ML-driven optimization with advanced analytics

Respond only in JSON format:
{{"price": 10.50, "production": 60, "marketing": 800, "reasoning": "Strategy explanation"}}
"""
        
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.model.generate_content(prompt)
            )
            
            if not response or not response.text:
                raise ValueError("Empty response from Gemini")
            
            # Parse response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Validate and constrain
                price = max(8.0, min(15.0, float(data.get('price', 10))))
                production = max(10, min(150, int(data.get('production', 50))))
                marketing = max(0, min(2000, int(data.get('marketing', 500))))
                
                return {
                    'price': round(price, 2),
                    'production': production,
                    'marketing': marketing,
                    'reasoning': f"{framework.upper()} AI ({self.model_name}): {data.get('reasoning', 'Strategic decision')}",
                    'ai_enhanced': True,
                    'framework': framework
                }
            else:
                raise ValueError("No valid JSON in response")
                
        except Exception as e:
            print(f"❌ Gemini API call failed: {e}")
            raise  # Re-raise to trigger fallback
    
    def _fallback_decision(self, framework: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Smart fallback decisions based on framework characteristics"""
        inventory = context.get('inventory', 100)
        competitors = context.get('competitor_prices', [10.0])
        avg_competitor = sum(competitors) / len(competitors)
        
        # Framework-specific strategic approaches
        if framework == "mesa":
            # Mesa: Agent-based coordination with specialization
            if inventory > 150:
                price = max(8.0, avg_competitor - random.uniform(0.8, 1.2))
                production = random.randint(20, 40)
                marketing = random.randint(1200, 1800)
                strategy = "Mesa agents coordinate: discount pricing + heavy marketing"
            elif inventory < 50:
                price = min(15.0, avg_competitor + random.uniform(0.5, 1.0))
                production = random.randint(100, 140)
                marketing = random.randint(200, 500)
                strategy = "Mesa agents coordinate: premium pricing + boost production"
            else:
                price = avg_competitor + random.uniform(-0.3, 0.3)
                production = random.randint(50, 80)
                marketing = random.randint(600, 1000)
                strategy = "Mesa agents coordinate: balanced approach"
                
        elif framework == "temporal":
            # Temporal: Sequential workflow activities
            if inventory > 150:
                price = max(8.0, avg_competitor - random.uniform(1.0, 1.5))
                production = random.randint(15, 35)
                marketing = random.randint(1400, 2000)
                strategy = "Temporal workflow: Analysis→Clearance→Execute"
            elif inventory < 50:
                price = min(15.0, avg_competitor + random.uniform(0.8, 1.3))
                production = random.randint(110, 150)
                marketing = random.randint(150, 400)
                strategy = "Temporal workflow: Analysis→Growth→Execute"
            else:
                price = avg_competitor + random.uniform(-0.4, 0.4)
                production = random.randint(60, 90)
                marketing = random.randint(700, 1100)
                strategy = "Temporal workflow: Analysis→Optimize→Execute"
                
        elif framework == "google_adk":
            # Google ADK: ML/Cloud optimization
            if inventory > 150:
                price = max(8.0, avg_competitor - random.uniform(1.2, 1.8))
                production = random.randint(10, 30)
                marketing = random.randint(1500, 2000)
                strategy = "ML models: Predictive clearance optimization"
            elif inventory < 50:
                price = min(15.0, avg_competitor + random.uniform(1.0, 1.5))
                production = random.randint(120, 150)
                marketing = random.randint(100, 350)
                strategy = "ML models: Predictive growth optimization"
            else:
                price_variance = random.uniform(0.7, 1.3)
                price = avg_competitor * price_variance
                production = random.randint(70, 100)
                marketing = random.randint(800, 1200)
                strategy = "ML ensemble: Multi-objective optimization"
        else:
            price, production, marketing = 10.0, 50, 500
            strategy = "Basic fallback strategy"
        
        mode = "Rule-based Fallback" if self.enabled else "Rule-based (No AI)"
        
        return {
            'price': round(max(8.0, min(15.0, price)), 2),
            'production': max(10, min(150, production)),
            'marketing': max(0, min(2000, marketing)),
            'reasoning': f"{framework.upper()}: {strategy} ({mode})",
            'ai_enhanced': False,
            'framework': framework
        }

class BrewMastersCoordinator:
    """Main coordinator for all three frameworks"""
    
    def __init__(self, gemini_api_key: Optional[str] = None):
        print("🔄 Initializing BrewMasters Coordinator...")
        print(f"🐍 Python version: {sys.version}")
        print(f"📁 Current directory: {os.getcwd()}")
        
        self.ai_helper = EnhancedGeminiHelper(gemini_api_key)
        self.turn = 0
        self.competition_mode = False
        
        # Initialize all teams
        self.teams = {
            'green': {
                'name': 'Human Player',
                'profit': 100000,
                'inventory': 100,
                'price': 10.0,
                'production': 50,
                'marketing': 500,
                'profit_this_turn': 0
            },
            'blue': {
                'name': 'Mesa Multi-Agent System',
                'profit': 100000,
                'inventory': 100,
                'price': 10.0,
                'production': 50,
                'marketing': 500,
                'profit_this_turn': 0
            },
            'purple': {
                'name': 'Temporal Workflow System',
                'profit': 100000,
                'inventory': 100,
                'price': 10.0,
                'production': 50,
                'marketing': 500,
                'profit_this_turn': 0
            },
            'orange': {
                'name': 'Google ADK ML System',
                'profit': 100000,
                'inventory': 100,
                'price': 10.0,
                'production': 50,
                'marketing': 500,
                'profit_this_turn': 0
            }
        }
        
        ai_status = f"Enhanced with Gemini ({self.ai_helper.model_name})" if self.ai_helper.enabled else "Smart Rule-based Fallbacks"
        
        self.event_log = [
            "🎮 Multi-Framework BrewMasters Started!",
            f"🤖 AI Status: {ai_status}",
            "🔧 Frameworks Active: Mesa MAS + Temporal Workflows + Google ADK ML",
            "⚔️ Ready for epic framework battles!"
        ]
        
        print("✅ BrewMasters Coordinator initialized successfully")
        if self.ai_helper.enabled:
            print(f"🎯 AI Model: {self.ai_helper.model_name}")
        else:
            print("⚠️ Using rule-based fallbacks (AI not available)")
    
    async def process_turn(self, human_decisions: Dict[str, Any]) -> Dict[str, Any]:
        """Process complete turn for all frameworks"""
        self.turn += 1
        print(f"\n🎯 Processing Turn {self.turn}")
        
        # Process human team
        human_team = self.teams['green']
        human_team.update({
            'price': float(human_decisions.get('price', 10)),
            'production': int(human_decisions.get('productionTarget', 50)),
            'marketing': int(human_decisions.get('marketingSpend', 500))
        })
        
        print(f"👤 Human decisions: ${human_team['price']:.2f}, {human_team['production']}u, ${human_team['marketing']}m")
        
        # Prepare context for AI frameworks
        base_context = {
            'turn': self.turn,
            'competitor_prices': [team['price'] for team in self.teams.values()],
        }
        
        # Process each AI framework
        frameworks = [
            ('blue', 'mesa'),
            ('purple', 'temporal'),
            ('orange', 'google_adk')
        ]
        
        framework_decisions = {}
        for color, framework_name in frameworks:
            context = base_context.copy()
            context.update({
                'inventory': self.teams[color]['inventory'],
                'current_price': self.teams[color]['price']
            })
            
            # Get AI decision
            decision = await self.ai_helper.make_decision(framework_name, context)
            framework_decisions[framework_name] = decision
            
            # Update team
            self.teams[color].update({
                'price': decision['price'],
                'production': decision['production'],
                'marketing': decision['marketing']
            })
            
            print(f"🤖 {framework_name}: ${decision['price']:.2f}, {decision['production']}u, ${decision['marketing']}m")
        
        # Calculate results for all teams
        turn_results = {}
        for color, team in self.teams.items():
            demand = self._calculate_demand(team['price'], team['marketing'])
            sales = min(demand, team['inventory'])
            revenue = sales * team['price']
            costs = (team['production'] * 3.5 + 
                    team['marketing'] + 
                    team['inventory'] * 0.5)
            profit_this_turn = revenue - costs
            
            # Update team
            team['inventory'] = max(0, team['inventory'] - sales + team['production'])
            team['profit'] += profit_this_turn
            team['profit_this_turn'] = profit_this_turn
            
            turn_results[color] = {
                'sales': sales,
                'demand': demand,
                'profit_this_turn': profit_this_turn
            }
        
        # Determine best performer
        ai_performances = [
            ('Mesa', turn_results['blue']['profit_this_turn']),
            ('Temporal', turn_results['purple']['profit_this_turn']),
            ('Google ADK', turn_results['orange']['profit_this_turn'])
        ]
        best_ai = max(ai_performances, key=lambda x: x[1])
        
        # Update event log
        self.event_log = [
            f"📊 Turn {self.turn} - Framework Battle Results",
            f"👤 Human: ${human_team['price']:.2f} → {turn_results['green']['sales']} sales, ${turn_results['green']['profit_this_turn']:+.0f} profit",
            f"🤖 Mesa: ${self.teams['blue']['price']:.2f} → {turn_results['blue']['sales']} sales, ${turn_results['blue']['profit_this_turn']:+.0f} profit",
            f"⚡ Temporal: ${self.teams['purple']['price']:.2f} → {turn_results['purple']['sales']} sales, ${turn_results['purple']['profit_this_turn']:+.0f} profit", 
            f"🧠 Google ADK: ${self.teams['orange']['price']:.2f} → {turn_results['orange']['sales']} sales, ${turn_results['orange']['profit_this_turn']:+.0f} profit",
            f"🏆 Best AI This Turn: {best_ai[0]} (${best_ai[1]:+.0f})",
            f"💡 AI Mode: {'Gemini Enhanced' if self.ai_helper.enabled else 'Smart Rule-based'}"
        ]
        
        print("✅ Turn processing complete\n")
        return self.get_game_state()
    
    def _calculate_demand(self, price: float, marketing: int) -> int:
        """Calculate market demand"""
        base_demand = 50
        marketing_boost = (marketing / 500) * 15
        price_effect = (10 - price) * 5
        random_factor = random.randint(-10, 10)
        
        demand = int(base_demand + marketing_boost + price_effect + random_factor)
        return max(10, min(120, demand))
    
    def get_game_state(self) -> Dict[str, Any]:
        """Get current game state in expected format"""
        state = {
            'turn': self.turn,
            'competition_mode': self.competition_mode,
            'event_log': self.event_log
        }
        
        # Add team data in expected format
        for color, team in self.teams.items():
            prefix = f"{color}_team"
            for key, value in team.items():
                if key != 'name':  # Exclude name from state
                    state[f"{prefix}_{key}"] = value
        
        return state
    
    async def run_competition(self, websocket: WebSocket, turns: int):
        """Run AI framework competition"""
        self.competition_mode = True
        print(f"🏆 Starting {turns}-turn AI competition")
        
        for turn_num in range(turns):
            if not self.competition_mode:
                break
            
            # Human team frozen during competition
            mock_human_decisions = {
                'price': 10.0,
                'productionTarget': 50,
                'marketingSpend': 500
            }
            
            # Process turn
            updated_state = await self.process_turn(mock_human_decisions)
            
            # Send progress update
            progress = ((turn_num + 1) / turns) * 100
            response = updated_state.copy()
            response.update({
                'compete_progress': progress,
                'compete_complete': (turn_num + 1 == turns),
                'competition_turn': turn_num + 1
            })
            
            await websocket.send_text(json.dumps(response))
            await asyncio.sleep(0.8)  # Pause for visualization
        
        self.competition_mode = False
        
        # Send final results
        final_profits = {
            'Mesa': self.teams['blue']['profit'],
            'Temporal': self.teams['purple']['profit'],
            'Google ADK': self.teams['orange']['profit']
        }
        winner = max(final_profits.items(), key=lambda x: x[1])
        
        final_response = self.get_game_state()
        final_response.update({
            'event_log': [
                "🏆 FRAMEWORK COMPETITION COMPLETE!",
                f"🥇 Champion: {winner[0]} with ${winner[1]:,.0f} total profit!",
                "",
                "📊 Final Standings:",
                *[f"  {name}: ${profit:,.0f}" for name, profit in 
                  sorted(final_profits.items(), key=lambda x: x[1], reverse=True)],
                "",
                f"⚔️ Battle Duration: {turns} turns",
                f"🤖 AI Mode: {'Gemini Enhanced' if self.ai_helper.enabled else 'Smart Rule-based'}"
            ],
            'compete_progress': 100,
            'compete_complete': True,
            'competition_results': final_profits,
            'winner': winner[0]
        })
        
        await websocket.send_text(json.dumps(final_response))
        print(f"🏆 Competition complete! Winner: {winner[0]}")

# FastAPI Application
app = FastAPI(title="BrewMasters Multi-Framework Battle - Enhanced")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global coordinator
coordinator: Optional[BrewMastersCoordinator] = None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")
    
    global coordinator
    if coordinator is None:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        coordinator = BrewMastersCoordinator(gemini_api_key)
    
    try:
        # Send initial state
        initial_state = coordinator.get_game_state()
        initial_state['server_info'] = {
            'version': '2.1.2-enhanced',
            'frameworks': ['Mesa MAS', 'Temporal Workflows', 'Google ADK ML'],
            'ai_enabled': coordinator.ai_helper.enabled,
            'gemini_model': getattr(coordinator.ai_helper, 'model_name', 'N/A')
        }
        await websocket.send_text(json.dumps(initial_state))
        print("📤 Initial state sent")
        
        async for message in websocket.iter_text():
            try:
                data = json.loads(message)
                print(f"📨 Received: {data}")
                
                # Handle commands
                if data.get('restart'):
                    print("🔄 Restarting game")
                    coordinator = BrewMastersCoordinator(os.getenv("GEMINI_API_KEY"))
                    await websocket.send_text(json.dumps(coordinator.get_game_state()))
                    continue
                
                if data.get('compete'):
                    turns = data.get('turns', 10)
                    asyncio.create_task(coordinator.run_competition(websocket, turns))
                    continue
                
                if data.get('stopCompete'):
                    coordinator.competition_mode = False
                    print("⏹️ Competition stopped")
                    continue
                
                # Normal turn processing
                if not coordinator.competition_mode:
                    updated_state = await coordinator.process_turn(data)
                    await websocket.send_text(json.dumps(updated_state))
                else:
                    await websocket.send_text(json.dumps({
                        "message": "Turn ignored - competition in progress"
                    }))
                
            except json.JSONDecodeError as e:
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
    return {
        "message": "🎮 BrewMasters Multi-Framework Battle Arena (Enhanced)",
        "version": "2.1.2-enhanced",
        "status": "Ready for battle!",
        "frameworks": {
            "mesa": "Multi-Agent System with specialized coordination",
            "temporal": "Workflow orchestration with sequential activities",
            "google_adk": "ML/Cloud optimization with predictive analytics"
        },
        "ai_status": coordinator.ai_helper.enabled if coordinator else "Not initialized",
        "gemini_model": getattr(coordinator.ai_helper, 'model_name', 'N/A') if coordinator else "N/A"
    }

@app.get("/status")
async def get_status():
    if not coordinator:
        return {"status": "not_initialized"}
    
    return {
        "status": "ready",
        "turn": coordinator.turn,
        "competition_mode": coordinator.competition_mode,
        "ai_enabled": coordinator.ai_helper.enabled,
        "gemini_model": getattr(coordinator.ai_helper, 'model_name', 'N/A'),
        "teams": {
            color: {
                "name": team["name"],
                "profit": team["profit"],
                "inventory": team["inventory"]
            }
            for color, team in coordinator.teams.items()
        }
    }

if __name__ == "__main__":
    print("🚀 Starting Enhanced BrewMasters Multi-Framework Battle Arena...")
    print("🔧 Frameworks: Mesa MAS + Temporal Workflows + Google ADK ML")
    print("🤖 Enhanced Gemini AI integration with detailed diagnostics")
    print("⚔️ Ready for epic framework battles!")
    
    # Detailed environment check
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        print(f"✅ Gemini API key detected: {gemini_key[:10]}...")
    else:
        print("⚠️ No valid Gemini API key found")
        print("💡 Set GEMINI_API_KEY environment variable for AI enhancement")
    
    print("🌐 Starting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)