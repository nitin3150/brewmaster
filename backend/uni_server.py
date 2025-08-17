import asyncio
import json
import random
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, Optional, List
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleAI:
    """Simple AI that makes strategic decisions"""
    
    def __init__(self, team_name: str):
        self.team_name = team_name
        self.strategy_state = {
            'aggressive': False,
            'last_profit_change': 0,
            'panic_mode': False
        }
    
    def make_decision(self, current_state: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced AI decision making with varying strategies"""
        inventory = current_state['inventory']
        profit = current_state.get('profit', 100000)
        turn = current_state.get('turn', 1)
        competitor_prices = current_state.get('competitor_prices', [10])
        
        # Calculate average competitor price for competitive analysis
        avg_competitor_price = sum(competitor_prices) / len(competitor_prices) if competitor_prices else 10
        
        # Different AI personalities
        if self.team_name == "Mesa":
            # Mesa AI: Analytical and reactive to inventory levels
            if inventory > 150:
                price = round(max(8.0, avg_competitor_price - random.uniform(1.0, 2.0)), 2)
                marketing = random.randint(1500, 2000)
                production = random.randint(10, 30)
                strategy = "Clearance mode - high inventory"
                
            elif inventory > 100:
                price = round(avg_competitor_price - random.uniform(0.2, 0.8), 2)
                marketing = random.randint(800, 1200)
                production = random.randint(40, 60)
                strategy = "Competitive pricing"
                
            elif inventory < 50:
                price = round(min(14.0, avg_competitor_price + random.uniform(0.5, 1.5)), 2)
                marketing = random.randint(200, 500)
                production = random.randint(80, 120)
                strategy = "Premium pricing - low stock"
                
            else:
                price = round(avg_competitor_price + random.uniform(-0.3, 0.3), 2)
                marketing = random.randint(600, 1000)
                production = random.randint(45, 75)
                strategy = "Balanced strategy"
                
        elif self.team_name == "Temporal":
            # Temporal AI: More aggressive and trend-following
            inventory_ratio = inventory / 100
            
            if inventory_ratio > 2.0:
                price = round(random.uniform(8.0, 9.5), 2)
                marketing = random.randint(1800, 2000)
                production = random.randint(5, 25)
                strategy = "Panic liquidation"
                
            elif inventory_ratio > 1.5:
                price = round(avg_competitor_price - random.uniform(0.5, 1.2), 2)
                marketing = random.randint(1200, 1600)
                production = random.randint(30, 50)
                strategy = "Moderate discounting"
                
            elif inventory_ratio < 0.3:
                price = round(min(15.0, avg_competitor_price + random.uniform(1.0, 2.5)), 2)
                marketing = random.randint(100, 300)
                production = random.randint(100, 150)
                strategy = "Scarcity premium"
                
            elif inventory_ratio < 0.7:
                price = round(avg_competitor_price + random.uniform(0.3, 1.0), 2)
                marketing = random.randint(400, 700)
                production = random.randint(70, 100)
                strategy = "Premium approach"
                
            else:
                trend_adjustment = random.uniform(-0.5, 0.5)
                price = round(avg_competitor_price + trend_adjustment, 2)
                marketing = random.randint(500, 1000)
                production = random.randint(40, 80)
                strategy = "Trend following"
        
        elif self.team_name == "GoogleADK":
            # Google ADK AI: Data-driven and machine learning-based decisions
            profit_velocity = (profit - 100000) / max(turn, 1)
            market_score = sum(competitor_prices) / len(competitor_prices) if competitor_prices else 10
            
            if inventory > 180:
                price = round(max(8.0, market_score * random.uniform(0.6, 0.8)), 2)
                marketing = random.randint(1600, 2000)
                production = random.randint(5, 20)
                strategy = "ML Liquidation Algorithm"
                
            elif inventory > 120:
                price_optimization = market_score * random.uniform(0.85, 0.95)
                price = round(price_optimization, 2)
                marketing = random.randint(1200, 1500)
                production = random.randint(25, 45)
                strategy = "Predictive Pricing Model"
                
            elif inventory < 30:
                scarcity_multiplier = random.uniform(1.15, 1.4)
                price = round(min(15.0, market_score * scarcity_multiplier), 2)
                marketing = random.randint(100, 400)
                production = random.randint(90, 140)
                strategy = "Scarcity Revenue Optimization"
                
            elif inventory < 70:
                forecast_adjustment = random.uniform(1.05, 1.2)
                price = round(market_score * forecast_adjustment, 2)
                marketing = random.randint(500, 800)
                production = random.randint(70, 100)
                strategy = "Demand Forecasting Model"
                
            else:
                neural_adjustment = random.uniform(0.95, 1.15)
                price = round(market_score * neural_adjustment, 2)
                marketing = random.randint(600, 1000)
                production = random.randint(50, 80)
                strategy = "Neural Network Adaptive"
            
            # Advanced analytics features
            if random.random() < 0.15:
                feature_type = random.choice(['A/B_test', 'reinforcement_learning', 'ensemble_method'])
                
                if feature_type == 'A/B_test':
                    if random.choice([True, False]):
                        price *= random.uniform(0.9, 1.1)
                        marketing = int(marketing * random.uniform(1.2, 1.5))
                    strategy += " + A/B Testing"
                    
                elif feature_type == 'reinforcement_learning':
                    if profit_velocity > 0:
                        production = int(production * random.uniform(1.1, 1.3))
                    else:
                        price *= random.uniform(0.8, 1.2)
                    strategy += " + Reinforcement Learning"
                    
                elif feature_type == 'ensemble_method':
                    ensemble_price = (price + market_score + avg_competitor_price) / 3
                    price = round(ensemble_price * random.uniform(0.95, 1.05), 2)
                    strategy += " + Ensemble Method"
        
        else:
            # Fallback for other teams
            price = round(random.uniform(9.0, 12.0), 2)
            marketing = random.randint(400, 800)
            production = random.randint(40, 70)
            strategy = "Basic strategy"
        
        # Add random events (10% chance)
        if random.random() < 0.1:
            event_type = random.choice(['aggressive', 'conservative', 'experimental'])
            if event_type == 'aggressive':
                price *= 0.9
                marketing = int(marketing * 1.5)
                strategy += " + aggressive campaign"
            elif event_type == 'conservative':
                price *= 1.1
                marketing = int(marketing * 0.7)
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
            'reasoning': f"{self.team_name}: {strategy} → ${price:.2f}, {production}u, ${marketing}m",
            'strategy': strategy
        }

class SimpleGameSession:
    def __init__(self):
        self.turn = 0
        self.competition_mode = False
        
        # Create AI players - IMPORTANT: All four must be initialized here
        self.mesa_ai = SimpleAI("Mesa")
        self.temporal_ai = SimpleAI("Temporal")
        self.google_adk_ai = SimpleAI("GoogleADK")  # This was missing!
        self.human_ai = SimpleAI("Human")
        
        # Game state with all four teams
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
            # Google ADK Team (Orange)
            'orange_team_profit': 100000,
            'orange_team_inventory': 100,
            'orange_team_price': 10,
            'orange_team_production': 50,
            'orange_team_marketing': 500,
            'orange_team_projected_demand': 50,
            'orange_team_profit_this_turn': 0,
            # Event log
            'event_log': ["🎮 Server Started!", "Four-way competition: Human vs Mesa vs Temporal vs Google ADK"],
            # History
            'sales_history': {'green': [], 'blue': [], 'purple': [], 'orange': []},
            'production_history': {'green': [], 'blue': [], 'purple': [], 'orange': []},
            'price_history': {'green': [], 'blue': [], 'purple': [], 'orange': []}
        }
    
    async def process_turn(self, human_decisions: dict) -> dict:
        """Process one turn for all four teams"""
        print(f"🎯 Processing turn {self.turn + 1}")
        
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
            'competitor_prices': [human_price, self.game_state['purple_team_price'], self.game_state['orange_team_price']]
        }
        mesa_decisions = self.mesa_ai.make_decision(mesa_state)
        
        self.game_state['blue_team_price'] = mesa_decisions['price']
        self.game_state['blue_team_production'] = mesa_decisions['production']
        self.game_state['blue_team_marketing'] = mesa_decisions['marketing']
        
        # Process Temporal (Purple) team
        temporal_state = {
            'inventory': self.game_state['purple_team_inventory'],
            'profit': self.game_state['purple_team_profit'],
            'turn': self.turn,
            'competitor_prices': [human_price, mesa_decisions['price'], self.game_state['orange_team_price']]
        }
        temporal_decisions = self.temporal_ai.make_decision(temporal_state)
        
        self.game_state['purple_team_price'] = temporal_decisions['price']
        self.game_state['purple_team_production'] = temporal_decisions['production']
        self.game_state['purple_team_marketing'] = temporal_decisions['marketing']
        
        # Process Google ADK (Orange) team
        google_state = {
            'inventory': self.game_state['orange_team_inventory'],
            'profit': self.game_state['orange_team_profit'],
            'turn': self.turn,
            'competitor_prices': [human_price, mesa_decisions['price'], temporal_decisions['price']]
        }
        google_decisions = self.google_adk_ai.make_decision(google_state)
        
        self.game_state['orange_team_price'] = google_decisions['price']
        self.game_state['orange_team_production'] = google_decisions['production']
        self.game_state['orange_team_marketing'] = google_decisions['marketing']
        
        # Calculate results for all teams
        teams = [
            ('green_team', 'green'),
            ('blue_team', 'blue'), 
            ('purple_team', 'purple'),
            ('orange_team', 'orange')
        ]
        
        for team_prefix, color in teams:
            price = self.game_state[f'{team_prefix}_price']
            marketing = self.game_state[f'{team_prefix}_marketing']
            production = self.game_state[f'{team_prefix}_production']
            inventory = self.game_state[f'{team_prefix}_inventory']
            
            # Simple demand calculation
            base_demand = 50
            price_effect = (12 - price) * 5
            marketing_effect = marketing * 0.02
            random_effect = random.randint(-15, 15)
            
            demand = max(10, min(120, int(base_demand + price_effect + marketing_effect + random_effect)))
            sales = min(demand, inventory)
            
            # Update inventory
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
            f"🔶 Google ADK: {google_decisions['reasoning']}",
            f"Four-way competition active"
        ]
        
        print(f"✅ Turn {self.turn} processed successfully")
        return self.game_state
    
    async def run_competition(self, websocket: WebSocket, turns: int):
        """Run AI vs AI competition"""
        self.competition_mode = True
        print(f"🏆 Starting competition: {turns} turns")
        
        for i in range(turns):
            if not self.competition_mode:
                break
            
            self.turn += 1
            self.game_state['turn'] = self.turn
            
            # All teams use AI in competition mode
            # Process all AI teams
            mesa_state = {'inventory': self.game_state['blue_team_inventory'], 'profit': self.game_state['blue_team_profit'], 'turn': self.turn, 'competitor_prices': [10]}
            mesa_decisions = self.mesa_ai.make_decision(mesa_state)
            
            temporal_state = {'inventory': self.game_state['purple_team_inventory'], 'profit': self.game_state['purple_team_profit'], 'turn': self.turn, 'competitor_prices': [mesa_decisions['price']]}
            temporal_decisions = self.temporal_ai.make_decision(temporal_state)
            
            google_state = {'inventory': self.game_state['orange_team_inventory'], 'profit': self.game_state['orange_team_profit'], 'turn': self.turn, 'competitor_prices': [mesa_decisions['price'], temporal_decisions['price']]}
            google_decisions = self.google_adk_ai.make_decision(google_state)
            
            # Update all teams
            self.game_state['blue_team_price'] = mesa_decisions['price']
            self.game_state['blue_team_production'] = mesa_decisions['production']
            self.game_state['blue_team_marketing'] = mesa_decisions['marketing']
            
            self.game_state['purple_team_price'] = temporal_decisions['price']
            self.game_state['purple_team_production'] = temporal_decisions['production']
            self.game_state['purple_team_marketing'] = temporal_decisions['marketing']
            
            self.game_state['orange_team_price'] = google_decisions['price']
            self.game_state['orange_team_production'] = google_decisions['production']
            self.game_state['orange_team_marketing'] = google_decisions['marketing']
            
            # Calculate results for AI teams only (human frozen)
            for team_prefix, color in [('blue_team', 'blue'), ('purple_team', 'purple'), ('orange_team', 'orange')]:
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
                f"🔶 Google ADK: {google_decisions['reasoning']}",
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
        profits = {
            'Mesa': self.game_state['blue_team_profit'],
            'Temporal': self.game_state['purple_team_profit'], 
            'Google ADK': self.game_state['orange_team_profit']
        }
        winner = max(profits.items(), key=lambda x: x[1])[0]
        
        final_response = self.game_state.copy()
        final_response['event_log'] = [
            "🏆 AI COMPETITION COMPLETE!",
            f"Winner: {winner}",
            f"Mesa: ${self.game_state['blue_team_profit']:,.0f}",
            f"Temporal: ${self.game_state['purple_team_profit']:,.0f}",
            f"Google ADK: ${self.game_state['orange_team_profit']:,.0f}"
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
                
                # Handle normal turn
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
    return {"message": "🎮 BrewMasters Four-Way Competition Server", "status": "OK"}

@app.get("/status")
async def get_status():
    return {
        "server": "brewmasters_four_way",
        "turn": game_session.turn,
        "competition_mode": game_session.competition_mode,
        "teams": 4
    }

if __name__ == "__main__":
    print("🚀 Starting BrewMasters Four-Way Competition Server...")
    print("🎯 Teams: Human vs Mesa vs Temporal vs Google ADK")
    uvicorn.run(app, host="0.0.0.0", port=8000)