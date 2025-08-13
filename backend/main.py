# BrewMasters Multi-Agent System using Temporal
# Requirements: pip install temporalio fastapi websockets uvicorn

import asyncio
import json
import numpy as np
from datetime import timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

from temporalio import activity, workflow
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.common import RetryPolicy

# FastAPI for WebSocket support
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- Constants ---
DESIRED_INVENTORY_WEEKS = 2.5
PROFIT_MARGIN_TARGET = 0.35
UNIT_PRODUCTION_COST = 3.0
UNIT_HOLDING_COST = 0.5
BASE_MARKET_PRICE = 10.0

# --- Data Classes ---
@dataclass
class GameState:
    turn: int
    green_team_profit: float
    green_team_inventory: int
    green_team_price: float
    green_team_projected_demand: int
    blue_team_profit: float
    blue_team_inventory: int
    blue_team_price: float
    blue_team_projected_demand: int
    event_log: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SharedKnowledge:
    turn_history: List[Dict[str, Any]]
    current_inventory: int
    current_price: float
    sales_history: List[int]
    demand_history: List[int]
    demand_forecast: float
    unit_cost: float
    total_unit_cost: float
    competitor_price: float

@dataclass
class AgentProposal:
    agent_type: str
    proposal: Dict[str, Any]

@dataclass
class MASDecision:
    price: float
    marketing_spend: int
    production_target: int

# --- Agent Base Classes ---
class BaseAgent(ABC):
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
    
    @abstractmethod
    async def make_proposal(self, knowledge: SharedKnowledge) -> Dict[str, Any]:
        pass

# --- Concrete Agents ---
class PricingAgent(BaseAgent):
    async def make_proposal(self, kb: SharedKnowledge) -> Dict[str, Any]:
        base_price = BASE_MARKET_PRICE
        
        # Inventory pressure
        inventory_ratio = kb.current_inventory / (kb.demand_forecast * DESIRED_INVENTORY_WEEKS + 1e-6)
        
        if inventory_ratio > 1.2:
            price_adjustment = -1.0 * (inventory_ratio - 1.0)
        elif inventory_ratio < 0.7:
            price_adjustment = 0.8 * (0.7 - inventory_ratio)
        else:
            price_adjustment = 0
        
        # Competitive pricing
        if kb.competitor_price < kb.current_price - 1.0:
            price_adjustment -= 0.3
        elif kb.competitor_price > kb.current_price + 1.0:
            price_adjustment += 0.3
        
        new_price = base_price + price_adjustment
        
        # Ensure minimum profit margin
        min_profitable_price = kb.total_unit_cost / (1 - PROFIT_MARGIN_TARGET * 0.8)
        new_price = max(new_price, min_profitable_price)
        
        # Apply bounds
        new_price = max(8.5, min(14.0, new_price))
        
        return {'price': round(new_price, 2)}

class MarketingAgent(BaseAgent):
    async def make_proposal(self, kb: SharedKnowledge) -> Dict[str, Any]:
        # This agent needs the price proposal, so we'll pass it through the workflow
        return {'needs_price': True}
    
    async def make_final_proposal(self, kb: SharedKnowledge, proposed_price: float) -> Dict[str, Any]:
        profit_per_unit = proposed_price - kb.total_unit_cost
        
        if profit_per_unit <= 0:
            return {'spend': 0}
        
        # Inventory-based marketing
        turns_of_inventory = kb.current_inventory / (kb.demand_forecast + 1)
        
        if turns_of_inventory < 0.5:
            inventory_multiplier = 0.1
        elif turns_of_inventory < 1.0:
            inventory_multiplier = 0.3
        elif turns_of_inventory < 2.0:
            inventory_multiplier = 0.7
        elif turns_of_inventory < 3.0:
            inventory_multiplier = 1.2
        elif turns_of_inventory < 4.0:
            inventory_multiplier = 1.6
        else:
            inventory_multiplier = 2.0
        
        # Calculate base spend
        excess_inventory = max(0, kb.current_inventory - (kb.demand_forecast * 2))
        base_spend = min(1000, excess_inventory * 5)
        
        # Apply multipliers (simplified for brevity)
        optimal_spend = base_spend * inventory_multiplier
        
        if optimal_spend < 100 and turns_of_inventory > 0.5:
            optimal_spend = 100
        
        optimal_spend = max(0, min(2000, optimal_spend))
        optimal_spend = round(optimal_spend / 50) * 50
        
        return {'spend': int(optimal_spend)}

class ProductionAgent(BaseAgent):
    async def make_proposal(self, kb: SharedKnowledge) -> Dict[str, Any]:
        target_inventory = kb.demand_forecast * DESIRED_INVENTORY_WEEKS
        
        # Expected sales (simplified)
        expected_sales = kb.demand_forecast
        
        production_needed = (target_inventory - kb.current_inventory) + expected_sales
        
        # Apply smoothing
        if len(kb.turn_history) > 2:
            recent_production = [h.get('production_target', 60) for h in kb.turn_history[-3:]]
            avg_recent_production = np.mean(recent_production)
            production_needed = 0.6 * production_needed + 0.4 * avg_recent_production
        
        production_target = max(20, min(150, int(production_needed)))
        
        return {'target': production_target}

class CEOAgent(BaseAgent):
    async def make_decision(self, kb: SharedKnowledge, proposals: Dict[str, Dict[str, Any]]) -> MASDecision:
        price_prop = proposals['pricing']['price']
        prod_prop = proposals['production']['target']
        mktg_prop = proposals['marketing']['spend']
        
        # Strategic overrides
        inventory_ratio = kb.current_inventory / (kb.demand_forecast + 1e-6)
        
        if inventory_ratio > 3.0:
            # Clearance mode
            final_price = max(8.5, price_prop - 0.5)
            final_production = max(0, int(prod_prop * 0.5))
            final_marketing = min(2000, int(mktg_prop * 1.5))
        elif inventory_ratio < 0.8:
            # Conservative mode
            final_price = min(14.0, price_prop + 0.5)
            final_production = int(prod_prop * 1.3)
            final_marketing = int(mktg_prop * 0.7)
        else:
            # Normal operations
            final_price = price_prop
            final_production = prod_prop
            final_marketing = mktg_prop
        
        if final_marketing < 100 and inventory_ratio > 0.5:
            final_marketing = 100
        
        final_marketing = round(final_marketing / 10) * 10
        
        return MASDecision(
            price=round(final_price, 2),
            marketing_spend=int(final_marketing),
            production_target=final_production
        )

# --- Temporal Activities ---
@activity.defn
async def create_shared_knowledge(game_state: Dict[str, Any], turn_history: List[Dict[str, Any]]) -> SharedKnowledge:
    """Create shared knowledge base from game state"""
    sales_history = [h['sales'] for h in turn_history if 'sales' in h]
    demand_history = [h.get('demand', h['sales']) for h in turn_history if 'sales' in h]
    
    if len(demand_history) >= 3:
        weights = [0.2, 0.3, 0.5]
        demand_forecast = np.average(demand_history[-3:], weights=weights)
    elif demand_history:
        demand_forecast = np.mean(demand_history)
    else:
        demand_forecast = 60
    
    return SharedKnowledge(
        turn_history=turn_history,
        current_inventory=game_state['blue_team_inventory'],
        current_price=game_state['blue_team_price'],
        sales_history=sales_history,
        demand_history=demand_history,
        demand_forecast=demand_forecast,
        unit_cost=UNIT_PRODUCTION_COST,
        total_unit_cost=UNIT_PRODUCTION_COST + UNIT_HOLDING_COST,
        competitor_price=game_state.get('green_team_price', 10)
    )

@activity.defn
async def pricing_agent_activity(knowledge: SharedKnowledge) -> Dict[str, Any]:
    """Pricing agent makes proposal"""
    agent = PricingAgent("pricing_agent")
    return await agent.make_proposal(knowledge)

@activity.defn
async def marketing_agent_activity(knowledge: SharedKnowledge, proposed_price: float) -> Dict[str, Any]:
    """Marketing agent makes proposal"""
    agent = MarketingAgent("marketing_agent")
    return await agent.make_final_proposal(knowledge, proposed_price)

@activity.defn
async def production_agent_activity(knowledge: SharedKnowledge) -> Dict[str, Any]:
    """Production agent makes proposal"""
    agent = ProductionAgent("production_agent")
    return await agent.make_proposal(knowledge)

@activity.defn
async def ceo_decision_activity(knowledge: SharedKnowledge, proposals: Dict[str, Dict[str, Any]]) -> MASDecision:
    """CEO makes final decision"""
    agent = CEOAgent("ceo_agent")
    return await agent.make_decision(knowledge, proposals)

@activity.defn
async def calculate_game_outcome(
    game_state: Dict[str, Any],
    team_prefix: str,
    price: float,
    marketing_spend: int,
    production_target: int
) -> Dict[str, Any]:
    """Calculate sales and profit outcomes"""
    base_demand = 50
    marketing_boost = (marketing_spend / 500) * 15
    price_elasticity = (10 - price) * 5
    demand = int(base_demand + marketing_boost + price_elasticity)
    demand = max(0, demand)
    
    inventory_key = f"{team_prefix}_inventory"
    profit_key = f"{team_prefix}_profit"
    
    sales = min(demand, game_state[inventory_key])
    revenue = sales * price
    production_cost = production_target * 3
    inventory_cost = game_state[inventory_key] * 0.5
    profit_this_turn = revenue - production_cost - inventory_cost - marketing_spend
    
    # Update state
    game_state[profit_key] += profit_this_turn
    game_state[inventory_key] -= sales
    game_state[inventory_key] += production_target
    game_state[f"{team_prefix}_price"] = price
    
    return {
        'sales': sales,
        'profit_this_turn': profit_this_turn,
        'demand': demand,
        'updated_state': game_state
    }

# --- Temporal Workflows ---
@workflow.defn
class MASDecisionWorkflow:
    """Workflow for MAS team decision making"""
    
    @workflow.run
    async def run(self, game_state: Dict[str, Any], turn_history: List[Dict[str, Any]]) -> MASDecision:
        # Create shared knowledge
        knowledge = await workflow.execute_activity(
            create_shared_knowledge,
            args=[game_state, turn_history],
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=RetryPolicy(maximum_attempts=3)
        )
        
        # Get pricing proposal
        pricing_proposal = await workflow.execute_activity(
            pricing_agent_activity,
            args=[knowledge],
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        # Get marketing proposal (needs price)
        marketing_proposal = await workflow.execute_activity(
            marketing_agent_activity,
            args=[knowledge, pricing_proposal['price']],
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        # Get production proposal
        production_proposal = await workflow.execute_activity(
            production_agent_activity,
            args=[knowledge],
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        # CEO makes final decision
        proposals = {
            'pricing': pricing_proposal,
            'marketing': marketing_proposal,
            'production': production_proposal
        }
        
        decision = await workflow.execute_activity(
            ceo_decision_activity,
            args=[knowledge, proposals],
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        return decision

@workflow.defn
class GameTurnWorkflow:
    """Workflow for processing a complete game turn"""
    
    @workflow.run
    async def run(
        self,
        game_state: Dict[str, Any],
        turn_history: List[Dict[str, Any]],
        human_decisions: Dict[str, Any]
    ) -> Dict[str, Any]:
        turn = game_state['turn'] + 1
        game_state['turn'] = turn
        game_state['event_log'] = [f"--- Turn {turn} ---"]
        
        # Process human team
        human_outcome = await workflow.execute_activity(
            calculate_game_outcome,
            args=[
                game_state.copy(),
                "green_team",
                float(human_decisions.get('price', 10)),
                float(human_decisions.get('marketingSpend', 0)),
                int(human_decisions.get('productionTarget', 50))
            ],
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        game_state = human_outcome['updated_state']
        game_state['event_log'].append(
            f"Human: Sold {human_outcome['sales']} units (Demand: {human_outcome['demand']})"
        )
        
        # Get MAS decision
        mas_decision = await workflow.execute_child_workflow(
            MASDecisionWorkflow.run,
            args=[game_state, turn_history],
            id=f"mas_decision_turn_{turn}"
        )
        
        # Process MAS team
        mas_outcome = await workflow.execute_activity(
            calculate_game_outcome,
            args=[
                game_state.copy(),
                "blue_team",
                mas_decision.price,
                mas_decision.marketing_spend,
                mas_decision.production_target
            ],
            start_to_close_timeout=timedelta(seconds=10)
        )
        
        game_state = mas_outcome['updated_state']
        game_state['event_log'].append(
            f"MAS: Sold {mas_outcome['sales']} units (Demand: {mas_outcome['demand']})"
        )
        game_state['event_log'].append(
            f"MAS Decision: Price ${mas_decision.price:.2f}, "
            f"Produce {mas_decision.production_target}, "
            f"Marketing ${mas_decision.marketing_spend}"
        )
        
        # Update turn history
        turn_history.append({
            "turn": turn,
            "price": mas_decision.price,
            "marketing_spend": mas_decision.marketing_spend,
            "production_target": mas_decision.production_target,
            "sales": mas_outcome['sales'],
            "demand": mas_outcome['demand'],
            "profit_this_turn": mas_outcome['profit_this_turn']
        })
        
        return {
            'game_state': game_state,
            'turn_history': turn_history
        }

# --- FastAPI WebSocket Server ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store game sessions
game_sessions = {}

class GameSession:
    def __init__(self, temporal_client: Client):
        self.client = temporal_client
        self.game_state = {
            "turn": 0,
            "green_team_profit": 100000,
            "green_team_inventory": 100,
            "green_team_price": 10,
            "green_team_projected_demand": 50,
            "blue_team_profit": 100000,
            "blue_team_inventory": 100,
            "blue_team_price": 10,
            "blue_team_projected_demand": 50,
            "event_log": ["Game Started!"]
        }
        self.turn_history = []
    
    async def process_turn(self, human_decisions: Dict[str, Any]) -> Dict[str, Any]:
        # Execute turn workflow
        result = await self.client.execute_workflow(
            GameTurnWorkflow.run,
            args=[self.game_state, self.turn_history, human_decisions],
            id=f"game_turn_{self.game_state['turn'] + 1}_{id(self)}",
            task_queue="brewmasters-queue"
        )
        
        self.game_state = result['game_state']
        self.turn_history = result['turn_history']
        
        return self.game_state

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected via WebSocket")
    
    # Create Temporal client
    temporal_client = await Client.connect("localhost:7233")
    
    # Create game session
    connection_id = id(websocket)
    game_sessions[connection_id] = GameSession(temporal_client)
    session = game_sessions[connection_id]
    
    try:
        # Send initial game state
        await websocket.send_text(json.dumps(session.game_state))
        
        while True:
            message = await websocket.receive_text()
            
            # Check for restart
            if message == '{"restart": true}':
                game_sessions[connection_id] = GameSession(temporal_client)
                session = game_sessions[connection_id]
                await websocket.send_text(json.dumps(session.game_state))
                continue
            
            # Process turn
            human_decisions = json.loads(message)
            print(f"Received human decisions: {human_decisions}")
            
            # Process turn through Temporal workflow
            updated_state = await session.process_turn(human_decisions)
            
            # Send updated state
            await websocket.send_text(json.dumps(updated_state))
    
    except WebSocketDisconnect:
        print("Client disconnected")
        if connection_id in game_sessions:
            del game_sessions[connection_id]
    except Exception as e:
        print(f"Error: {e}")
        if connection_id in game_sessions:
            del game_sessions[connection_id]

# --- Main Application ---
async def run_temporal_worker():
    """Run Temporal worker"""
    client = await Client.connect("localhost:7233")
    
    worker = Worker(
        client,
        task_queue="brewmasters-queue",
        workflows=[MASDecisionWorkflow, GameTurnWorkflow],
        activities=[
            create_shared_knowledge,
            pricing_agent_activity,
            marketing_agent_activity,
            production_agent_activity,
            ceo_decision_activity,
            calculate_game_outcome
        ]
    )
    
    await worker.run()

async def main():
    """Start both Temporal worker and FastAPI server"""
    # Start Temporal worker in background
    worker_task = asyncio.create_task(run_temporal_worker())
    
    # Run FastAPI server
    config = uvicorn.Config(app, host="0.0.0.0", port=8000)
    server = uvicorn.Server(config)
    
    try:
        await server.serve()
    finally:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass

if __name__ == "__main__":
    # To run:
    # 1. Start Temporal: temporal server start-dev
    # 2. Run this script: python brewmasters_temporal.py
    asyncio.run(main())