import json
from mesa import Model
from mesa.time import RandomActivation
from agent import PricingAgent, MarketingAgent, ProductionAgent, CEOAgent, SharedKnowledgeBase

class BrewMastersModel(Model):
    def __init__(self):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.turn = 0
        self.turn_history = []
        self.knowledge_base = None
        
        # Game State Dictionary
        self.game_state = {
            "turn": 0,
            "green_team_profit": 100000,
            "green_team_inventory": 100,
            "green_team_price": 10,
            "green_team_projected_demand": 50,  # Added
            "blue_team_profit": 100000,
            "blue_team_inventory": 100,
            "blue_team_price": 10,
            "blue_team_projected_demand": 50,   # Added
            "event_log": ["Game Started!"]
        }
        
        # Create the MAS team
        self.ceo_agent = CEOAgent(1, self)
        self.specialist_agents = [
            PricingAgent(2, self),
            MarketingAgent(3, self),
            ProductionAgent(4, self)
        ]
        for agent in self.specialist_agents:
            self.schedule.add(agent)
    
    def calculate_demand(self, price, marketing_spend):
        """Calculate demand based on price and marketing spend."""
        base_demand = 50
        marketing_boost = (marketing_spend / 500) * 15
        price_elasticity = (10 - price) * 5
        demand = int(base_demand + marketing_boost + price_elasticity)
        return max(0, demand)
    
    def calculate_outcome(self, team_prefix, price, marketing_spend, production_target):
        """Calculates sales and profit for one turn for one team."""
        # Calculate actual demand
        demand = self.calculate_demand(price, marketing_spend)
        
        inventory_key = f"{team_prefix}_inventory"
        profit_key = f"{team_prefix}_profit"
        
        sales = min(demand, self.game_state[inventory_key])
        revenue = sales * price
        production_cost = production_target * 3
        inventory_cost = self.game_state[inventory_key] * 0.5
        profit_this_turn = revenue - production_cost - inventory_cost - marketing_spend
        
        # Update state
        self.game_state[profit_key] += profit_this_turn
        self.game_state[inventory_key] -= sales
        self.game_state[inventory_key] += production_target
        self.game_state[f"{team_prefix}_price"] = price
        self.game_state[f"{team_prefix}_profit_this_turn"] = profit_this_turn
        self.game_state[f"{team_prefix}_production_target"] = production_target
        
        # Store actual demand for this turn
        self.game_state[f"{team_prefix}_actual_demand"] = demand
        
        return {'sales': sales, 'profit_this_turn': profit_this_turn, 'demand': demand}
    
    def update_projected_demands(self):
        """Update projected demands for both teams based on current state."""
        # For human team - simple projection based on current price
        if hasattr(self, 'last_human_marketing'):
            human_projected = self.calculate_demand(
                self.game_state['green_team_price'], 
                self.last_human_marketing
            )
        else:
            human_projected = self.calculate_demand(self.game_state['green_team_price'], 500)
        
        self.game_state['green_team_projected_demand'] = human_projected
        
        # For MAS team - use the knowledge base forecast
        if self.knowledge_base:
            self.game_state['blue_team_projected_demand'] = int(self.knowledge_base.demand_forecast)
        else:
            self.game_state['blue_team_projected_demand'] = 50
    
    def get_state_as_json(self):
        """Serializes the current game state to JSON."""
        return json.dumps(self.game_state, indent=2)
    
    def step(self, human_decisions):
        """Processes one full turn of the game with the new MAS logic."""
        self.turn += 1
        self.game_state['turn'] = self.turn
        self.game_state['event_log'] = [f"--- Turn {self.turn} ---"]
        
        # Process Green Team (Human)
        human_price = float(human_decisions.get('price', 10))
        human_marketing = float(human_decisions.get('marketingSpend', 0))
        human_production = int(human_decisions.get('productionTarget', 50))
        
        # Store for projection
        self.last_human_marketing = human_marketing
        
        human_outcome = self.calculate_outcome("green_team", human_price, human_marketing, human_production)
        self.game_state['event_log'].append(
            f"Human: Sold {human_outcome['sales']} units (Demand: {human_outcome['demand']})"
        )
        
        # --- Process Blue Team (MAS) ---
        # 1. Update the shared knowledge base with current state
        self.knowledge_base = SharedKnowledgeBase(self)
        
        # 2. Specialist agents generate proposals
        self.schedule.step() 
        
        # 3. Collect proposals for the CEO
        proposals = {agent.type: agent.proposal for agent in self.specialist_agents}
        
        # 4. CEO agent makes the final decision
        self.ceo_agent.step(proposals)
        mas_decisions = self.ceo_agent.final_decisions
        
        # 5. Execute the MAS decision and calculate the outcome
        mas_outcome = self.calculate_outcome(
            "blue_team", 
            mas_decisions['price'], 
            mas_decisions['marketing_spend'], 
            mas_decisions['production_target']
        )
        
        self.game_state['event_log'].append(
            f"MAS: Sold {mas_outcome['sales']} units (Demand: {mas_outcome['demand']})"
        )
        
        # 6. Record the Blue Team's results for future analysis
        history_entry = {
            "turn": self.turn,
            "price": mas_decisions['price'],
            "marketing_spend": mas_decisions['marketing_spend'],
            "production_target": mas_decisions['production_target'],
            "sales": mas_outcome['sales'],
            "demand": mas_outcome['demand'],
            "profit_this_turn": mas_outcome['profit_this_turn']
        }
        self.turn_history.append(history_entry)
        
        # 7. Update projected demands for next turn
        self.update_projected_demands()