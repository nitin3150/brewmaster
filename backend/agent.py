import numpy as np
from mesa import Agent
# --- Constants for the MAS ---
# These can be tuned to change the MAS's "personality"
DESIRED_INVENTORY_WEEKS = 2.5  # How many weeks of sales to keep as safety stock
PROFIT_MARGIN_TARGET = 0.4 # Target a 40% profit margin as a baseline
UNIT_PRODUCTION_COST = 3.0
UNIT_HOLDING_COST = 0.5
class SharedKnowledgeBase:
    """
    An object to hold data and calculations shared across all agents for a single turn.
    This is calculated once at the beginning of the MAS's turn.
    """
    def __init__(self, model):
        self.turn_history = model.turn_history
        self.current_inventory = model.game_state['blue_team_inventory']
        # Core Analytics
        self.sales_history = [h['sales'] for h in self.turn_history if 'sales' in h]
        # Use a moving average of the last 3 turns to predict demand.
        if len(self.sales_history) >= 3:
            self.demand_forecast = np.mean(self.sales_history[-3:])
        elif self.sales_history:
            self.demand_forecast = np.mean(self.sales_history)
        else:
            self.demand_forecast = 50 # Initial guess
        self.unit_cost = UNIT_PRODUCTION_COST + UNIT_HOLDING_COST
class CEOAgent(Agent):
    """
    Coordinates the other agents and makes an optimized final decision based on a global objective.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.final_decisions = {}

    def step(self, proposals):
        """
        The CEO's role is to resolve conflicts and make a final, globally optimal decision.
        It evaluates the combined impact of the proposals on the primary objective: maximizing profit.
        """
        price_prop = proposals['pricing']['price']
        prod_prop = proposals['production']['target']
        mktg_prop = proposals['marketing']['spend']
        # Strategic Override Logic
        # If inventory is dangerously high, prioritize proposals that clear stock.
        if self.model.knowledge_base.current_inventory > (self.model.knowledge_base.demand_forecast * 4):
            final_price = price_prop
            final_production = 0 
            final_marketing = 0 
            self.model.game_state['event_log'].append("MAS CEO: High inventory, prioritizing liquidation.")
        else:
            # Normal operations: Trust the specialized agents' proposals.
            final_price = price_prop
            final_production = prod_prop
            final_marketing = mktg_prop
        self.final_decisions = {
            'price': final_price,
            'marketing_spend': final_marketing,
            'production_target': final_production,
        }
        self.model.game_state['event_log'].append(
            f"MAS CEO Final Decision: Price ${final_price:.2f}, Production {final_production} units, Marketing ${final_marketing}."
        )
class PricingAgent(Agent):
    """
    Dynamically decides the optimal price to maximize profit margin and manage inventory.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'pricing'
        self.proposal = {}
    def step(self):
        kb = self.model.knowledge_base
        base_price = kb.unit_cost / (1 - PROFIT_MARGIN_TARGET)
        inventory_ratio = kb.current_inventory / (kb.demand_forecast * DESIRED_INVENTORY_WEEKS + 1e-6)
        price_adjustment_factor = -np.tanh(inventory_ratio - 1) * 1.5 
        new_price = round(base_price + price_adjustment_factor, 2)
        new_price = max(8.0, min(15.0, new_price))
        self.proposal = {'price': new_price}
class MarketingAgent(Agent):
    """
    Decides on marketing spend by evaluating the potential Return on Investment (ROI).
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'marketing'
        self.proposal = {}
    def step(self):
        kb = self.model.knowledge_base
        if kb.current_inventory < (kb.demand_forecast * 0.8):
            self.proposal = {'spend': 0}
            return
        best_spend = 0
        max_roi = 0.0
        proposed_price = self.model.specialist_agents[0].proposal['price']
        profit_per_unit = proposed_price - kb.unit_cost
        for option in [0, 500, 2000]:
            estimated_demand_boost = (option / 500) * 15
            estimated_additional_profit = estimated_demand_boost * profit_per_unit
            roi = estimated_additional_profit - option
            if roi > max_roi:
                max_roi = roi
                best_spend = option
        self.proposal = {'spend': best_spend}
class ProductionAgent(Agent):
    """
    Decides how much to produce based on a demand forecast and a target inventory level.
    This is designed to dampen the bullwhip effect.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'production'
        self.proposal = {}
    def step(self):
        kb = self.model.knowledge_base
        target_inventory = kb.demand_forecast * DESIRED_INVENTORY_WEEKS
        production_target = (target_inventory - kb.current_inventory) + kb.demand_forecast
        production_target = max(0, int(production_target))
        self.proposal = {'target': production_target}