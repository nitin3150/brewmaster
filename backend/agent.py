import numpy as np
from mesa import Agent

# --- Constants for the MAS ---
DESIRED_INVENTORY_WEEKS = 2.5  # How many weeks of sales to keep as safety stock
PROFIT_MARGIN_TARGET = 0.35    # Target a 35% profit margin (reduced from 40%)
UNIT_PRODUCTION_COST = 3.0
UNIT_HOLDING_COST = 0.5
BASE_MARKET_PRICE = 10.0       # Market reference price

class SharedKnowledgeBase:
    """
    An object to hold data and calculations shared across all agents for a single turn.
    This is calculated once at the beginning of the MAS's turn.
    """
    def __init__(self, model):
        self.turn_history = model.turn_history
        self.current_inventory = model.game_state['blue_team_inventory']
        self.current_price = model.game_state['blue_team_price']
        
        # Core Analytics
        self.sales_history = [h['sales'] for h in self.turn_history if 'sales' in h]
        self.demand_history = [h.get('demand', h['sales']) for h in self.turn_history if 'sales' in h]
        
        # Use demand (not just sales) for better forecasting
        if len(self.demand_history) >= 3:
            # Weighted moving average (recent turns have more weight)
            weights = [0.2, 0.3, 0.5]
            self.demand_forecast = np.average(self.demand_history[-3:], weights=weights)
        elif self.demand_history:
            self.demand_forecast = np.mean(self.demand_history)
        else:
            self.demand_forecast = 50  # Initial guess
        
        # Calculate effective unit cost including holding
        self.unit_cost = UNIT_PRODUCTION_COST
        self.total_unit_cost = UNIT_PRODUCTION_COST + UNIT_HOLDING_COST
        
        # Analyze competitor (human) behavior if available
        self.competitor_price = model.game_state.get('green_team_price', 10)

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
        """
        kb = self.model.knowledge_base
        
        price_prop = proposals['pricing']['price']
        prod_prop = proposals['production']['target']
        mktg_prop = proposals['marketing']['spend']
        
        # Strategic Override Logic
        inventory_ratio = kb.current_inventory / (kb.demand_forecast + 1e-6)
        
        if inventory_ratio > 4:  # Very high inventory
            # Aggressive clearance mode
            final_price = max(8.0, price_prop - 1.0)  # Extra discount
            final_production = 0
            final_marketing = min(1000, mktg_prop * 2)  # Boost marketing to clear stock
            self.model.game_state['event_log'].append("MAS CEO: Critical inventory - clearance mode activated.")
        elif inventory_ratio < 0.5:  # Very low inventory
            # Conservative mode - avoid stockouts
            final_price = min(15.0, price_prop + 0.5)  # Premium pricing
            final_production = int(prod_prop * 1.2)  # Boost production
            final_marketing = 0  # Save money for production
            self.model.game_state['event_log'].append("MAS CEO: Low inventory - boosting production.")
        else:
            # Normal operations
            final_price = price_prop
            final_production = prod_prop
            final_marketing = mktg_prop
        
        self.final_decisions = {
            'price': round(final_price, 2),
            'marketing_spend': final_marketing,
            'production_target': final_production,
        }
        
        self.model.game_state['event_log'].append(
            f"MAS Decision: Price ${final_price:.2f}, Produce {final_production}, Marketing ${final_marketing}"
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
        
        # Start with a market-based approach
        base_price = BASE_MARKET_PRICE
        
        # Adjust based on inventory pressure
        inventory_ratio = kb.current_inventory / (kb.demand_forecast * DESIRED_INVENTORY_WEEKS + 1e-6)
        
        if inventory_ratio > 1.5:  # Too much inventory
            price_adjustment = -1.5 * (inventory_ratio - 1.0)
        elif inventory_ratio < 0.5:  # Too little inventory
            price_adjustment = 1.0 * (0.5 - inventory_ratio)
        else:
            price_adjustment = 0
        
        # Competitive pricing consideration
        if kb.competitor_price < kb.current_price - 1.0:
            # We're significantly more expensive, consider matching
            price_adjustment -= 0.5
        
        # Calculate final price
        new_price = base_price + price_adjustment
        
        # Ensure minimum profit margin
        min_profitable_price = kb.total_unit_cost / (1 - PROFIT_MARGIN_TARGET)
        new_price = max(new_price, min_profitable_price)
        
        # Apply bounds
        new_price = max(8.0, min(15.0, new_price))
        
        self.proposal = {'price': round(new_price, 2)}

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
        
        # Don't market if we have very low inventory
        if kb.current_inventory < (kb.demand_forecast * 0.5):
            self.proposal = {'spend': 0}
            return
        
        # Get proposed price from pricing agent
        proposed_price = self.model.specialist_agents[0].proposal.get('price', kb.current_price)
        profit_per_unit = proposed_price - kb.total_unit_cost
        
        # Marketing effectiveness calculation
        best_spend = 0
        max_expected_profit = 0
        
        for spend_option in [0, 500, 1000, 1500, 2000]:
            # Estimate demand boost from marketing
            marketing_boost = (spend_option / 500) * 15
            
            # Consider inventory constraint
            max_possible_sales = min(kb.demand_forecast + marketing_boost, kb.current_inventory)
            expected_additional_sales = max_possible_sales - kb.demand_forecast
            
            # Calculate expected profit
            expected_profit = (expected_additional_sales * profit_per_unit) - spend_option
            
            if expected_profit > max_expected_profit:
                max_expected_profit = expected_profit
                best_spend = spend_option
        
        self.proposal = {'spend': best_spend}

class ProductionAgent(Agent):
    """
    Decides how much to produce based on a demand forecast and a target inventory level.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'production'
        self.proposal = {}
    
    def step(self):
        kb = self.model.knowledge_base
        
        # Calculate target inventory based on forecasted demand
        target_inventory = kb.demand_forecast * DESIRED_INVENTORY_WEEKS
        
        # Expected sales next turn (considering marketing proposal)
        marketing_proposal = self.model.specialist_agents[1].proposal.get('spend', 0)
        expected_marketing_boost = (marketing_proposal / 500) * 15
        expected_sales = kb.demand_forecast + expected_marketing_boost
        
        # Production needed to reach target inventory after expected sales
        production_needed = (target_inventory - kb.current_inventory) + expected_sales
        
        # Apply smoothing to avoid overreaction
        if len(kb.turn_history) > 2:
            # Dampen production swings
            recent_production = [h.get('production_target', 50) for h in kb.turn_history[-3:]]
            avg_recent_production = np.mean(recent_production)
            production_needed = 0.7 * production_needed + 0.3 * avg_recent_production
        
        # Ensure non-negative production
        production_target = max(0, int(production_needed))
        
        self.proposal = {'target': production_target}