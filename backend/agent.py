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
            # Better initial guess based on average price
            self.demand_forecast = 60  # Increased from 50
        
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
        
        # Strategic Override Logic - ADJUSTED THRESHOLDS
        inventory_ratio = kb.current_inventory / (kb.demand_forecast + 1e-6)
        
        if inventory_ratio > 3.0:  # Reduced from 4.0
            # Moderate clearance mode
            final_price = max(8.5, price_prop - 0.5)  # Less aggressive discount
            final_production = max(0, int(prod_prop * 0.5))  # Reduce but don't stop
            # Boost marketing by 50% for clearance, but cap at 2000
            final_marketing = min(2000, int(mktg_prop * 1.5))
            self.model.game_state['event_log'].append("MAS CEO: High inventory - moderate clearance mode.")
        elif inventory_ratio < 0.8:  # Increased from 0.5
            # Conservative mode - avoid stockouts
            final_price = min(14.0, price_prop + 0.5)  # Not max price
            final_production = int(prod_prop * 1.3)  # Moderate boost
            # Reduce marketing by 30% when low on inventory
            final_marketing = int(mktg_prop * 0.7)
            self.model.game_state['event_log'].append("MAS CEO: Low inventory - boosting production.")
        else:
            # Normal operations - trust the specialists
            final_price = price_prop
            final_production = prod_prop
            final_marketing = mktg_prop
        
        # Ensure minimum marketing in most cases (unless very low inventory)
        if final_marketing < 100 and inventory_ratio > 0.5:
            final_marketing = 100  # Minimum marketing presence
        
        # Round marketing to nearest $10 for cleaner numbers
        final_marketing = round(final_marketing / 10) * 10
        
        self.final_decisions = {
            'price': round(final_price, 2),
            'marketing_spend': int(final_marketing),
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
        
        if inventory_ratio > 1.2:  # Reduced threshold
            price_adjustment = -1.0 * (inventory_ratio - 1.0)  # Less aggressive
        elif inventory_ratio < 0.7:  # Adjusted threshold
            price_adjustment = 0.8 * (0.7 - inventory_ratio)
        else:
            price_adjustment = 0
        
        # Competitive pricing consideration
        if kb.competitor_price < kb.current_price - 1.0:
            # We're significantly more expensive, consider matching
            price_adjustment -= 0.3  # Less reactive
        elif kb.competitor_price > kb.current_price + 1.0:
            # Competitor is more expensive, we can increase slightly
            price_adjustment += 0.3
        
        # Calculate final price
        new_price = base_price + price_adjustment
        
        # Ensure minimum profit margin but be more flexible
        min_profitable_price = kb.total_unit_cost / (1 - PROFIT_MARGIN_TARGET * 0.8)  # Accept 80% of target margin
        new_price = max(new_price, min_profitable_price)
        
        # Apply bounds with wider range
        new_price = max(8.5, min(14.0, new_price))  # Adjusted bounds
        
        self.proposal = {'price': round(new_price, 2)}

class MarketingAgent(Agent):
    """
    Decides on marketing spend primarily based on inventory levels and profit margins.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'marketing'
        self.proposal = {}
    
    def step(self):
        kb = self.model.knowledge_base
        
        # Get proposed price from pricing agent
        proposed_price = self.model.specialist_agents[0].proposal.get('price', kb.current_price)
        profit_per_unit = proposed_price - kb.total_unit_cost
        
        # If we're losing money per unit, minimal marketing
        if profit_per_unit <= 0:
            self.proposal = {'spend': 0}
            return
        
        # PRIMARY FACTOR: Inventory level
        # Calculate how many turns of inventory we have
        turns_of_inventory = kb.current_inventory / (kb.demand_forecast + 1)
        
        # Inventory-based marketing curve
        if turns_of_inventory < 0.5:
            # Very low inventory - minimal marketing
            inventory_multiplier = 0.1
        elif turns_of_inventory < 1.0:
            # Low inventory - reduced marketing
            inventory_multiplier = 0.3
        elif turns_of_inventory < 2.0:
            # Normal inventory - moderate marketing
            inventory_multiplier = 0.7
        elif turns_of_inventory < 3.0:
            # High inventory - increased marketing
            inventory_multiplier = 1.2
        elif turns_of_inventory < 4.0:
            # Very high inventory - aggressive marketing
            inventory_multiplier = 1.6
        else:
            # Excess inventory - maximum marketing
            inventory_multiplier = 2.0
        
        # SECONDARY FACTORS
        
        # 1. Profit margin factor (higher margins = can afford more marketing)
        margin_percentage = (profit_per_unit / proposed_price) * 100
        if margin_percentage > 40:
            margin_multiplier = 1.2
        elif margin_percentage > 30:
            margin_multiplier = 1.0
        elif margin_percentage > 20:
            margin_multiplier = 0.8
        else:
            margin_multiplier = 0.6
        
        # 2. Competition factor
        price_diff = proposed_price - kb.competitor_price
        if price_diff > 2:
            # We're much more expensive - need marketing to justify
            competition_multiplier = 1.3
        elif price_diff > 0:
            # Slightly more expensive
            competition_multiplier = 1.1
        elif price_diff > -2:
            # Competitive pricing
            competition_multiplier = 1.0
        else:
            # We're cheaper - less marketing needed
            competition_multiplier = 0.8
        
        # 3. Sales velocity factor (are we moving inventory?)
        if len(kb.sales_history) >= 2:
            recent_sales_avg = np.mean(kb.sales_history[-2:])
            if recent_sales_avg > kb.demand_forecast * 1.2:
                # Selling very well
                velocity_multiplier = 0.9
            elif recent_sales_avg < kb.demand_forecast * 0.7:
                # Selling poorly - boost marketing if we have inventory
                velocity_multiplier = 1.2 if turns_of_inventory > 1 else 0.8
            else:
                velocity_multiplier = 1.0
        else:
            velocity_multiplier = 1.0
        
        # Calculate base spend
        # Base formula: we want to spend more when we have more excess inventory to move
        excess_inventory = max(0, kb.current_inventory - (kb.demand_forecast * 2))
        base_spend = min(1000, excess_inventory * 5)  # $5 per excess unit, capped at $1000
        
        # Apply all multipliers
        optimal_spend = base_spend * inventory_multiplier * margin_multiplier * competition_multiplier * velocity_multiplier
        
        # Add a small base amount to ensure some marketing presence
        if optimal_spend < 100 and turns_of_inventory > 0.5:
            optimal_spend = 100
        
        # Apply bounds
        optimal_spend = max(0, min(2000, optimal_spend))
        
        # Round to nearest $50
        optimal_spend = round(optimal_spend / 50) * 50
        
        # Log decision reasoning
        self.model.game_state['event_log'].append(
            f"MAS Marketing: Inventory {kb.current_inventory} units ({turns_of_inventory:.1f} turns) → ${int(optimal_spend)}"
        )
        
        self.proposal = {'spend': int(optimal_spend)}

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
        marketing_proposal = self.model.specialist_agents[1].proposal.get('spend', 500)
        expected_marketing_boost = (marketing_proposal / 500) * 15
        
        # Consider price effect on demand
        price_proposal = self.model.specialist_agents[0].proposal.get('price', kb.current_price)
        price_effect = (10 - price_proposal) * 5
        
        expected_sales = 50 + expected_marketing_boost + price_effect
        
        # Production needed to reach target inventory after expected sales
        production_needed = (target_inventory - kb.current_inventory) + expected_sales
        
        # Apply smoothing to avoid overreaction
        if len(kb.turn_history) > 2:
            # Dampen production swings
            recent_production = [h.get('production_target', 60) for h in kb.turn_history[-3:]]
            avg_recent_production = np.mean(recent_production)
            production_needed = 0.6 * production_needed + 0.4 * avg_recent_production
        
        # Ensure reasonable production bounds
        production_target = max(20, min(150, int(production_needed)))  # Never less than 20
        
        self.proposal = {'target': production_target}