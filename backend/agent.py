import numpy as np
from mesa import Agent
import random

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
        
        # Enhanced demand forecasting with volatility awareness
        if len(self.demand_history) >= 5:
            # Calculate volatility
            self.demand_volatility = np.std(self.demand_history[-5:])
            
            # Weighted moving average with more weight on recent
            weights = [0.1, 0.15, 0.2, 0.25, 0.3]
            base_forecast = np.average(self.demand_history[-5:], weights=weights)
            
            # Add uncertainty based on volatility
            uncertainty = random.gauss(0, self.demand_volatility * 0.3)
            self.demand_forecast = base_forecast + uncertainty
            
        elif len(self.demand_history) >= 3:
            # Simpler forecast for less data
            weights = [0.2, 0.3, 0.5]
            self.demand_forecast = np.average(self.demand_history[-3:], weights=weights)
            self.demand_volatility = np.std(self.demand_history) if len(self.demand_history) > 1 else 10
        elif self.demand_history:
            self.demand_forecast = np.mean(self.demand_history)
            self.demand_volatility = 15  # High uncertainty with little data
        else:
            # Better initial guess with randomness
            self.demand_forecast = random.uniform(45, 65)
            self.demand_volatility = 20  # Very high uncertainty initially
        
        # Ensure reasonable bounds
        self.demand_forecast = max(20, min(120, self.demand_forecast))
        
        # Calculate effective unit cost including holding
        self.unit_cost = UNIT_PRODUCTION_COST
        self.total_unit_cost = UNIT_PRODUCTION_COST + UNIT_HOLDING_COST
        
        # Analyze competitor (human) behavior if available
        self.competitor_price = model.game_state.get('green_team_price', 10)
        
        # Track market trends
        if len(self.sales_history) >= 3:
            recent_trend = self.sales_history[-1] - self.sales_history[-3]
            self.market_trending = "up" if recent_trend > 5 else "down" if recent_trend < -5 else "stable"
        else:
            self.market_trending = "unknown"

class CEOAgent(Agent):
    """
    Coordinates the other agents and makes an optimized final decision based on a global objective.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.final_decisions = {}
        self.risk_tolerance = 0.5  # 0 = very conservative, 1 = very aggressive

    def step(self, proposals):
        """
        The CEO's role is to resolve conflicts and make a final, globally optimal decision.
        """
        kb = self.model.knowledge_base
        
        price_prop = proposals['pricing']['price']
        prod_prop = proposals['production']['target']
        mktg_prop = proposals['marketing']['spend']
        
        # Adjust risk tolerance based on performance
        if len(kb.turn_history) >= 3:
            recent_profits = [h.get('profit_this_turn', 0) for h in kb.turn_history[-3:]]
            if all(p > 0 for p in recent_profits):
                self.risk_tolerance = min(0.8, self.risk_tolerance + 0.1)
            elif all(p < 0 for p in recent_profits):
                self.risk_tolerance = max(0.2, self.risk_tolerance - 0.1)
        
        # Strategic Override Logic with risk adjustment
        inventory_ratio = kb.current_inventory / (kb.demand_forecast + 1e-6)
        
        if inventory_ratio > 2.5:  # High inventory
            # More aggressive clearance based on risk tolerance
            price_discount = 0.5 + (0.5 * self.risk_tolerance)
            final_price = max(8.5, price_prop - price_discount)
            final_production = max(0, int(prod_prop * (0.7 - 0.3 * self.risk_tolerance)))
            final_marketing = min(2000, int(mktg_prop * (1.3 + 0.3 * self.risk_tolerance)))
            self.model.game_state['event_log'].append(f"MAS CEO: High inventory - clearance mode (risk: {self.risk_tolerance:.2f})")
            
        elif inventory_ratio < 1.0:  # Low inventory
            # Conservative production boost
            price_increase = 0.3 + (0.3 * (1 - self.risk_tolerance))
            final_price = min(14.0, price_prop + price_increase)
            final_production = int(prod_prop * (1.2 + 0.2 * self.risk_tolerance))
            final_marketing = int(mktg_prop * (0.8 - 0.2 * self.risk_tolerance))
            self.model.game_state['event_log'].append("MAS CEO: Low inventory - boosting production")
            
        else:
            # Normal operations with small random adjustments
            price_noise = random.uniform(-0.2, 0.2)
            final_price = price_prop + price_noise
            final_production = int(prod_prop * random.uniform(0.95, 1.05))
            final_marketing = mktg_prop
        
        # Add some randomness to avoid predictability
        if random.random() < 0.1:  # 10% chance of random adjustment
            final_price *= random.uniform(0.95, 1.05)
            final_marketing = int(final_marketing * random.uniform(0.8, 1.2))
        
        # Ensure minimum marketing in most cases
        if final_marketing < 100 and inventory_ratio > 0.5:
            final_marketing = random.randint(100, 200)
        
        # Round marketing to nearest $50 with some randomness
        final_marketing = round(final_marketing / 50) * 50 + random.choice([-50, 0, 0, 0, 50])
        final_marketing = max(0, min(2000, final_marketing))
        
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
        self.price_memory = []  # Remember past pricing decisions
    
    def step(self):
        kb = self.model.knowledge_base
        
        # Start with a market-based approach
        base_price = BASE_MARKET_PRICE
        
        # Adjust based on demand volatility
        if hasattr(kb, 'demand_volatility'):
            volatility_adjustment = random.uniform(-0.3, 0.3) * (kb.demand_volatility / 20)
            base_price += volatility_adjustment
        
        # Adjust based on inventory pressure with randomness
        inventory_ratio = kb.current_inventory / (kb.demand_forecast * DESIRED_INVENTORY_WEEKS + 1e-6)
        
        if inventory_ratio > 1.2:
            # Random discount depth
            discount_factor = random.uniform(0.8, 1.2)
            price_adjustment = -discount_factor * (inventory_ratio - 1.0)
        elif inventory_ratio < 0.7:
            # Random premium
            premium_factor = random.uniform(0.6, 1.0)
            price_adjustment = premium_factor * (0.7 - inventory_ratio)
        else:
            # Small random walk
            price_adjustment = random.uniform(-0.2, 0.2)
        
        # Competitive pricing with noise
        if kb.competitor_price < kb.current_price - 1.0:
            competitive_response = random.uniform(0.2, 0.4)
            price_adjustment -= competitive_response
        elif kb.competitor_price > kb.current_price + 1.0:
            competitive_response = random.uniform(0.2, 0.4)
            price_adjustment += competitive_response
        
        # Market trend adjustment
        if hasattr(kb, 'market_trending'):
            if kb.market_trending == "up":
                price_adjustment += random.uniform(0, 0.3)
            elif kb.market_trending == "down":
                price_adjustment -= random.uniform(0, 0.3)
        
        # Calculate final price
        new_price = base_price + price_adjustment
        
        # Ensure minimum profit margin with flexibility
        margin_flexibility = random.uniform(0.7, 0.9)
        min_profitable_price = kb.total_unit_cost / (1 - PROFIT_MARGIN_TARGET * margin_flexibility)
        new_price = max(new_price, min_profitable_price)
        
        # Apply bounds with slight randomness
        lower_bound = random.uniform(8.0, 9.0)
        upper_bound = random.uniform(13.5, 14.5)
        new_price = max(lower_bound, min(upper_bound, new_price))
        
        # Avoid price oscillation
        if self.price_memory:
            recent_avg = np.mean(self.price_memory[-3:])
            if abs(new_price - recent_avg) > 2:
                # Dampen large changes
                new_price = recent_avg + (new_price - recent_avg) * 0.6
        
        self.price_memory.append(new_price)
        if len(self.price_memory) > 5:
            self.price_memory.pop(0)
        
        self.proposal = {'price': round(new_price, 2)}

class MarketingAgent(Agent):
    """
    Decides on marketing spend based on multiple factors with adaptive behavior
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'marketing'
        self.proposal = {}
        self.marketing_effectiveness = 1.0  # Learn from past results
    
    def step(self):
        kb = self.model.knowledge_base
        
        # Get proposed price from pricing agent
        proposed_price = self.model.specialist_agents[0].proposal.get('price', kb.current_price)
        profit_per_unit = proposed_price - kb.total_unit_cost
        
        # If we're losing money per unit, minimal marketing with some randomness
        if profit_per_unit <= 0:
            self.proposal = {'spend': random.randint(0, 100)}
            return
        
        # Update marketing effectiveness based on past results
        if len(kb.turn_history) >= 2:
            last_marketing = kb.turn_history[-1].get('marketing_spend', 500)
            last_sales = kb.turn_history[-1].get('sales', 50)
            expected_sales = kb.turn_history[-2].get('sales', 50) * 1.1
            
            if last_marketing > 0:
                effectiveness_ratio = last_sales / expected_sales
                self.marketing_effectiveness = 0.8 * self.marketing_effectiveness + 0.2 * effectiveness_ratio
                self.marketing_effectiveness = max(0.5, min(1.5, self.marketing_effectiveness))
        
        # PRIMARY FACTOR: Inventory level with randomness
        turns_of_inventory = kb.current_inventory / (kb.demand_forecast + 1)
        
        # Add noise to inventory perception
        perceived_inventory = turns_of_inventory * random.uniform(0.9, 1.1)
        
        if perceived_inventory < 0.5:
            inventory_multiplier = random.uniform(0.05, 0.15)
        elif perceived_inventory < 1.0:
            inventory_multiplier = random.uniform(0.2, 0.4)
        elif perceived_inventory < 2.0:
            inventory_multiplier = random.uniform(0.6, 0.8)
        elif perceived_inventory < 3.0:
            inventory_multiplier = random.uniform(1.0, 1.4)
        elif perceived_inventory < 4.0:
            inventory_multiplier = random.uniform(1.4, 1.8)
        else:
            inventory_multiplier = random.uniform(1.8, 2.2)
        
        # Adjust by learned effectiveness
        inventory_multiplier *= self.marketing_effectiveness
        
        # Calculate base spend with randomness
        excess_inventory = max(0, kb.current_inventory - (kb.demand_forecast * 2))
        spend_per_unit = random.uniform(4, 6)
        base_spend = min(1000, excess_inventory * spend_per_unit)
        
        # Add market volatility factor
        if hasattr(kb, 'demand_volatility'):
            volatility_factor = 1 + (kb.demand_volatility / 50)
            base_spend *= volatility_factor
        
        # Random events (15% chance of marketing campaign)
        if random.random() < 0.15:
            campaign_boost = random.uniform(1.3, 1.7)
            base_spend *= campaign_boost
            self.model.game_state['event_log'].append("MAS Marketing: Special campaign!")
        
        # Calculate final spend
        optimal_spend = base_spend * inventory_multiplier
        
        # Add base amount with randomness
        if optimal_spend < 100 and turns_of_inventory > 0.5:
            optimal_spend = random.randint(100, 200)
        
        # Apply bounds
        optimal_spend = max(0, min(2000, optimal_spend))
        
        # Round to nearest $50 with noise
        optimal_spend = round(optimal_spend / 50) * 50
        if random.random() < 0.3:  # 30% chance of small adjustment
            optimal_spend += random.choice([-50, 50])
        
        optimal_spend = max(0, min(2000, optimal_spend))
        
        self.proposal = {'spend': int(optimal_spend)}

class ProductionAgent(Agent):
    """
    Decides production with adaptive forecasting and uncertainty handling
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.type = 'production'
        self.proposal = {}
        self.forecast_accuracy = 1.0  # Track how accurate our forecasts are
    
    def step(self):
        kb = self.model.knowledge_base
        
        # Update forecast accuracy
        if len(kb.turn_history) >= 2:
            last_production = kb.turn_history[-1].get('production_target', 50)
            last_sales = kb.turn_history[-1].get('sales', 50)
            last_inventory = kb.turn_history[-2].get('inventory', 100) if len(kb.turn_history) > 2 else 100
            
            expected_inventory = last_inventory + last_production - kb.demand_forecast
            actual_inventory = kb.current_inventory
            
            if expected_inventory > 0:
                accuracy = min(expected_inventory, actual_inventory) / max(expected_inventory, actual_inventory)
                self.forecast_accuracy = 0.7 * self.forecast_accuracy + 0.3 * accuracy
        
        # Add uncertainty to target inventory based on volatility
        uncertainty_factor = 1.0
        if hasattr(kb, 'demand_volatility'):
            uncertainty_factor = 1 + (kb.demand_volatility / 100)
        
        # Dynamic target inventory
        base_target_weeks = DESIRED_INVENTORY_WEEKS
        if kb.market_trending == "up":
            base_target_weeks *= random.uniform(1.1, 1.3)
        elif kb.market_trending == "down":
            base_target_weeks *= random.uniform(0.8, 0.95)
        
        target_inventory = kb.demand_forecast * base_target_weeks * uncertainty_factor
        
        # Expected sales with noise
        marketing_proposal = self.model.specialist_agents[1].proposal.get('spend', 500)
        price_proposal = self.model.specialist_agents[0].proposal.get('price', kb.current_price)
        
        # Add randomness to expected effects
        marketing_effect_random = random.uniform(0.02, 0.04)
        expected_marketing_boost = (marketing_proposal / 500) * 15 * marketing_effect_random
        
        price_effect_random = random.uniform(4, 6)
        price_effect = (10 - price_proposal) * price_effect_random
        
        # Base demand with random variation
        base_expected = 50 * random.uniform(0.9, 1.1)
        expected_sales = base_expected + expected_marketing_boost + price_effect
        
        # Adjust by forecast accuracy
        expected_sales *= self.forecast_accuracy
        
        # Production needed with safety margin
        safety_margin = random.uniform(0.9, 1.1)
        production_needed = (target_inventory - kb.current_inventory) + expected_sales
        production_needed *= safety_margin
        
        # Apply smoothing with some randomness
        if len(kb.turn_history) > 2:
            recent_production = [h.get('production_target', 60) for h in kb.turn_history[-3:]]
            avg_recent_production = np.mean(recent_production)
            
            # Random smoothing factor
            smooth_factor = random.uniform(0.5, 0.7)
            production_needed = smooth_factor * production_needed + (1 - smooth_factor) * avg_recent_production
        
        # Add occasional production shocks (10% chance)
        if random.random() < 0.1:
            shock_factor = random.choice([0.7, 0.8, 1.2, 1.3])
            production_needed *= shock_factor
            self.model.game_state['event_log'].append(f"MAS Production: Shock adjustment {shock_factor:.1f}x")
        
        # Ensure reasonable bounds with randomness
        min_production = random.randint(15, 25)
        max_production = random.randint(140, 160)
        production_target = max(min_production, min(max_production, int(production_needed)))
        
        self.proposal = {'target': production_target}