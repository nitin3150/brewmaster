# production_agent.py - Specialized agent for production planning decisions

import random  # For generating random values in fallback scenarios
from typing import Dict, Any, List  # Type hints for better code clarity

# Import base classes and configuration
from base_agent import BaseAgent, AgentConfig  # Parent class and shared config
from gemini_helper import GeminiAIHelper      # AI integration helper

class ProductionAgent(BaseAgent):
    """
    Specialized agent responsible for production planning decisions
    Inherits from BaseAgent and implements production-specific decision logic
    """
    
    def __init__(self, ai_helper: GeminiAIHelper):
        """
        Initialize the production agent with AI capabilities
        
        Args:
            ai_helper: Shared AI helper instance for intelligent decision making
        """
        # Initialize parent class with agent type "production"
        super().__init__("production")
        
        # Store reference to AI helper for making intelligent decisions
        self.ai_helper = ai_helper
        
        # Production-specific tracking variables
        self.production_history = []      # Track all production quantities set
        self.demand_forecasts = []       # Track demand predictions for accuracy
        self.inventory_efficiency = []   # Track how well we manage inventory
        
        print("Production Agent initialized and ready for planning")
    
    async def make_decision(self, context: Dict[str, Any], framework_context: str = "") -> Dict[str, Any]:
        """
        Make a production planning decision based on market conditions
        
        Args:
            context: Market context containing inventory, demand, competitor data
            framework_context: Information about which framework is calling this agent
            
        Returns:
            Dictionary containing production decision and detailed reasoning
        """
        print(f"Production Agent analyzing production needs for {framework_context}")
        
        # Store current context for internal analysis
        self.current_context = context
        
        # Calculate production recommendations using multiple approaches
        production_analysis = self._analyze_production_needs(context)
        
        # Add production analysis to context for AI decision
        enhanced_context = context.copy()
        enhanced_context['production_analysis'] = production_analysis
        
        # Get AI decision (with fallback to rule-based if AI fails)
        decision = await self.ai_helper.get_ai_decision(
            "production", enhanced_context, framework_context
        )
        
        # If AI decision doesn't contain production data, create it
        if 'production' not in decision:
            decision = await self._make_production_specific_decision(context, framework_context)
        
        # Validate production quantity is within bounds
        if 'production' in decision:
            decision['production'] = int(self.validate_decision_bounds(
                "production", decision['production']
            ))
        
        # Add production-specific analysis and insights
        decision = self._add_production_analysis(decision, context, production_analysis)
        
        # Store this decision in agent history
        self.add_decision_to_history(decision)
        
        # Track production quantity for trend analysis
        self.production_history.append(decision['production'])
        if len(self.production_history) > 50:  # Limit history size
            self.production_history = self.production_history[-50:]
        
        print(f"Production Agent decision: {decision['production']} units")
        return decision
    
    def _analyze_production_needs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current production needs based on market conditions
        
        Args:
            context: Market context data
            
        Returns:
            Dictionary containing production need analysis
        """
        # Extract key variables from context
        current_inventory = context.get('inventory', 100)
        sales_history = context.get('sales_history', [])
        turn = context.get('turn', 1)
        
        # Calculate demand forecast based on historical sales
        if len(sales_history) >= 3:
            # Use weighted average of recent sales (more weight on recent data)
            weights = [0.2, 0.3, 0.5]  # 50% weight on most recent, 30% on second, 20% on third
            recent_sales = sales_history[-3:]
            demand_forecast = sum(sale * weight for sale, weight in zip(recent_sales, weights))
        elif len(sales_history) >= 1:
            # If limited history, use simple average
            demand_forecast = sum(sales_history) / len(sales_history)
        else:
            # If no history, use baseline estimate
            demand_forecast = 50  # Default expected demand
        
        # Calculate target inventory level (weeks of sales to maintain as buffer)
        target_inventory = demand_forecast * AgentConfig.DESIRED_INVENTORY_WEEKS
        
        # Calculate inventory gap (positive = need more, negative = have excess)
        inventory_gap = target_inventory - current_inventory
        
        # Calculate base production need (gap + expected sales)
        base_production_need = inventory_gap + demand_forecast
        
        # Calculate inventory velocity (how fast we move inventory)
        inventory_velocity = demand_forecast / current_inventory if current_inventory > 0 else 0
        
        return {
            'demand_forecast': round(demand_forecast, 1),
            'target_inventory': round(target_inventory, 1),
            'current_inventory': current_inventory,
            'inventory_gap': round(inventory_gap, 1),
            'base_production_need': round(base_production_need, 1),
            'inventory_velocity': round(inventory_velocity, 3),
            'production_urgency': self._assess_production_urgency(inventory_gap, inventory_velocity)
        }
    
    def _assess_production_urgency(self, inventory_gap: float, velocity: float) -> str:
        """
        Assess how urgently we need to adjust production
        
        Args:
            inventory_gap: Difference between target and current inventory
            velocity: How fast we're moving inventory
            
        Returns:
            String describing production urgency level
        """
        # High urgency: Large inventory gap or very fast/slow velocity
        if abs(inventory_gap) > 50 or velocity > 0.8 or velocity < 0.2:
            return "high"
        
        # Medium urgency: Moderate inventory gap
        elif abs(inventory_gap) > 20:
            return "medium"
        
        # Low urgency: Small adjustments needed
        else:
            return "low"
    
    async def _make_production_specific_decision(self, context: Dict[str, Any], 
                                               framework_context: str) -> Dict[str, Any]:
        """
        Create a production-specific decision with detailed prompts
        
        Args:
            context: Market context
            framework_context: Framework calling this agent
            
        Returns:
            Production decision dictionary
        """
        # Get our detailed production analysis
        production_analysis = context.get('production_analysis', {})
        
        # Create specialized prompt for production planning
        production_prompt = self._create_production_prompt(context, framework_context, production_analysis)
        
        # Try AI decision with specialized prompt
        if self.ai_helper.enabled:
            try:
                import asyncio
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.ai_helper.model.generate_content(production_prompt)
                )
                
                if response and response.text:
                    return self.ai_helper._parse_ai_response(response.text, "production")
                else:
                    raise ValueError("Empty AI response")
                    
            except Exception as e:
                print(f"Specialized production AI failed: {e}")
        
        # Fallback to rule-based decision
        return self._production_fallback_decision(context, production_analysis)
    
    def _create_production_prompt(self, context: Dict[str, Any], framework_context: str, 
                                production_analysis: Dict[str, Any]) -> str:
        """
        Create detailed prompt specifically for production planning
        
        Args:
            context: Market context
            framework_context: Framework information
            production_analysis: Detailed production needs analysis
            
        Returns:
            Formatted prompt for production decisions
        """
        # Extract variables for prompt
        inventory = context.get('inventory', 100)
        turn = context.get('turn', 1)
        
        # Create comprehensive production planning prompt
        prompt = f"""
SPECIALIZED PRODUCTION PLANNING AGENT

You are an expert PRODUCTION PLANNER responsible for optimizing production quantities.
Framework Context: {framework_context}

CURRENT PRODUCTION ANALYSIS:
- Current Inventory: {inventory} units
- Demand Forecast: {production_analysis.get('demand_forecast', 50)} units/turn
- Target Inventory: {production_analysis.get('target_inventory', 125)} units
- Inventory Gap: {production_analysis.get('inventory_gap', 0)} units
- Base Production Need: {production_analysis.get('base_production_need', 50)} units
- Inventory Velocity: {production_analysis.get('inventory_velocity', 0.5)} (turnover rate)
- Production Urgency: {production_analysis.get('production_urgency', 'medium')}

PRODUCTION STRATEGY CONSIDERATIONS:

1. INVENTORY OPTIMIZATION:
   - Maintain {AgentConfig.DESIRED_INVENTORY_WEEKS} weeks of sales as safety stock
   - Avoid stockouts (lost sales) and overstock (holding costs)
   
2. COST MANAGEMENT:
   - Production cost: ${AgentConfig.UNIT_PRODUCTION_COST}/unit
   - Holding cost: ${AgentConfig.UNIT_HOLDING_COST}/unit/turn
   - Total cost optimization
   
3. DEMAND RESPONSIVENESS:
   - Expected demand: {production_analysis.get('demand_forecast', 50)} units
   - Seasonal variations and trends
   - Market growth/decline patterns
   
4. CAPACITY PLANNING:
   - Production constraints: {AgentConfig.MIN_PRODUCTION}-{AgentConfig.MAX_PRODUCTION} units
   - Efficiency and utilization optimization

PRODUCTION SCENARIOS:
- If inventory < 50 units: URGENT production increase needed
- If inventory > 150 units: Reduce production to avoid excess
- If demand increasing: Scale up production proactively  
- If demand stable: Maintain steady production flow

YOUR TASK:
Determine optimal production quantity that:
- Maintains adequate inventory levels
- Minimizes total costs (production + holding)
- Responds to demand patterns
- Supports business strategy

RESPOND IN EXACT JSON FORMAT:
{{
    "production": 65,
    "reasoning": "Detailed production planning strategy and rationale", 
    "confidence": 0.85,
    "inventory_strategy": "build_up/maintain/reduce",
    "cost_optimization": "explanation of cost considerations",
    "demand_response": "how production responds to expected demand"
}}
"""
        return prompt
    
    def _production_fallback_decision(self, context: Dict[str, Any], 
                                   production_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback for production decisions when AI is unavailable
        
        Args:
            context: Market context
            production_analysis: Production needs analysis
            
        Returns:
            Rule-based production decision
        """
        # Extract key metrics
        inventory = context.get('inventory', 100)
        demand_forecast = production_analysis.get('demand_forecast', 50)
        inventory_gap = production_analysis.get('inventory_gap', 0)
        urgency = production_analysis.get('production_urgency', 'medium')
        
        # Rule-based production logic based on inventory situation
        if inventory < 30:
            # Critical low inventory - maximum production
            production = AgentConfig.MAX_PRODUCTION - random.randint(0, 20)
            strategy = "build_up"
            reasoning = f"Critical inventory shortage ({inventory} units) - maximum production to prevent stockouts"
            
        elif inventory < 70:
            # Low inventory - increased production  
            production = min(AgentConfig.MAX_PRODUCTION, int(demand_forecast * 1.8 + random.randint(10, 30)))
            strategy = "build_up"
            reasoning = f"Low inventory ({inventory} units) - increased production to rebuild stock"
            
        elif inventory > 200:
            # Excess inventory - minimal production
            production = AgentConfig.MIN_PRODUCTION + random.randint(0, 15)
            strategy = "reduce"
            reasoning = f"Excess inventory ({inventory} units) - minimal production to reduce holding costs"
            
        elif inventory > 140:
            # High inventory - reduced production
            production = max(AgentConfig.MIN_PRODUCTION, int(demand_forecast * 0.6 + random.randint(-10, 10)))
            strategy = "reduce"
            reasoning = f"High inventory ({inventory} units) - reduced production to normalize levels"
            
        else:
            # Normal inventory - demand-based production
            production = int(demand_forecast + inventory_gap * 0.5 + random.randint(-5, 15))
            strategy = "maintain"
            reasoning = f"Normal inventory ({inventory} units) - demand-responsive production"
        
        # Ensure production is within valid bounds
        production = max(AgentConfig.MIN_PRODUCTION, min(AgentConfig.MAX_PRODUCTION, production))
        
        # Calculate efficiency metrics for this decision
        holding_cost_impact = inventory * AgentConfig.UNIT_HOLDING_COST
        production_cost_impact = production * AgentConfig.UNIT_PRODUCTION_COST
        
        return {
            'production': production,
            'reasoning': reasoning,
            'confidence': 0.65,  # Moderate confidence for rule-based decisions
            'inventory_strategy': strategy,
            'cost_optimization': f"Production cost: ${production_cost_impact:.0f}, Holding cost: ${holding_cost_impact:.0f}",
            'demand_response': f"Responds to forecast demand of {demand_forecast:.0f} units",
            'agent_type': 'production',
            'ai_enhanced': False,
            'source': 'rule_based_fallback'
        }
    
    def _add_production_analysis(self, decision: Dict[str, Any], context: Dict[str, Any], 
                               production_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add production-specific analysis and insights to the decision
        
        Args:
            decision: Base decision to enhance
            context: Market context
            production_analysis: Production needs analysis
            
        Returns:
            Enhanced decision with production analysis
        """
        # Calculate production efficiency metrics
        production_qty = decision.get('production', 50)
        current_inventory = context.get('inventory', 100)
        
        # Calculate expected inventory after production
        expected_inventory_after_production = current_inventory + production_qty
        
        # Estimate weeks of coverage this inventory provides
        demand_forecast = production_analysis.get('demand_forecast', 50)
        weeks_coverage = expected_inventory_after_production / demand_forecast if demand_forecast > 0 else 0
        
        # Calculate cost implications
        production_cost = production_qty * AgentConfig.UNIT_PRODUCTION_COST
        expected_holding_cost = expected_inventory_after_production * AgentConfig.UNIT_HOLDING_COST
        total_cost = production_cost + expected_holding_cost
        
        # Add comprehensive production analysis
        decision['production_analysis'] = {
            'production_quantity': production_qty,
            'expected_inventory_after': expected_inventory_after_production,
            'weeks_of_coverage': round(weeks_coverage, 1),
            'cost_breakdown': {
                'production_cost': round(production_cost, 2),
                'expected_holding_cost': round(expected_holding_cost, 2),
                'total_cost': round(total_cost, 2)
            },
            'efficiency_metrics': {
                'cost_per_unit': round(total_cost / production_qty if production_qty > 0 else 0, 2),
                'inventory_turnover': round(1 / weeks_coverage if weeks_coverage > 0 else 0, 2)
            }
        }
        
        return decision
    
    def forecast_demand(self, sales_history: List[int], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advanced demand forecasting based on historical sales and market conditions
        
        Args:
            sales_history: List of historical sales figures
            context: Current market context for forecast adjustment
            
        Returns:
            Dictionary containing demand forecast and confidence
        """
        # If no sales history, use baseline forecast
        if not sales_history:
            return {
                'forecast': 50,                    # Default baseline demand
                'confidence': 0.3,                 # Low confidence without data
                'method': 'baseline_estimate'
            }
        
        # Simple moving average forecast
        if len(sales_history) >= 5:
            # Use weighted moving average for better accuracy
            weights = [0.1, 0.15, 0.2, 0.25, 0.3]  # More weight on recent data
            recent_sales = sales_history[-5:]
            forecast = sum(sale * weight for sale, weight in zip(recent_sales, weights))
            confidence = 0.8
            method = 'weighted_moving_average'
            
        elif len(sales_history) >= 3:
            # Use simple moving average
            forecast = sum(sales_history[-3:]) / 3
            confidence = 0.6
            method = 'simple_moving_average'
            
        else:
            # Use what data we have
            forecast = sum(sales_history) / len(sales_history)
            confidence = 0.4
            method = 'limited_data_average'
        
        # Adjust forecast based on market conditions
        price = context.get('current_price', 10.0)
        marketing = context.get('marketing_planned', 500)
        
        # Price elasticity adjustment (lower price = higher demand)
        price_adjustment = (10.0 - price) * 3  # Each $1 below $10 adds ~3 units demand
        
        # Marketing impact adjustment (every $500 marketing adds ~15 units demand)
        marketing_adjustment = (marketing / 500) * 15
        
        # Apply adjustments to base forecast
        adjusted_forecast = forecast + price_adjustment + marketing_adjustment
        
        # Ensure forecast is within reasonable bounds
        adjusted_forecast = max(10, min(120, adjusted_forecast))
        
        return {
            'forecast': round(adjusted_forecast, 1),
            'confidence': confidence,
            'method': method,
            'base_forecast': round(forecast, 1),
            'price_adjustment': round(price_adjustment, 1),
            'marketing_adjustment': round(marketing_adjustment, 1)
        }
    
    def calculate_optimal_production(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate mathematically optimal production quantity
        
        Args:
            context: Market context for optimization
            
        Returns:
            Dictionary with optimal production calculation
        """
        # Get demand forecast
        demand_info = self.forecast_demand(context.get('sales_history', []), context)
        demand_forecast = demand_info['forecast']
        
        # Current inventory level
        current_inventory = context.get('inventory', 100)
        
        # Target inventory calculation
        target_inventory = demand_forecast * AgentConfig.DESIRED_INVENTORY_WEEKS
        
        # Basic production need (what we need to reach target plus cover expected sales)
        basic_need = (target_inventory - current_inventory) + demand_forecast
        
        # Apply safety factor based on demand confidence
        safety_factor = 1.0 + (1.0 - demand_info['confidence']) * 0.3  # Up to 30% safety buffer
        optimal_production = basic_need * safety_factor
        
        # Round and constrain to valid range
        optimal_production = max(AgentConfig.MIN_PRODUCTION, 
                               min(AgentConfig.MAX_PRODUCTION, round(optimal_production)))
        
        return {
            'optimal_quantity': optimal_production,
            'demand_forecast': demand_forecast,
            'target_inventory': target_inventory,
            'safety_factor': round(safety_factor, 2),
            'calculation_confidence': demand_info['confidence']
        }
    
    def get_production_insights(self) -> Dict[str, Any]:
        """
        Get detailed insights about production agent performance
        
        Returns:
            Dictionary containing production-specific insights
        """
        # Get base agent statistics
        base_stats = self.get_agent_stats()
        
        # Add production-specific insights
        production_insights = {
            'production_patterns': {
                'min_production': min(self.production_history) if self.production_history else 0,
                'max_production': max(self.production_history) if self.production_history else 0,
                'avg_production': sum(self.production_history) / len(self.production_history) if self.production_history else 0,
                'production_volatility': self._calculate_production_volatility()
            },
            'efficiency_tracking': self._analyze_efficiency_trends(),
            'forecast_accuracy': self._evaluate_forecast_accuracy()
        }
        
        # Combine base stats with production insights
        base_stats['production_insights'] = production_insights
        return base_stats
    
    def _calculate_production_volatility(self) -> float:
        """
        Calculate how much production varies over time (measure of consistency)
        
        Returns:
            Standard deviation of production history
        """
        if len(self.production_history) < 2:
            return 0.0
        
        # Calculate mean production
        mean_production = sum(self.production_history) / len(self.production_history)
        
        # Calculate variance
        variance = sum((prod - mean_production) ** 2 for prod in self.production_history) / len(self.production_history)
        
        # Return standard deviation
        return variance ** 0.5
    
    def _analyze_efficiency_trends(self) -> Dict[str, Any]:
        """
        Analyze production efficiency trends over time
        
        Returns:
            Dictionary with efficiency analysis
        """
        if len(self.decision_history) < 3:
            return {'status': 'insufficient_data'}
        
        # Track different production strategies used
        strategy_counts = {}
        for decision in self.decision_history:
            strategy = decision.get('inventory_strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Find most common strategy
        most_common_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else 'unknown'
        
        # Calculate average cost efficiency
        total_cost_efficiency = 0
        efficiency_decisions = 0
        
        for decision in self.decision_history:
            if 'production_analysis' in decision and 'efficiency_metrics' in decision['production_analysis']:
                cost_per_unit = decision['production_analysis']['efficiency_metrics'].get('cost_per_unit', 0)
                if cost_per_unit > 0:
                    total_cost_efficiency += 1 / cost_per_unit  # Higher efficiency = lower cost per unit
                    efficiency_decisions += 1
        
        avg_efficiency = total_cost_efficiency / efficiency_decisions if efficiency_decisions > 0 else 0
        
        return {
            'strategy_distribution': strategy_counts,
            'most_common_strategy': most_common_strategy,
            'average_cost_efficiency': round(avg_efficiency, 3),
            'decisions_analyzed': len(self.decision_history)
        }
    
    def _evaluate_forecast_accuracy(self) -> Dict[str, Any]:
        """
        Evaluate how accurate our demand forecasts have been
        
        Returns:
            Dictionary with forecast accuracy analysis
        """
        if len(self.demand_forecasts) < 2:
            return {'status': 'insufficient_forecast_data'}
        
        # Compare forecasts to actual sales (if available)
        # This would be populated during actual gameplay
        accuracy_scores = []
        
        # For now, simulate accuracy evaluation
        # In real implementation, this would compare forecasts to actual sales
        estimated_accuracy = random.uniform(0.6, 0.9)  # Simulated accuracy
        
        return {
            'estimated_accuracy': round(estimated_accuracy, 2),
            'forecasts_made': len(self.demand_forecasts),
            'accuracy_trend': 'improving',  # Could be calculated from actual data
            'forecast_method': 'weighted_moving_average'
        }

# Example usage and testing
if __name__ == "__main__":
    """
    Test the production agent independently
    This section only runs when this file is executed directly
    """
    import asyncio
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    async def test_production_agent():
        """Test function to verify production agent functionality"""
        print("Testing Production Agent...")
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize AI helper and production agent
        ai_helper = GeminiAIHelper(api_key)
        production_agent = ProductionAgent(ai_helper)
        
        # Create test scenarios with different inventory levels
        test_scenarios = [
            {
                'name': 'Low Inventory Scenario',
                'context': {
                    'inventory': 25,                    # Very low inventory
                    'current_price': 10.50,
                    'competitor_prices': [9.80, 11.20],
                    'turn': 3,
                    'sales_history': [48, 52, 55]      # Increasing sales trend
                }
            },
            {
                'name': 'High Inventory Scenario', 
                'context': {
                    'inventory': 180,                   # Excess inventory
                    'current_price': 9.50,
                    'competitor_prices': [9.20, 10.80],
                    'turn': 8,
                    'sales_history': [35, 42, 38, 40]  # Stable but low sales
                }
            },
            {
                'name': 'Normal Inventory Scenario',
                'context': {
                    'inventory': 90,                    # Normal inventory
                    'current_price': 10.75,
                    'competitor_prices': [10.00, 11.50, 10.25],
                    'turn': 12,
                    'sales_history': [50, 48, 52, 49, 51]  # Stable sales
                }
            }
        ]
        
        # Test each scenario
        for scenario in test_scenarios:
            print(f"\n=== {scenario['name']} ===")
            
            # Make production decision
            decision = await production_agent.make_decision(
                scenario['context'], 
                f"TEST_{scenario['name'].replace(' ', '_').upper()}"
            )
            
            # Display results
            print(f"Production Decision: {decision['production']} units")
            print(f"Strategy: {decision.get('inventory_strategy', 'N/A')}")
            print(f"Reasoning: {decision['reasoning'][:80]}...")
            print(f"Confidence: {decision['confidence']:.2f}")
            print(f"AI Enhanced: {decision['ai_enhanced']}")
            
            # Show production analysis if available
            if 'production_analysis' in decision:
                analysis = decision['production_analysis']
                print(f"Expected inventory after: {analysis['expected_inventory_after']} units")
                print(f"Weeks coverage: {analysis['weeks_of_coverage']} weeks")
                print(f"Total cost: ${analysis['cost_breakdown']['total_cost']:.2f}")
        
        # Show overall agent insights
        print(f"\n=== Production Agent Performance ===")
        insights = production_agent.get_production_insights()
        print(f"Total decisions: {insights['decisions_made']}")
        print(f"Average confidence: {insights['average_confidence']:.2f}")
        
        if 'production_insights' in insights:
            prod_insights = insights['production_insights']['production_patterns']
            print(f"Production range: {prod_insights['min_production']}-{prod_insights['max_production']} units")
            print(f"Average production: {prod_insights['avg_production']:.1f} units")
    
    # Run the test
    asyncio.run(test_production_agent())