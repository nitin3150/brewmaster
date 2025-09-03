import random  
from typing import Dict, Any
from base_agent import BaseAgent, AgentConfig
from gemini_helper import GeminiAIHelper

class PricingAgent(BaseAgent):
    """
    Specialized agent responsible for making pricing decisions
    This agent inherits from BaseAgent and implements pricing-specific logic
    """
    
    def __init__(self, ai_helper: GeminiAIHelper):
        """
        Initialize the pricing agent
        
        Args:
            ai_helper: Shared Gemini AI helper instance for making AI decisions
        """
        # Call the parent class constructor with agent type "pricing"
        super().__init__("pricing")
        
        # Store reference to the AI helper for making intelligent decisions
        self.ai_helper = ai_helper
        
        # Pricing-specific tracking variables
        self.price_history = []           # Track all prices this agent has set
        self.competitor_analysis = []     # Track competitor price analysis
        self.margin_performance = []      # Track profit margin performance
        
        print("Pricing Agent initialized")
    
    async def make_decision(self, context: Dict[str, Any], framework_context: str = "") -> Dict[str, Any]:
        """
        Make a pricing decision based on market context
        
        Args:
            context: Dictionary containing market data like inventory, competitors, etc.
            framework_context: String describing which framework is calling this agent
            
        Returns:
            Dictionary containing pricing decision, reasoning, and metadata
        """
        print(f"Pricing Agent making decision for {framework_context}")
        
        # Store current context for analysis
        self.current_context = context
        
        # Try to get AI-enhanced decision first
        decision = await self.ai_helper.get_ai_decision("pricing", context, framework_context)
        
        # If decision doesn't have required pricing data, create it
        if 'price' not in decision:
            # This means AI failed and we got a generic fallback
            decision = await self._make_pricing_specific_decision(context, framework_context)
        
        # Validate the price is within acceptable bounds
        if 'price' in decision:
            decision['price'] = self.validate_decision_bounds("price", decision['price'])
        
        # Add pricing-specific analysis to the decision
        decision = self._add_pricing_analysis(decision, context)
        
        # Track this decision in our history
        self.add_decision_to_history(decision)
        
        # Store price for trend analysis
        self.price_history.append(decision['price'])
        if len(self.price_history) > 50:  # Keep last 50 prices
            self.price_history = self.price_history[-50:]
        
        print(f"Pricing Agent decision: ${decision['price']:.2f}")
        return decision
    
    async def _make_pricing_specific_decision(self, context: Dict[str, Any], 
                                           framework_context: str) -> Dict[str, Any]:
        """
        Make a pricing-specific decision using enhanced prompts
        
        Args:
            context: Market context
            framework_context: Framework calling this agent
            
        Returns:
            Pricing decision dictionary
        """
        # Create detailed pricing-specific prompt
        pricing_prompt = self._create_pricing_prompt(context, framework_context)
        
        # If AI is available, try the specialized pricing prompt
        if self.ai_helper.enabled:
            try:
                # Use the AI helper but with our specialized prompt
                import asyncio
                response = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    lambda: self.ai_helper.model.generate_content(pricing_prompt)
                )
                
                if response and response.text:
                    return self.ai_helper._parse_ai_response(response.text, "pricing")
                else:
                    raise ValueError("Empty AI response")
                    
            except Exception as e:
                print(f"Specialized pricing AI failed: {e}")
        
        # If AI fails or is disabled, use pricing-specific fallback
        return self._pricing_fallback_decision(context)
    
    def _create_pricing_prompt(self, context: Dict[str, Any], framework_context: str) -> str:
        """
        Create a detailed prompt specifically for pricing decisions
        
        Args:
            context: Market context
            framework_context: Framework information
            
        Returns:
            Formatted prompt string for pricing decisions
        """
        # Extract context variables for prompt
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitor_prices = context.get('competitor_prices', [10.0])
        turn = context.get('turn', 1)
        
        # Calculate competitive analysis for the prompt
        avg_competitor = sum(competitor_prices) / len(competitor_prices) if competitor_prices else 10.0
        min_competitor = min(competitor_prices) if competitor_prices else 10.0
        max_competitor = max(competitor_prices) if competitor_prices else 10.0
        
        # Create specialized pricing prompt
        prompt = f"""
SPECIALIZED PRICING AGENT DECISION

You are an expert PRICING SPECIALIST making strategic pricing decisions for a brewery.
Framework Context: {framework_context}

CURRENT MARKET ANALYSIS:
- Your Current Price: ${current_price:.2f}
- Competitor Prices: {competitor_prices}
- Average Competitor Price: ${avg_competitor:.2f}
- Lowest Competitor: ${min_competitor:.2f}
- Highest Competitor: ${max_competitor:.2f}
- Your Inventory Level: {inventory} units
- Current Turn: {turn}

PRICING STRATEGY CONSIDERATIONS:
1. COMPETITIVE POSITIONING:
   - Are you priced above or below competitors?
   - What's your competitive advantage?
   
2. INVENTORY MANAGEMENT:
   - High inventory ({inventory} > 120): Consider lower prices to move stock
   - Low inventory ({inventory} < 50): Can charge premium prices
   - Normal inventory (50-120): Optimize for profit
   
3. PROFIT OPTIMIZATION:
   - Minimum viable price: ${AgentConfig.UNIT_PRODUCTION_COST / (1 - AgentConfig.TARGET_PROFIT_MARGIN):.2f}
   - Target profit margin: {AgentConfig.TARGET_PROFIT_MARGIN * 100}%
   
4. MARKET DYNAMICS:
   - Price elasticity: Lower prices increase demand
   - Brand positioning: Premium vs value strategy

PRICING CONSTRAINTS:
- Absolute minimum price: ${AgentConfig.MIN_PRICE:.2f}
- Absolute maximum price: ${AgentConfig.MAX_PRICE:.2f}
- Must remain profitable above production costs

YOUR TASK:
Analyze the market situation and determine the optimal price that:
- Maximizes profitability
- Manages inventory effectively  
- Maintains competitive position
- Supports overall business strategy

RESPOND IN EXACT JSON FORMAT:
{{
    "price": 10.50,
    "reasoning": "Detailed explanation of pricing strategy and rationale",
    "confidence": 0.85,
    "competitive_strategy": "premium/competitive/discount",
    "inventory_impact": "how this price helps with inventory management",
    "profit_projection": "expected profit margin at this price"
}}
"""
        return prompt
    
    def _pricing_fallback_decision(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback decision specifically for pricing
        
        Args:
            context: Market context
            
        Returns:
            Rule-based pricing decision
        """
        # Extract context for decision logic
        inventory = context.get('inventory', 100)
        competitor_prices = context.get('competitor_prices', [10.0])
        current_price = context.get('current_price', 10.0)
        
        # Calculate key metrics
        avg_competitor = sum(competitor_prices) / len(competitor_prices) if competitor_prices else 10.0
        inventory_ratio = inventory / 100.0  # Normalize inventory level
        
        # Rule-based pricing logic
        if inventory_ratio > 1.8:
            # High inventory - aggressive pricing to clear stock
            price = max(AgentConfig.MIN_PRICE, avg_competitor - random.uniform(1.0, 1.5))
            strategy = "discount"
            reasoning = f"High inventory ({inventory} units) - aggressive discount to clear stock"
            
        elif inventory_ratio > 1.2:
            # Moderate inventory - competitive pricing
            price = max(AgentConfig.MIN_PRICE, avg_competitor - random.uniform(0.3, 0.7))
            strategy = "competitive"
            reasoning = f"Moderate inventory ({inventory} units) - competitive pricing to maintain flow"
            
        elif inventory_ratio < 0.4:
            # Very low inventory - premium pricing
            price = min(AgentConfig.MAX_PRICE, avg_competitor + random.uniform(1.2, 2.0))
            strategy = "premium"
            reasoning = f"Low inventory ({inventory} units) - premium pricing due to scarcity"
            
        elif inventory_ratio < 0.8:
            # Low inventory - moderate premium
            price = min(AgentConfig.MAX_PRICE, avg_competitor + random.uniform(0.5, 1.0))
            strategy = "premium"
            reasoning = f"Below-normal inventory ({inventory} units) - moderate premium pricing"
            
        else:
            # Normal inventory - market-based pricing with small adjustments
            price = avg_competitor + random.uniform(-0.3, 0.3)
            strategy = "competitive"
            reasoning = f"Normal inventory ({inventory} units) - market-competitive pricing"
        
        # Ensure price is within bounds
        price = max(AgentConfig.MIN_PRICE, min(AgentConfig.MAX_PRICE, price))
        
        # Calculate expected profit margin
        expected_margin = (price - AgentConfig.UNIT_PRODUCTION_COST) / price if price > 0 else 0
        
        return {
            'price': round(price, 2),
            'reasoning': reasoning,
            'confidence': 0.6,
            'competitive_strategy': strategy,
            'inventory_impact': f"Addresses {inventory} unit inventory level",
            'profit_projection': f"Expected margin: {expected_margin*100:.1f}%",
            'agent_type': 'pricing',
            'ai_enhanced': False,
            'source': 'rule_based_fallback'
        }
    
    def _add_pricing_analysis(self, decision: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add pricing-specific analysis and metadata to the decision
        
        Args:
            decision: The base decision to enhance
            context: Market context for analysis
            
        Returns:
            Enhanced decision with pricing analysis
        """
        # Calculate price change from current price
        current_price = context.get('current_price', 10.0)
        new_price = decision.get('price', current_price)
        price_change = new_price - current_price
        
        # Analyze price trend if we have history
        price_trend = "stable"
        if len(self.price_history) >= 3:
            recent_prices = self.price_history[-3:]
            if recent_prices[-1] > recent_prices[0] + 0.5:
                price_trend = "increasing"
            elif recent_prices[-1] < recent_prices[0] - 0.5:
                price_trend = "decreasing"
        
        # Add pricing analysis to the decision
        decision['pricing_analysis'] = {
            'price_change': round(price_change, 2),
            'price_change_direction': 'increase' if price_change > 0 else 'decrease' if price_change < 0 else 'no_change',
            'price_trend': price_trend,
            'competitive_position': self._analyze_competitive_position(new_price, context),
            'margin_estimate': self._calculate_margin_estimate(new_price)
        }
        
        return decision
    
    def _analyze_competitive_position(self, price: float, context: Dict[str, Any]) -> str:
        """
        Analyze where our price stands relative to competitors
        
        Args:
            price: Our proposed price
            context: Market context with competitor information
            
        Returns:
            String describing competitive position
        """
        competitor_prices = context.get('competitor_prices', [10.0])
        
        if not competitor_prices:
            return "no_competition_data"
        
        # Calculate percentile position
        sorted_prices = sorted(competitor_prices + [price])
        our_position = sorted_prices.index(price)
        percentile = our_position / len(sorted_prices)
        
        # Classify competitive position
        if percentile < 0.25:
            return "lowest_price"      # We're in the bottom 25% (cheapest)
        elif percentile < 0.5:
            return "below_average"     # Below median price
        elif percentile < 0.75:
            return "above_average"     # Above median price
        else:
            return "highest_price"     # We're in the top 25% (most expensive)
    
    def _calculate_margin_estimate(self, price: float) -> Dict[str, float]:
        """
        Calculate estimated profit margin for a given price
        
        Args:
            price: The price to analyze
            
        Returns:
            Dictionary with margin calculations
        """
        # Calculate gross margin (price minus production cost)
        gross_margin = price - AgentConfig.UNIT_PRODUCTION_COST
        
        # Calculate margin percentage
        margin_percentage = (gross_margin / price) * 100 if price > 0 else 0
        
        # Check if margin meets target
        meets_target = margin_percentage >= (AgentConfig.TARGET_PROFIT_MARGIN * 100)
        
        return {
            'gross_margin_per_unit': round(gross_margin, 2),
            'margin_percentage': round(margin_percentage, 1),
            'meets_target_margin': meets_target,
            'target_margin': AgentConfig.TARGET_PROFIT_MARGIN * 100
        }
    
    def get_pricing_insights(self) -> Dict[str, Any]:
        """
        Get detailed insights about this agent's pricing performance
        
        Returns:
            Dictionary with pricing-specific insights and statistics
        """
        # Get base agent stats
        base_stats = self.get_agent_stats()
        
        # Add pricing-specific insights
        pricing_insights = {
            'price_range_used': {
                'min_price_set': min(self.price_history) if self.price_history else 0,
                'max_price_set': max(self.price_history) if self.price_history else 0,
                'avg_price_set': sum(self.price_history) / len(self.price_history) if self.price_history else 0
            },
            'pricing_volatility': self._calculate_pricing_volatility(),
            'competitive_positioning': self._analyze_pricing_patterns(),
            'margin_performance': self._analyze_margin_performance()
        }
        
        # Combine base stats with pricing insights
        base_stats['pricing_insights'] = pricing_insights
        return base_stats
    
    def _calculate_pricing_volatility(self) -> float:
        """
        Calculate how much this agent's pricing varies over time
        
        Returns:
            Standard deviation of price history (measure of volatility)
        """
        if len(self.price_history) < 2:
            return 0.0  # Can't calculate volatility with less than 2 data points
        
        # Calculate mean price
        mean_price = sum(self.price_history) / len(self.price_history)
        
        # Calculate variance
        variance = sum((price - mean_price) ** 2 for price in self.price_history) / len(self.price_history)
        
        # Return standard deviation (square root of variance)
        return variance ** 0.5
    
    def _analyze_pricing_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in pricing decisions over time
        
        Returns:
            Dictionary with pricing pattern analysis
        """
        if len(self.price_history) < 3:
            return {'status': 'insufficient_data'}
        
        # Count pricing strategies used
        strategy_counts = {}
        for decision in self.decision_history:
            strategy = decision.get('competitive_strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Find most common strategy
        most_common_strategy = max(strategy_counts.items(), key=lambda x: x[1])[0] if strategy_counts else 'unknown'
        
        # Calculate price trend
        recent_trend = self._calculate_price_trend()
        
        return {
            'strategy_distribution': strategy_counts,
            'most_common_strategy': most_common_strategy,
            'price_trend': recent_trend,
            'decisions_analyzed': len(self.decision_history)
        }
    
    def _calculate_price_trend(self) -> str:
        """
        Calculate the overall trend in pricing over recent decisions
        
        Returns:
            String describing the pricing trend
        """
        if len(self.price_history) < 5:
            return "insufficient_data"
        
        # Compare recent prices to earlier prices
        recent_avg = sum(self.price_history[-3:]) / 3  # Last 3 prices
        earlier_avg = sum(self.price_history[-6:-3]) / 3  # 3 prices before that
        
        # Determine trend direction
        if recent_avg > earlier_avg + 0.2:
            return "increasing"
        elif recent_avg < earlier_avg - 0.2:
            return "decreasing"
        else:
            return "stable"
    
    def _analyze_margin_performance(self) -> Dict[str, Any]:
        """
        Analyze how well this agent's pricing achieves target margins
        
        Returns:
            Dictionary with margin performance analysis
        """
        if not self.decision_history:
            return {'status': 'no_decisions_to_analyze'}
        
        # Count decisions that met target margin
        margin_successes = 0
        total_margin = 0
        
        for decision in self.decision_history:
            # Check if this decision had margin analysis
            if 'pricing_analysis' in decision and 'margin_estimate' in decision['pricing_analysis']:
                margin_data = decision['pricing_analysis']['margin_estimate']
                
                # Count if it met target
                if margin_data.get('meets_target_margin', False):
                    margin_successes += 1
                
                # Add to total for average calculation
                total_margin += margin_data.get('margin_percentage', 0)
        
        # Calculate success rate and average margin
        decisions_with_margin = len([d for d in self.decision_history if 'pricing_analysis' in d])
        success_rate = margin_successes / decisions_with_margin if decisions_with_margin > 0 else 0
        avg_margin = total_margin / decisions_with_margin if decisions_with_margin > 0 else 0
        
        return {
            'margin_success_rate': round(success_rate * 100, 1),  # Percentage of decisions meeting target
            'average_margin_achieved': round(avg_margin, 1),       # Average margin percentage
            'target_margin': AgentConfig.TARGET_PROFIT_MARGIN * 100, # Target margin for reference
            'decisions_analyzed': decisions_with_margin
        }

# Example usage and testing
if __name__ == "__main__":
    """
    Test the pricing agent independently
    This section only runs when this file is executed directly
    """
    import asyncio
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    async def test_pricing_agent():
        """Test function to verify pricing agent works correctly"""
        print("Testing Pricing Agent...")
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Initialize AI helper and pricing agent
        ai_helper = GeminiAIHelper(api_key)
        pricing_agent = PricingAgent(ai_helper)
        
        # Create test context
        test_context = {
            'inventory': 85,                    # Current inventory level
            'current_price': 10.50,             # Current price setting
            'competitor_prices': [9.20, 11.80, 10.00],  # What competitors are charging
            'turn': 5,                          # Current game turn
            'sales_history': [45, 52, 48, 55]   # Historical sales data
        }
        
        # Test the pricing decision
        print("Making pricing decision...")
        decision = await pricing_agent.make_decision(test_context, "TEST_FRAMEWORK")
        
        # Display results
        print(f"Pricing Decision: ${decision['price']:.2f}")
        print(f"Strategy: {decision.get('competitive_strategy', 'N/A')}")
        print(f"Reasoning: {decision['reasoning'][:100]}...")
        print(f"Confidence: {decision['confidence']:.2f}")
        print(f"AI Enhanced: {decision['ai_enhanced']}")
        
        # Test multiple decisions to see agent performance
        print("\nTesting multiple decisions...")
        for i in range(3):
            # Modify context for each test
            test_context['inventory'] = random.randint(30, 150)
            test_context['turn'] = 6 + i
            
            await pricing_agent.make_decision(test_context, f"TEST_ROUND_{i+1}")
        
        # Show agent insights
        insights = pricing_agent.get_pricing_insights()
        print(f"\nPricing Agent Insights:")
        print(f"Total decisions: {insights['decisions_made']}")
        print(f"Average confidence: {insights['average_confidence']:.2f}")
        print(f"Price range used: ${insights['pricing_insights']['price_range_used']['min_price_set']:.2f} - ${insights['pricing_insights']['price_range_used']['max_price_set']:.2f}")
    
    # Run the test
    asyncio.run(test_pricing_agent())