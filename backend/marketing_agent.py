# marketing_agent.py - Specialized agent for marketing campaign decisions

import random  # For generating random values in fallback decisions
from typing import Dict, Any, List  # Type hints for function parameters

# Import base classes and helpers
from base_agent import BaseAgent, AgentConfig  # Parent class and configuration
from gemini_helper import GeminiAIHelper      # AI integration helper

class MarketingAgent(BaseAgent):
    """
    Specialized agent responsible for marketing campaign optimization
    This agent determines optimal marketing spend to maximize demand and ROI
    """
    
    def __init__(self, ai_helper: GeminiAIHelper):
        """
        Initialize the marketing agent with AI capabilities
        
        Args:
            ai_helper: Shared AI helper for intelligent decision making
        """
        # Initialize parent class with agent type "marketing"
        super().__init__("marketing")
        
        # Store reference to AI helper
        self.ai_helper = ai_helper
        
        # Marketing-specific tracking variables
        self.marketing_history = []       # Track all marketing spends
        self.roi_performance = []        # Track return on investment performance
        self.campaign_effectiveness = [] # Track campaign effectiveness metrics
        
        print("Marketing Agent initialized and ready for campaign optimization")
    
    async def make_decision(self, context: Dict[str, Any], framework_context: str = "") -> Dict[str, Any]:
        """
        Make a marketing spend decision based on market conditions and inventory
        
        Args:
            context: Market context including inventory, pricing, competition
            framework_context: Which framework is requesting this decision
            
        Returns:
            Dictionary containing marketing decision and campaign strategy
        """
        print(f"Marketing Agent optimizing campaign for {framework_context}")
        
        # Store current context for analysis
        self.current_context = context
        
        # Perform marketing-specific analysis first
        marketing_analysis = self._analyze_marketing_opportunity(context)
        
        # Add marketing analysis to context for AI decision
        enhanced_context = context.copy()
        enhanced_context['marketing_analysis'] = marketing_analysis
        
        # Get AI-enhanced decision
        decision = await self.ai_helper.get_ai_decision(
            "marketing", enhanced_context, framework_context
        )
        
        # If AI decision lacks marketing data, create specialized decision
        if 'marketing' not in decision:
            decision = await self._make_marketing_specific_decision(context, framework_context)
        
        # Validate marketing spend is within bounds
        if 'marketing' in decision:
            decision['marketing'] = int(self.validate_decision_bounds(
                "marketing", decision['marketing']
            ))
        
        # Add marketing-specific analysis and ROI projections
        decision = self._add_marketing_analysis(decision, context, marketing_analysis)
        
        # Store decision in history
        self.add_decision_to_history(decision)
        
        # Track marketing spend for trend analysis
        self.marketing_history.append(decision['marketing'])
        if len(self.marketing_history) > 50:  # Limit history size
            self.marketing_history = self.marketing_history[-50:]
        
        print(f"Marketing Agent decision: ${decision['marketing']} campaign spend")
        return decision
    
    def _analyze_marketing_opportunity(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze current marketing opportunity and needs
        
        Args:
            context: Market context for opportunity analysis
            
        Returns:
            Dictionary containing marketing opportunity analysis
        """
        # Extract key context variables
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitor_prices = context.get('competitor_prices', [10.0])
        sales_history = context.get('sales_history', [])
        
        # Calculate inventory pressure (how urgently we need to move inventory)
        inventory_pressure = self._calculate_inventory_pressure(inventory)
        
        # Calculate competitive pressure (how marketing can help vs competitors)
        competitive_pressure = self._calculate_competitive_pressure(current_price, competitor_prices)
        
        # Calculate expected marketing ROI based on current conditions
        expected_roi = self._calculate_expected_roi(context)
        
        # Determine marketing urgency level
        marketing_urgency = self._assess_marketing_urgency(inventory_pressure, competitive_pressure)
        
        return {
            'inventory_pressure': inventory_pressure,
            'competitive_pressure': competitive_pressure,
            'expected_roi': expected_roi,
            'marketing_urgency': marketing_urgency,
            'recommended_budget_range': self._calculate_budget_range(inventory_pressure, expected_roi)
        }
    
    def _calculate_inventory_pressure(self, inventory: int) -> Dict[str, Any]:
        """
        Calculate how much marketing pressure is needed based on inventory
        
        Args:
            inventory: Current inventory level
            
        Returns:
            Dictionary describing inventory pressure
        """
        # Normalize inventory against baseline of 100 units
        inventory_ratio = inventory / 100.0
        
        if inventory_ratio > 2.0:
            pressure_level = "critical"
            multiplier = 2.5
            description = "Critical excess inventory requires aggressive marketing"
        elif inventory_ratio > 1.5:
            pressure_level = "high"
            multiplier = 1.8
            description = "High inventory needs increased marketing push"
        elif inventory_ratio > 1.2:
            pressure_level = "moderate"
            multiplier = 1.2
            description = "Moderate inventory allows normal marketing levels"
        elif inventory_ratio < 0.3:
            pressure_level = "low"
            multiplier = 0.2
            description = "Low inventory requires minimal marketing (scarcity drives demand)"
        elif inventory_ratio < 0.6:
            pressure_level = "reduced"
            multiplier = 0.6
            description = "Below-normal inventory suggests reduced marketing spend"
        else:
            pressure_level = "normal"
            multiplier = 1.0
            description = "Normal inventory supports standard marketing levels"
        
        return {
            'level': pressure_level,
            'multiplier': multiplier,
            'description': description,
            'inventory_ratio': round(inventory_ratio, 2)
        }
    
    def _calculate_competitive_pressure(self, our_price: float, competitor_prices: List[float]) -> Dict[str, Any]:
        """
        Calculate competitive pressure and marketing opportunity
        
        Args:
            our_price: Our current price
            competitor_prices: List of competitor prices
            
        Returns:
            Dictionary describing competitive pressure
        """
        if not competitor_prices:
            return {
                'level': 'unknown',
                'multiplier': 1.0,
                'description': 'No competitor data available'
            }
        
        # Calculate our position relative to competitors
        avg_competitor = sum(competitor_prices) / len(competitor_prices)
        price_difference = our_price - avg_competitor
        
        if price_difference > 1.5:
            # We're significantly more expensive - need marketing to justify premium
            pressure_level = "high"
            multiplier = 1.6
            description = f"Premium pricing (+${price_difference:.2f}) requires marketing to justify value"
        elif price_difference > 0.5:
            # We're moderately more expensive
            pressure_level = "moderate"
            multiplier = 1.3
            description = f"Above-market pricing (+${price_difference:.2f}) needs marketing support"
        elif price_difference < -1.0:
            # We're significantly cheaper - marketing can amplify advantage
            pressure_level = "opportunity"
            multiplier = 1.4
            description = f"Competitive pricing (-${abs(price_difference):.2f}) creates marketing opportunity"
        elif price_difference < -0.3:
            # We're moderately cheaper
            pressure_level = "low"
            multiplier = 0.9
            description = f"Good pricing (-${abs(price_difference):.2f}) reduces marketing pressure"
        else:
            # We're competitively priced
            pressure_level = "normal"
            multiplier = 1.0
            description = f"Competitive pricing (${price_difference:+.2f}) supports standard marketing"
        
        return {
            'level': pressure_level,
            'multiplier': multiplier,
            'description': description,
            'price_difference': round(price_difference, 2),
            'competitive_position': 'premium' if price_difference > 0 else 'discount' if price_difference < 0 else 'competitive'
        }
    
    def _calculate_expected_roi(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate expected return on investment for marketing spend
        
        Args:
            context: Market context for ROI calculation
            
        Returns:
            Dictionary with ROI analysis
        """
        # Base ROI: $500 marketing typically generates +15 units demand
        base_demand_per_500 = 15
        current_price = context.get('current_price', 10.0)
        
        # Revenue per additional unit sold
        revenue_per_unit = current_price
        
        # Calculate ROI for different spend levels
        roi_scenarios = {}
        for spend in [500, 1000, 1500, 2000]:
            # Diminishing returns: each additional $500 is less effective
            effectiveness_factor = 1.0 - (spend - 500) / 5000  # Reduces effectiveness at higher spends
            effectiveness_factor = max(0.3, effectiveness_factor)  # Minimum 30% effectiveness
            
            expected_additional_demand = (spend / 500) * base_demand_per_500 * effectiveness_factor
            expected_additional_revenue = expected_additional_demand * revenue_per_unit
            roi_ratio = expected_additional_revenue / spend if spend > 0 else 0
            
            roi_scenarios[spend] = {
                'additional_demand': round(expected_additional_demand, 1),
                'additional_revenue': round(expected_additional_revenue, 2),
                'roi_ratio': round(roi_ratio, 2),
                'roi_percentage': round((roi_ratio - 1) * 100, 1)  # ROI as percentage
            }
        
        # Find optimal spend level (highest ROI)
        best_spend = max(roi_scenarios.items(), key=lambda x: x[1]['roi_ratio'])[0]
        
        return {
            'roi_scenarios': roi_scenarios,
            'optimal_spend_level': best_spend,
            'base_effectiveness': f"${500} = +{base_demand_per_500} units demand",
            'current_price_factor': current_price
        }
    
    def _assess_marketing_urgency(self, inventory_pressure: Dict[str, Any], 
                                competitive_pressure: Dict[str, Any]) -> str:
        """
        Assess overall marketing urgency combining inventory and competitive factors
        
        Args:
            inventory_pressure: Analysis of inventory-driven marketing needs
            competitive_pressure: Analysis of competition-driven marketing needs
            
        Returns:
            String describing overall marketing urgency
        """
        # Combine pressure multipliers
        total_multiplier = inventory_pressure['multiplier'] * competitive_pressure['multiplier']
        
        # Classify urgency based on combined pressure
        if total_multiplier > 2.5:
            return "critical"      # Very high marketing spend needed
        elif total_multiplier > 1.8:
            return "high"         # High marketing spend recommended
        elif total_multiplier > 1.3:
            return "moderate"     # Normal to increased marketing
        elif total_multiplier > 0.8:
            return "normal"       # Standard marketing levels
        else:
            return "low"          # Reduced marketing sufficient
    
    def _calculate_budget_range(self, inventory_pressure: Dict[str, Any], 
                              expected_roi: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate recommended marketing budget range
        
        Args:
            inventory_pressure: Inventory pressure analysis
            expected_roi: ROI analysis for different spend levels
            
        Returns:
            Dictionary with minimum and maximum recommended spend
        """
        # Base budget calculation
        base_budget = 800  # Standard marketing spend
        
        # Adjust based on inventory pressure
        inventory_adjustment = base_budget * (inventory_pressure['multiplier'] - 1)
        adjusted_budget = base_budget + inventory_adjustment
        
        # Find optimal spend from ROI analysis
        optimal_spend = expected_roi['optimal_spend_level']
        
        # Calculate recommended range
        min_spend = max(AgentConfig.MIN_MARKETING, int(adjusted_budget * 0.7))
        max_spend = min(AgentConfig.MAX_MARKETING, int(max(adjusted_budget * 1.3, optimal_spend)))
        
        return {
            'min_recommended': min_spend,
            'max_recommended': max_spend,
            'optimal_spend': optimal_spend,
            'base_calculation': int(adjusted_budget)
        }
    
    async def _make_marketing_specific_decision(self, context: Dict[str, Any], 
                                             framework_context: str) -> Dict[str, Any]:
        """
        Create marketing-specific decision with detailed campaign planning
        
        Args:
            context: Market context
            framework_context: Framework calling this agent
            
        Returns:
            Marketing decision with campaign details
        """
        # Get marketing analysis
        marketing_analysis = context.get('marketing_analysis', {})
        
        # Create specialized marketing prompt
        marketing_prompt = self._create_marketing_prompt(context, framework_context, marketing_analysis)
        
        # Try AI with specialized prompt
        if self.ai_helper.enabled:
            try:
                import asyncio
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.ai_helper.model.generate_content(marketing_prompt)
                )
                
                if response and response.text:
                    return self.ai_helper._parse_ai_response(response.text, "marketing")
                else:
                    raise ValueError("Empty AI response")
                    
            except Exception as e:
                print(f"Specialized marketing AI failed: {e}")
        
        # Fallback to rule-based marketing decision
        return self._marketing_fallback_decision(context, marketing_analysis)
    
    def _create_marketing_prompt(self, context: Dict[str, Any], framework_context: str,
                               marketing_analysis: Dict[str, Any]) -> str:
        """
        Create detailed prompt for marketing campaign optimization
        
        Args:
            context: Market context
            framework_context: Framework information
            marketing_analysis: Marketing opportunity analysis
            
        Returns:
            Formatted prompt for marketing decisions
        """
        # Extract key variables
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitor_prices = context.get('competitor_prices', [10.0])
        
        # Get analysis results
        inventory_pressure = marketing_analysis.get('inventory_pressure', {})
        competitive_pressure = marketing_analysis.get('competitive_pressure', {})
        expected_roi = marketing_analysis.get('expected_roi', {})
        urgency = marketing_analysis.get('marketing_urgency', 'normal')
        
        prompt = f"""
SPECIALIZED MARKETING CAMPAIGN OPTIMIZATION AGENT

You are an expert MARKETING STRATEGIST optimizing campaign spend for maximum impact.
Framework Context: {framework_context}

CURRENT MARKET SITUATION:
- Inventory Level: {inventory} units
- Current Price: ${current_price:.2f}
- Competitor Prices: {competitor_prices}
- Marketing Urgency: {urgency}

MARKETING ANALYSIS:
- Inventory Pressure: {inventory_pressure.get('level', 'normal')} ({inventory_pressure.get('description', 'N/A')})
- Competitive Pressure: {competitive_pressure.get('level', 'normal')} ({competitive_pressure.get('description', 'N/A')})
- Optimal Spend Level: ${expected_roi.get('optimal_spend_level', 800)} (based on ROI analysis)

MARKETING EFFECTIVENESS RULES:
- Base effectiveness: $500 marketing = +15 units demand
- Diminishing returns: Higher spends are less efficient per dollar
- Price synergy: Lower prices amplify marketing effectiveness
- Competition factor: Marketing helps differentiate from competitors

BUDGET CONSTRAINTS:
- Minimum spend: ${AgentConfig.MIN_MARKETING}
- Maximum spend: ${AgentConfig.MAX_MARKETING}
- Recommended range: ${marketing_analysis.get('recommended_budget_range', {}).get('min_recommended', 400)}-${marketing_analysis.get('recommended_budget_range', {}).get('max_recommended', 1200)}

CAMPAIGN STRATEGY OPTIONS:

1. AGGRESSIVE CAMPAIGN (High Spend):
   - Use when: High inventory, competitive pricing
   - Budget: $1,500-$2,000
   - Goal: Maximum market penetration

2. TARGETED CAMPAIGN (Medium Spend):
   - Use when: Normal inventory, balanced strategy
   - Budget: $600-$1,200
   - Goal: Efficient demand generation

3. MINIMAL CAMPAIGN (Low Spend):
   - Use when: Low inventory, premium pricing
   - Budget: $100-$500
   - Goal: Cost-efficient brand maintenance

YOUR TASK:
Determine optimal marketing spend that:
- Maximizes return on investment
- Supports inventory movement goals
- Responds to competitive pressure
- Aligns with pricing strategy

RESPOND IN EXACT JSON FORMAT:
{{
    "marketing": 850,
    "reasoning": "Detailed marketing strategy and campaign rationale",
    "confidence": 0.85,
    "campaign_type": "aggressive/targeted/minimal",
    "expected_demand_boost": 25,
    "roi_projection": "expected return on marketing investment",
    "target_audience": "description of target market segment"
}}
"""
        return prompt
    
    def _marketing_fallback_decision(self, context: Dict[str, Any], 
                                   marketing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based fallback for marketing decisions
        
        Args:
            context: Market context
            marketing_analysis: Marketing opportunity analysis
            
        Returns:
            Rule-based marketing decision
        """
        # Extract key factors
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitor_prices = context.get('competitor_prices', [10.0])
        
        # Get pressure analysis
        inventory_pressure = marketing_analysis.get('inventory_pressure', {})
        competitive_pressure = marketing_analysis.get('competitive_pressure', {})
        urgency = marketing_analysis.get('marketing_urgency', 'normal')
        
        # Rule-based marketing spend logic
        if urgency == "critical":
            # Critical situation - maximum marketing effort
            marketing_spend = random.randint(1600, 2000)
            campaign_type = "aggressive"
            reasoning = f"Critical marketing push needed - {inventory_pressure.get('description', 'high pressure situation')}"
            
        elif urgency == "high":
            # High urgency - increased marketing
            marketing_spend = random.randint(1200, 1600)
            campaign_type = "aggressive"
            reasoning = f"High marketing investment - {inventory_pressure.get('description', 'elevated pressure')}"
            
        elif urgency == "moderate":
            # Moderate urgency - balanced marketing
            marketing_spend = random.randint(800, 1200)
            campaign_type = "targeted"
            reasoning = f"Balanced marketing approach - {competitive_pressure.get('description', 'moderate conditions')}"
            
        elif urgency == "low":
            # Low urgency - minimal marketing
            marketing_spend = random.randint(100, 400)
            campaign_type = "minimal"
            reasoning = f"Reduced marketing spend - {inventory_pressure.get('description', 'low pressure situation')}"
            
        else:
            # Normal conditions - standard marketing
            marketing_spend = random.randint(600, 1000)
            campaign_type = "targeted"
            reasoning = "Standard marketing campaign for normal market conditions"
        
        # Calculate expected impact
        expected_boost = (marketing_spend / 500) * 15  # Base formula: $500 = +15 demand
        
        # Estimate ROI
        expected_revenue = expected_boost * current_price
        roi_ratio = expected_revenue / marketing_spend if marketing_spend > 0 else 0
        
        return {
            'marketing': marketing_spend,
            'reasoning': reasoning,
            'confidence': 0.65,
            'campaign_type': campaign_type,
            'expected_demand_boost': round(expected_boost, 1),
            'roi_projection': f"Expected ROI: {roi_ratio:.1f}x (${expected_revenue:.0f} revenue for ${marketing_spend} spend)",
            'target_audience': self._determine_target_audience(campaign_type, competitive_pressure),
            'agent_type': 'marketing',
            'ai_enhanced': False,
            'source': 'rule_based_fallback'
        }
    
    def _determine_target_audience(self, campaign_type: str, competitive_pressure: Dict[str, Any]) -> str:
        """
        Determine target audience based on campaign type and market position
        
        Args:
            campaign_type: Type of campaign (aggressive/targeted/minimal)
            competitive_pressure: Competitive analysis
            
        Returns:
            String describing target audience
        """
        competitive_position = competitive_pressure.get('competitive_position', 'competitive')
        
        if campaign_type == "aggressive":
            if competitive_position == 'premium':
                return "Quality-focused customers willing to pay premium prices"
            elif competitive_position == 'discount':
                return "Price-sensitive customers seeking value deals"
            else:
                return "Broad market with mass appeal messaging"
                
        elif campaign_type == "targeted":
            return "Core customer segments with high conversion potential"
            
        else:  # minimal
            return "Existing customers and high-intent prospects only"
    
    def _add_marketing_analysis(self, decision: Dict[str, Any], context: Dict[str, Any],
                              marketing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add marketing-specific analysis and projections to the decision
        
        Args:
            decision: Base decision to enhance
            context: Market context
            marketing_analysis: Marketing opportunity analysis
            
        Returns:
            Enhanced decision with marketing analysis
        """
        marketing_spend = decision.get('marketing', 500)
        current_price = context.get('current_price', 10.0)
        
        # Calculate detailed marketing projections
        expected_demand_boost = (marketing_spend / 500) * 15
        expected_revenue = expected_demand_boost * current_price
        roi_ratio = expected_revenue / marketing_spend if marketing_spend > 0 else 0
        
        # Calculate cost efficiency
        cost_per_additional_customer = marketing_spend / expected_demand_boost if expected_demand_boost > 0 else float('inf')
        
        # Add comprehensive marketing analysis
        decision['marketing_analysis'] = {
            'campaign_budget': marketing_spend,
            'expected_impact': {
                'demand_boost': round(expected_demand_boost, 1),
                'revenue_boost': round(expected_revenue, 2),
                'roi_ratio': round(roi_ratio, 2),
                'roi_percentage': round((roi_ratio - 1) * 100, 1)
            },
            'efficiency_metrics': {
                'cost_per_customer': round(cost_per_additional_customer, 2),
                'revenue_per_dollar_spent': round(expected_revenue / marketing_spend if marketing_spend > 0 else 0, 2)
            },
            'market_positioning': marketing_analysis.get('competitive_pressure', {}).get('competitive_position', 'competitive'),
            'campaign_urgency': marketing_analysis.get('marketing_urgency', 'normal')
        }
        
        return decision
    
    def get_marketing_insights(self) -> Dict[str, Any]:
        """
        Get detailed insights about marketing agent performance and effectiveness
        
        Returns:
            Dictionary containing marketing-specific insights
        """
        # Get base agent statistics
        base_stats = self.get_agent_stats()
        
        # Calculate marketing-specific insights
        marketing_insights = {
            'spend_patterns': {
                'min_spend': min(self.marketing_history) if self.marketing_history else 0,
                'max_spend': max(self.marketing_history) if self.marketing_history else 0,
                'avg_spend': sum(self.marketing_history) / len(self.marketing_history) if self.marketing_history else 0,
                'spend_volatility': self._calculate_spend_volatility()
            },
            'campaign_effectiveness': self._analyze_campaign_patterns(),
            'roi_performance': self._analyze_roi_trends()
        }
        
        # Combine with base stats
        base_stats['marketing_insights'] = marketing_insights
        return base_stats
    
    def _calculate_spend_volatility(self) -> float:
        """
        Calculate variability in marketing spend over time
        
        Returns:
            Standard deviation of marketing spend history
        """
        if len(self.marketing_history) < 2:
            return 0.0
        
        mean_spend = sum(self.marketing_history) / len(self.marketing_history)
        variance = sum((spend - mean_spend) ** 2 for spend in self.marketing_history) / len(self.marketing_history)
        return variance ** 0.5
    
    def _analyze_campaign_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in campaign types and strategies used
        
        Returns:
            Dictionary with campaign pattern analysis
        """
        if not self.decision_history:
            return {'status': 'no_campaigns_to_analyze'}
        
        # Count different campaign types used
        campaign_types = {}
        target_audiences = {}
        
        for decision in self.decision_history:
            # Count campaign types
            campaign_type = decision.get('campaign_type', 'unknown')
            campaign_types[campaign_type] = campaign_types.get(campaign_type, 0) + 1
            
            # Count target audiences
            audience = decision.get('target_audience', 'unknown')
            target_audiences[audience] = target_audiences.get(audience, 0) + 1
        
        # Find most common approaches
        most_common_type = max(campaign_types.items(), key=lambda x: x[1])[0] if campaign_types else 'unknown'
        most_common_audience = max(target_audiences.items(), key=lambda x: x[1])[0] if target_audiences else 'unknown'
        
        return {
            'campaign_type_distribution': campaign_types,
            'target_audience_distribution': target_audiences,
            'most_common_campaign_type': most_common_type,
            'most_common_target_audience': most_common_audience,
            'campaigns_analyzed': len(self.decision_history)
        }
    
    def _analyze_roi_trends(self) -> Dict[str, Any]:
        """
        Analyze return on investment trends for marketing campaigns
        
        Returns:
            Dictionary with ROI trend analysis
        """
        if not self.decision_history:
            return {'status': 'no_roi_data'}
        
        # Extract ROI projections from decisions
        roi_projections = []
        for decision in self.decision_history:
            if 'marketing_analysis' in decision:
                roi_data = decision['marketing_analysis'].get('expected_impact', {})
                roi_ratio = roi_data.get('roi_ratio', 0)
                if roi_ratio > 0:
                    roi_projections.append(roi_ratio)
        
        if not roi_projections:
            return {'status': 'no_roi_projections'}
        
        # Calculate ROI statistics
        avg_roi = sum(roi_projections) / len(roi_projections)
        min_roi = min(roi_projections)
        max_roi = max(roi_projections)
        
        # Classify ROI performance
        if avg_roi > 2.0:
            performance = "excellent"
        elif avg_roi > 1.5:
            performance = "good"
        elif avg_roi > 1.0:
            performance = "profitable"
        else:
            performance = "poor"
        
        return {
            'average_roi': round(avg_roi, 2),
            'roi_range': [round(min_roi, 2), round(max_roi, 2)],
            'roi_performance': performance,
            'campaigns_analyzed': len(roi_projections)
        }

# Example usage and testing
if __name__ == "__main__":
    """
    Test the marketing agent independently
    """
    import asyncio
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test_marketing_agent():
        """Test marketing agent with different scenarios"""
        print("Testing Marketing Agent...")
        
        # Initialize marketing agent
        api_key = os.getenv("GEMINI_API_KEY")
        ai_helper = GeminiAIHelper(api_key)
        marketing_agent = MarketingAgent(ai_helper)
        
        # Test scenarios
        scenarios = [
            {
                'name': 'High Inventory - Need Marketing Push',
                'context': {
                    'inventory': 200,
                    'current_price': 9.50,
                    'competitor_prices': [10.00, 11.00],
                    'turn': 5,
                    'sales_history': [35, 40, 38]
                }
            },
            {
                'name': 'Low Inventory - Minimal Marketing', 
                'context': {
                    'inventory': 30,
                    'current_price': 12.00,
                    'competitor_prices': [10.50, 11.50],
                    'turn': 8,
                    'sales_history': [60, 55, 58]
                }
            },
            {
                'name': 'Competitive Market - Strategic Marketing',
                'context': {
                    'inventory': 100,
                    'current_price': 10.25,
                    'competitor_prices': [10.00, 10.50, 10.30],
                    'turn': 12,
                    'sales_history': [48, 50, 52, 49]
                }
            }
        ]
        
        # Test each scenario
        for scenario in scenarios:
            print(f"\n=== {scenario['name']} ===")
            
            decision = await marketing_agent.make_decision(
                scenario['context'],
                f"TEST_{scenario['name'].replace(' ', '_').upper()}"
            )
            
            print(f"Marketing Spend: ${decision['marketing']}")
            print(f"Campaign Type: {decision.get('campaign_type', 'N/A')}")
            print(f"Expected Boost: +{decision.get('expected_demand_boost', 0)} units")
            print(f"ROI Projection: {decision.get('roi_projection', 'N/A')}")
            print(f"Confidence: {decision['confidence']:.2f}")
            print(f"AI Enhanced: {decision['ai_enhanced']}")
        
        # Show marketing insights
        print(f"\n=== Marketing Agent Performance ===")
        insights = marketing_agent.get_marketing_insights()
        print(f"Total campaigns: {insights['decisions_made']}")
        print(f"Average confidence: {insights['average_confidence']:.2f}")
        
        if 'marketing_insights' in insights:
            spend_patterns = insights['marketing_insights']['spend_patterns']
            print(f"Spend range: ${spend_patterns['min_spend']}-${spend_patterns['max_spend']}")
            print(f"Average spend: ${spend_patterns['avg_spend']:.0f}")
    
    # Run the test
    asyncio.run(test_marketing_agent())