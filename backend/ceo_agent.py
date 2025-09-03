# ceo_agent.py - CEO agent for strategic coordination and final decision making

import random  # For random values in fallback scenarios
from typing import Dict, Any, List  # Type hints for better code documentation

# Import base classes and configuration
from base_agent import BaseAgent, AgentConfig  # Parent class and shared configuration
from gemini_helper import GeminiAIHelper      # AI integration helper

class CEOAgent(BaseAgent):
    """
    CEO Agent responsible for strategic coordination and final decision making
    This agent either makes comprehensive decisions or coordinates other agent proposals
    """
    
    def __init__(self, ai_helper: GeminiAIHelper):
        """
        Initialize the CEO agent with strategic decision-making capabilities
        
        Args:
            ai_helper: Shared AI helper for intelligent strategic decisions
        """
        # Initialize parent class with agent type "ceo"
        super().__init__("ceo")
        
        # Store reference to AI helper for strategic intelligence
        self.ai_helper = ai_helper
        
        # CEO-specific tracking variables
        self.strategic_decisions = []    # Track all strategic decisions made
        self.coordination_history = []   # Track agent coordination activities
        self.risk_assessments = []      # Track risk analysis for decisions
        self.performance_tracking = []  # Track business performance results
        
        print("CEO Agent initialized - ready for strategic leadership")
    
    async def make_decision(self, context: Dict[str, Any], framework_context: str = "") -> Dict[str, Any]:
        """
        Make a comprehensive strategic decision covering all business areas
        
        Args:
            context: Complete market context and business data
            framework_context: Information about which framework is requesting decision
            
        Returns:
            Dictionary containing comprehensive strategic decision
        """
        print(f"CEO Agent making strategic decision for {framework_context}")
        
        # Store current context for strategic analysis
        self.current_context = context
        
        # Perform comprehensive strategic analysis
        strategic_analysis = self._perform_strategic_analysis(context)
        
        # Add strategic analysis to context for AI decision
        enhanced_context = context.copy()
        enhanced_context['strategic_analysis'] = strategic_analysis
        
        # Get AI-enhanced strategic decision
        decision = await self.ai_helper.get_ai_decision(
            "ceo", enhanced_context, framework_context
        )
        
        # If AI decision lacks comprehensive data, create strategic decision
        if not all(key in decision for key in ['price', 'production', 'marketing']):
            decision = await self._make_strategic_decision(context, framework_context)
        
        # Validate all decision components are within bounds
        decision['price'] = self.validate_decision_bounds("price", decision.get('price', 10.0))
        decision['production'] = int(self.validate_decision_bounds("production", decision.get('production', 50)))
        decision['marketing'] = int(self.validate_decision_bounds("marketing", decision.get('marketing', 500)))
        
        # Add CEO-level strategic analysis and risk assessment
        decision = self._add_strategic_analysis(decision, context, strategic_analysis)
        
        # Store this strategic decision
        self.add_decision_to_history(decision)
        self.strategic_decisions.append(decision)
        
        print(f"CEO strategic decision: ${decision['price']:.2f}, {decision['production']}u, ${decision['marketing']}m")
        return decision
    
    async def coordinate_decisions(self, agent_proposals: Dict[str, Dict[str, Any]], 
                                 context: Dict[str, Any], framework_context: str = "") -> Dict[str, Any]:
        """
        Coordinate decisions from specialist agents and resolve conflicts
        
        Args:
            agent_proposals: Decisions from specialist agents (pricing, production, marketing)
            context: Market context for coordination
            framework_context: Framework requesting coordination
            
        Returns:
            Coordinated final decision with conflict resolution
        """
        print(f"CEO Agent coordinating specialist proposals for {framework_context}")
        
        # Add agent proposals to context for AI coordination
        coordination_context = context.copy()
        coordination_context['agent_proposals'] = agent_proposals
        coordination_context['coordination_task'] = True
        
        # Get AI-enhanced coordination decision
        coordination_decision = await self.ai_helper.get_ai_decision(
            "ceo", coordination_context, f"{framework_context} - Agent Coordination"
        )
        
        # If AI coordination fails, use rule-based coordination
        if not all(key in coordination_decision for key in ['price', 'production', 'marketing']):
            coordination_decision = self._coordinate_proposals_fallback(agent_proposals, context)
        
        # Validate coordinated decisions
        coordination_decision['price'] = self.validate_decision_bounds("price", coordination_decision.get('price', 10.0))
        coordination_decision['production'] = int(self.validate_decision_bounds("production", coordination_decision.get('production', 50)))
        coordination_decision['marketing'] = int(self.validate_decision_bounds("marketing", coordination_decision.get('marketing', 500)))
        
        # Add coordination analysis
        coordination_decision = self._add_coordination_analysis(coordination_decision, agent_proposals, context)
        
        # Track coordination activity
        self.coordination_history.append({
            'proposals': agent_proposals,
            'final_decision': coordination_decision,
            'context': context,
            'framework': framework_context
        })
        
        print(f"CEO coordination complete: ${coordination_decision['price']:.2f}, {coordination_decision['production']}u, ${coordination_decision['marketing']}m")
        return coordination_decision
    
    def _perform_strategic_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive strategic analysis of business situation
        
        Args:
            context: Market and business context
            
        Returns:
            Dictionary containing strategic analysis
        """
        # Extract key business metrics
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitor_prices = context.get('competitor_prices', [10.0])
        turn = context.get('turn', 1)
        profit = context.get('profit', 100000)
        
        # Calculate business health indicators
        inventory_ratio = inventory / 100.0  # Normalized inventory level
        
        # Competitive position analysis
        avg_competitor = sum(competitor_prices) / len(competitor_prices) if competitor_prices else 10.0
        price_competitiveness = (current_price - avg_competitor) / avg_competitor if avg_competitor > 0 else 0
        
        # Business situation assessment
        if inventory_ratio > 2.0 and price_competitiveness > 0.1:
            situation = "crisis"           # High inventory + high price = crisis
            urgency = "immediate_action"
        elif inventory_ratio > 1.8 or price_competitiveness > 0.15:
            situation = "challenging"      # Either high inventory or high price
            urgency = "prompt_action"
        elif inventory_ratio < 0.3:
            situation = "opportunity"      # Low inventory = premium opportunity
            urgency = "capitalize"
        elif 0.7 <= inventory_ratio <= 1.3 and abs(price_competitiveness) < 0.05:
            situation = "optimal"          # Good inventory and competitive price
            urgency = "maintain"
        else:
            situation = "adjusting"        # Normal adjustments needed
            urgency = "gradual_improvement"
        
        # Calculate strategic priorities
        priorities = self._determine_strategic_priorities(situation, inventory_ratio, price_competitiveness)
        
        return {
            'business_situation': situation,
            'urgency_level': urgency,
            'inventory_assessment': {
                'ratio': round(inventory_ratio, 2),
                'status': 'excess' if inventory_ratio > 1.5 else 'shortage' if inventory_ratio < 0.5 else 'normal'
            },
            'competitive_assessment': {
                'price_difference_percentage': round(price_competitiveness * 100, 1),
                'position': 'premium' if price_competitiveness > 0.05 else 'discount' if price_competitiveness < -0.05 else 'competitive'
            },
            'strategic_priorities': priorities,
            'risk_factors': self._identify_risk_factors(context)
        }
    
    def _determine_strategic_priorities(self, situation: str, inventory_ratio: float, 
                                      price_competitiveness: float) -> List[str]:
        """
        Determine strategic priorities based on business situation
        
        Args:
            situation: Overall business situation assessment
            inventory_ratio: Normalized inventory level
            price_competitiveness: Price position vs competitors
            
        Returns:
            List of strategic priorities in order of importance
        """
        priorities = []
        
        # Inventory-based priorities
        if inventory_ratio > 2.0:
            priorities.append("urgent_inventory_reduction")
        elif inventory_ratio > 1.5:
            priorities.append("inventory_optimization")
        elif inventory_ratio < 0.3:
            priorities.append("inventory_building")
        
        # Pricing-based priorities
        if price_competitiveness > 0.15:
            priorities.append("price_competitiveness_improvement")
        elif price_competitiveness < -0.10:
            priorities.append("margin_optimization")
        
        # General strategic priorities based on situation
        if situation == "crisis":
            priorities.extend(["cost_control", "market_share_defense"])
        elif situation == "opportunity":
            priorities.extend(["revenue_maximization", "market_expansion"])
        elif situation == "optimal":
            priorities.extend(["efficiency_optimization", "sustainable_growth"])
        
        # Always include profit optimization as a priority
        if "margin_optimization" not in priorities:
            priorities.append("profit_optimization")
        
        return priorities[:4]  # Return top 4 priorities
    
    def _identify_risk_factors(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identify potential risk factors in current business situation
        
        Args:
            context: Business context for risk assessment
            
        Returns:
            List of identified risk factors with severity levels
        """
        risks = []
        
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitor_prices = context.get('competitor_prices', [10.0])
        
        # Inventory-related risks
        if inventory > 200:
            risks.append({
                'type': 'excess_inventory',
                'severity': 'high',
                'description': f'Very high inventory ({inventory} units) increases holding costs and obsolescence risk'
            })
        elif inventory < 20:
            risks.append({
                'type': 'stockout_risk',
                'severity': 'high', 
                'description': f'Critically low inventory ({inventory} units) risks lost sales and customer dissatisfaction'
            })
        
        # Pricing-related risks
        min_competitor = min(competitor_prices) if competitor_prices else 10.0
        max_competitor = max(competitor_prices) if competitor_prices else 10.0
        
        if current_price > max_competitor + 1.0:
            risks.append({
                'type': 'price_premium_risk',
                'severity': 'medium',
                'description': f'Price ${current_price:.2f} significantly above highest competitor ${max_competitor:.2f}'
            })
        elif current_price < min_competitor - 1.0:
            risks.append({
                'type': 'margin_erosion_risk',
                'severity': 'medium',
                'description': f'Price ${current_price:.2f} significantly below lowest competitor ${min_competitor:.2f}'
            })
        
        # Profitability risk
        min_profitable_price = AgentConfig.UNIT_PRODUCTION_COST / (1 - AgentConfig.TARGET_PROFIT_MARGIN)
        if current_price < min_profitable_price:
            risks.append({
                'type': 'profitability_risk',
                'severity': 'high',
                'description': f'Price ${current_price:.2f} below minimum profitable price ${min_profitable_price:.2f}'
            })
        
        return risks
    
    async def _make_strategic_decision(self, context: Dict[str, Any], framework_context: str) -> Dict[str, Any]:
        """
        Make comprehensive strategic decision with detailed analysis
        
        Args:
            context: Market context
            framework_context: Framework information
            
        Returns:
            Comprehensive strategic decision
        """
        # Get strategic analysis
        strategic_analysis = context.get('strategic_analysis', {})
        
        # Create comprehensive CEO prompt
        ceo_prompt = self._create_ceo_prompt(context, framework_context, strategic_analysis)
        
        # Try AI with strategic prompt
        if self.ai_helper.enabled:
            try:
                import asyncio
                response = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.ai_helper.model.generate_content(ceo_prompt)
                )
                
                if response and response.text:
                    return self.ai_helper._parse_ai_response(response.text, "ceo")
                else:
                    raise ValueError("Empty CEO AI response")
                    
            except Exception as e:
                print(f"CEO AI decision failed: {e}")
        
        # Fallback to strategic rule-based decision
        return self._ceo_strategic_fallback(context, strategic_analysis)
    
    def _create_ceo_prompt(self, context: Dict[str, Any], framework_context: str,
                          strategic_analysis: Dict[str, Any]) -> str:
        """
        Create comprehensive CEO prompt for strategic decision making
        
        Args:
            context: Market context
            framework_context: Framework information
            strategic_analysis: Strategic business analysis
            
        Returns:
            Detailed prompt for CEO strategic decisions
        """
        # Extract key information
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitor_prices = context.get('competitor_prices', [10.0])
        turn = context.get('turn', 1)
        
        # Get strategic analysis components
        situation = strategic_analysis.get('business_situation', 'normal')
        urgency = strategic_analysis.get('urgency_level', 'normal')
        priorities = strategic_analysis.get('strategic_priorities', [])
        risks = strategic_analysis.get('risk_factors', [])
        
        prompt = f"""
CEO STRATEGIC DECISION MAKING

You are the CEO making comprehensive strategic decisions for the brewery business.
Framework Context: {framework_context}

CURRENT BUSINESS SITUATION:
- Overall Status: {situation.upper()}
- Urgency Level: {urgency}
- Current Inventory: {inventory} units
- Current Price: ${current_price:.2f}
- Competitor Prices: {competitor_prices}
- Business Turn: {turn}

STRATEGIC ANALYSIS:
- Business Situation: {strategic_analysis.get('business_situation', 'normal')}
- Inventory Status: {strategic_analysis.get('inventory_assessment', {}).get('status', 'normal')}
- Competitive Position: {strategic_analysis.get('competitive_assessment', {}).get('position', 'competitive')}
- Strategic Priorities: {priorities}

RISK ASSESSMENT:
{chr(10).join([f"- {risk['type']}: {risk['description']} (Severity: {risk['severity']})" for risk in risks]) if risks else "- No significant risks identified"}

CEO RESPONSIBILITIES:
1. STRATEGIC COORDINATION:
   - Align pricing, production, and marketing strategies
   - Balance short-term and long-term objectives
   - Optimize overall business performance

2. RISK MANAGEMENT:
   - Address identified risk factors
   - Implement risk mitigation strategies
   - Ensure business sustainability

3. PERFORMANCE OPTIMIZATION:
   - Maximize profitability and market position
   - Efficient resource allocation
   - Competitive advantage development

DECISION SCENARIOS:

CRISIS MANAGEMENT (High inventory + competitive pressure):
- Aggressive pricing reduction to move inventory
- Reduced production to stop inventory buildup
- Heavy marketing investment to drive demand

GROWTH OPPORTUNITY (Low inventory + strong position):
- Premium pricing to maximize margins
- Increased production to capture demand
- Targeted marketing to maintain momentum

OPTIMIZATION MODE (Balanced situation):
- Competitive pricing for market share
- Demand-responsive production planning
- Efficient marketing for ROI optimization

YOUR TASK:
Make comprehensive strategic decisions that:
- Address the current business situation effectively
- Implement strategic priorities in order of importance
- Mitigate identified risks
- Optimize overall business performance
- Coordinate all business functions effectively

RESPOND IN EXACT JSON FORMAT:
{{
    "price": 10.50,
    "production": 65,
    "marketing": 850,
    "reasoning": "Comprehensive strategic rationale and business logic",
    "confidence": 0.85,
    "strategy_type": "crisis/growth/optimization/defensive",
    "risk_mitigation": "how this decision addresses identified risks",
    "performance_projection": "expected business performance impact",
    "coordination_approach": "how pricing, production, and marketing work together"
}}
"""
        return prompt
    
    def _ceo_strategic_fallback(self, context: Dict[str, Any], 
                              strategic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based strategic fallback when AI is unavailable
        
        Args:
            context: Market context
            strategic_analysis: Strategic business analysis
            
        Returns:
            Rule-based strategic decision
        """
        # Extract key metrics
        inventory = context.get('inventory', 100)
        competitor_prices = context.get('competitor_prices', [10.0])
        situation = strategic_analysis.get('business_situation', 'normal')
        priorities = strategic_analysis.get('strategic_priorities', [])
        
        # Calculate baseline values
        avg_competitor = sum(competitor_prices) / len(competitor_prices) if competitor_prices else 10.0
        inventory_ratio = inventory / 100.0
        
        # Strategic decision logic based on business situation
        if situation == "crisis":
            # Crisis management: aggressive action needed
            price = max(AgentConfig.MIN_PRICE, avg_competitor - random.uniform(1.0, 2.0))
            production = AgentConfig.MIN_PRODUCTION + random.randint(0, 20)
            marketing = random.randint(1500, AgentConfig.MAX_MARKETING)
            strategy_type = "crisis"
            reasoning = "Crisis management: Aggressive pricing and marketing to clear excess inventory"
            
        elif situation == "opportunity":
            # Growth opportunity: maximize value
            price = min(AgentConfig.MAX_PRICE, avg_competitor + random.uniform(0.8, 1.5))
            production = random.randint(100, AgentConfig.MAX_PRODUCTION)
            marketing = random.randint(300, 600)
            strategy_type = "growth"
            reasoning = "Growth opportunity: Premium pricing with increased production capacity"
            
        elif situation == "optimal":
            # Optimization: fine-tune for efficiency
            price = avg_competitor + random.uniform(-0.2, 0.2)
            production = random.randint(60, 90)
            marketing = random.randint(700, 1100)
            strategy_type = "optimization"
            reasoning = "Optimization strategy: Balanced approach for sustained performance"
            
        else:
            # Default adjusting strategy
            if inventory_ratio > 1.5:
                # High inventory - focus on movement
                price = max(AgentConfig.MIN_PRICE, avg_competitor - 0.5)
                production = random.randint(40, 70)
                marketing = random.randint(1000, 1400)
                strategy_type = "defensive"
                reasoning = "Defensive strategy: Price reduction and marketing boost to manage inventory"
            elif inventory_ratio < 0.5:
                # Low inventory - focus on building
                price = min(AgentConfig.MAX_PRICE, avg_competitor + 0.5)
                production = random.randint(90, 120)
                marketing = random.randint(400, 700)
                strategy_type = "growth"
                reasoning = "Growth strategy: Premium pricing with production increase"
            else:
                # Normal operations
                price = avg_competitor + random.uniform(-0.3, 0.3)
                production = random.randint(50, 80)
                marketing = random.randint(600, 1000)
                strategy_type = "optimization"
                reasoning = "Balanced strategy: Market-competitive approach with normal operations"
        
        # Calculate performance projections
        expected_demand = 50 + (marketing / 500) * 15 + (10 - price) * 5
        expected_revenue = min(expected_demand, inventory + production) * price
        expected_costs = production * AgentConfig.UNIT_PRODUCTION_COST + marketing + inventory * AgentConfig.UNIT_HOLDING_COST
        expected_profit = expected_revenue - expected_costs
        
        return {
            'price': round(price, 2),
            'production': max(AgentConfig.MIN_PRODUCTION, min(AgentConfig.MAX_PRODUCTION, production)),
            'marketing': max(AgentConfig.MIN_MARKETING, min(AgentConfig.MAX_MARKETING, marketing)),
            'reasoning': reasoning,
            'confidence': 0.7,
            'strategy_type': strategy_type,
            'risk_mitigation': f"Addresses {situation} situation with {strategy_type} approach",
            'performance_projection': f"Expected profit: ${expected_profit:.0f} from {expected_demand:.0f} unit demand",
            'coordination_approach': "Integrated strategic approach balancing all business functions",
            'agent_type': 'ceo',
            'ai_enhanced': False,
            'source': 'strategic_rule_based'
        }
    
    def _coordinate_proposals_fallback(self, agent_proposals: Dict[str, Dict[str, Any]], 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rule-based coordination of specialist agent proposals
        
        Args:
            agent_proposals: Proposals from pricing, production, marketing agents
            context: Market context
            
        Returns:
            Coordinated decision resolving any conflicts
        """
        # Extract individual agent proposals
        pricing_proposal = agent_proposals.get('pricing', {})
        production_proposal = agent_proposals.get('production', {})
        marketing_proposal = agent_proposals.get('marketing', {})
        
        # Get proposed values with defaults
        proposed_price = pricing_proposal.get('price', 10.0)
        proposed_production = production_proposal.get('production', 50)
        proposed_marketing = marketing_proposal.get('marketing', 500)
        
        # Analyze for conflicts and coordination opportunities
        coordination_adjustments = []
        
        # Check for strategic conflicts
        inventory = context.get('inventory', 100)
        
        # Conflict 1: High price + high marketing might be wasteful
        if proposed_price > 12.0 and proposed_marketing > 1200:
            proposed_marketing = int(proposed_marketing * 0.8)  # Reduce marketing for premium pricing
            coordination_adjustments.append("Reduced marketing spend due to premium pricing strategy")
        
        # Conflict 2: Low inventory + high production might create excess
        if inventory < 50 and proposed_production > 100:
            proposed_production = min(proposed_production, 80)  # Cap production
            coordination_adjustments.append("Moderated production increase to avoid inventory swing")
        
        # Conflict 3: High inventory + low marketing might not clear stock
        if inventory > 150 and proposed_marketing < 600:
            proposed_marketing = max(proposed_marketing, 1000)  # Increase marketing
            coordination_adjustments.append("Increased marketing to address high inventory")
        
        # Ensure all values are within bounds
        final_price = self.validate_decision_bounds("price", proposed_price)
        final_production = int(self.validate_decision_bounds("production", proposed_production))
        final_marketing = int(self.validate_decision_bounds("marketing", proposed_marketing))
        
        return {
            'price': final_price,
            'production': final_production,
            'marketing': final_marketing,
            'reasoning': f"CEO coordination: Integrated specialist proposals with {len(coordination_adjustments)} strategic adjustments",
            'confidence': 0.75,
            'strategy_type': 'coordinated',
            'coordination_adjustments': coordination_adjustments,
            'original_proposals': {
                'pricing': proposed_price,
                'production': proposed_production,
                'marketing': proposed_marketing
            },
            'agent_type': 'ceo',
            'ai_enhanced': False,
            'source': 'coordination_fallback'
        }
    
    def _add_strategic_analysis(self, decision: Dict[str, Any], context: Dict[str, Any],
                              strategic_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add strategic analysis and business impact projections to the decision
        
        Args:
            decision: Base decision to enhance
            context: Market context
            strategic_analysis: Strategic business analysis
            
        Returns:
            Enhanced decision with strategic analysis
        """
        # Extract decision components
        price = decision.get('price', 10.0)
        production = decision.get('production', 50)
        marketing = decision.get('marketing', 500)
        
        # Calculate business impact projections
        inventory = context.get('inventory', 100)
        
        # Demand projection based on price and marketing
        base_demand = 50
        price_effect = (10 - price) * 5  # Lower price increases demand
        marketing_effect = (marketing / 500) * 15  # Marketing boost
        projected_demand = max(10, base_demand + price_effect + marketing_effect)
        
        # Sales projection (limited by inventory availability)
        projected_sales = min(projected_demand, inventory + production)
        
        # Revenue and cost projections
        projected_revenue = projected_sales * price
        production_costs = production * AgentConfig.UNIT_PRODUCTION_COST
        marketing_costs = marketing
        holding_costs = (inventory + production - projected_sales) * AgentConfig.UNIT_HOLDING_COST
        total_costs = production_costs + marketing_costs + holding_costs
        projected_profit = projected_revenue - total_costs
        
        # Calculate strategic metrics
        profit_margin = (projected_profit / projected_revenue) * 100 if projected_revenue > 0 else 0
        inventory_turnover = projected_sales / (inventory + production) if (inventory + production) > 0 else 0
        
        # Add comprehensive strategic analysis
        decision['strategic_analysis'] = {
            'business_projections': {
                'projected_demand': round(projected_demand, 1),
                'projected_sales': round(projected_sales, 1),
                'projected_revenue': round(projected_revenue, 2),
                'projected_profit': round(projected_profit, 2),
                'profit_margin_percentage': round(profit_margin, 1)
            },
            'cost_breakdown': {
                'production_costs': round(production_costs, 2),
                'marketing_costs': marketing_costs,
                'holding_costs': round(holding_costs, 2),
                'total_costs': round(total_costs, 2)
            },
            'efficiency_metrics': {
                'inventory_turnover': round(inventory_turnover, 2),
                'revenue_per_unit_produced': round(projected_revenue / production if production > 0 else 0, 2),
                'marketing_roi': round(((marketing_effect * price) / marketing if marketing > 0 else 0), 2)
            },
            'strategic_alignment': {
                'addresses_priorities': strategic_analysis.get('strategic_priorities', [])[:2],
                'risk_mitigation_score': self._calculate_risk_mitigation_score(decision, strategic_analysis),
                'overall_strategic_fit': self._assess_strategic_fit(decision, strategic_analysis)
            }
        }
        
        return decision
    
    def _add_coordination_analysis(self, coordination_decision: Dict[str, Any], 
                                 agent_proposals: Dict[str, Dict[str, Any]], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add analysis of how CEO coordinated the specialist agent proposals
        
        Args:
            coordination_decision: Final coordinated decision
            agent_proposals: Original proposals from specialist agents
            context: Market context
            
        Returns:
            Enhanced decision with coordination analysis
        """
        # Extract final values
        final_price = coordination_decision.get('price', 10.0)
        final_production = coordination_decision.get('production', 50)
        final_marketing = coordination_decision.get('marketing', 500)
        
        # Extract original proposals
        original_price = agent_proposals.get('pricing', {}).get('price', final_price)
        original_production = agent_proposals.get('production', {}).get('production', final_production)
        original_marketing = agent_proposals.get('marketing', {}).get('marketing', final_marketing)
        
        # Calculate adjustments made
        price_adjustment = final_price - original_price
        production_adjustment = final_production - original_production
        marketing_adjustment = final_marketing - original_marketing
        
        # Analyze coordination decisions
        adjustments_made = []
        if abs(price_adjustment) > 0.1:
            adjustments_made.append(f"Price adjusted by ${price_adjustment:+.2f}")
        if abs(production_adjustment) > 5:
            adjustments_made.append(f"Production adjusted by {production_adjustment:+d} units")
        if abs(marketing_adjustment) > 50:
            adjustments_made.append(f"Marketing adjusted by ${marketing_adjustment:+d}")
        
        # Calculate agreement score (how much CEO agreed with specialists)
        total_adjustments = abs(price_adjustment) + abs(production_adjustment/10) + abs(marketing_adjustment/100)
        agreement_score = max(0, 1 - (total_adjustments / 3))  # Normalize to 0-1 scale
        
        # Add coordination analysis
        coordination_decision['coordination_analysis'] = {
            'original_proposals': {
                'pricing_agent': original_price,
                'production_agent': original_production,
                'marketing_agent': original_marketing
            },
            'final_decisions': {
                'price': final_price,
                'production': final_production,
                'marketing': final_marketing
            },
            'adjustments_made': adjustments_made,
            'adjustment_summary': {
                'price_change': round(price_adjustment, 2),
                'production_change': production_adjustment,
                'marketing_change': marketing_adjustment
            },
            'coordination_metrics': {
                'agreement_score': round(agreement_score, 2),
                'total_adjustments': len(adjustments_made),
                'coordination_type': 'major_override' if agreement_score < 0.5 else 'minor_adjustments' if agreement_score < 0.9 else 'accepted_proposals'
            }
        }
        
        return coordination_decision
    
    def _calculate_risk_mitigation_score(self, decision: Dict[str, Any], 
                                       strategic_analysis: Dict[str, Any]) -> float:
        """
        Calculate how well this decision mitigates identified risks
        
        Args:
            decision: The strategic decision made
            strategic_analysis: Analysis containing identified risks
            
        Returns:
            Risk mitigation score (0.0 to 1.0)
        """
        risks = strategic_analysis.get('risk_factors', [])
        if not risks:
            return 1.0  # No risks to mitigate
        
        mitigation_score = 0
        for risk in risks:
            risk_type = risk['type']
            
            # Check if decision addresses this specific risk
            if risk_type == 'excess_inventory' and decision.get('marketing', 0) > 1000:
                mitigation_score += 0.3  # High marketing helps with excess inventory
            elif risk_type == 'stockout_risk' and decision.get('production', 0) > 80:
                mitigation_score += 0.3  # High production helps with stockout risk
            elif risk_type == 'price_premium_risk' and decision.get('price', 15) < 11.0:
                mitigation_score += 0.2  # Lower price reduces premium risk
            elif risk_type == 'margin_erosion_risk' and decision.get('price', 8) > 9.5:
                mitigation_score += 0.2  # Higher price helps margins
        
        # Normalize score
        return min(1.0, mitigation_score)
    
    def _assess_strategic_fit(self, decision: Dict[str, Any], 
                            strategic_analysis: Dict[str, Any]) -> str:
        """
        Assess how well the decision fits the strategic situation
        
        Args:
            decision: Strategic decision made
            strategic_analysis: Strategic situation analysis
            
        Returns:
            String describing strategic fit quality
        """
        situation = strategic_analysis.get('business_situation', 'normal')
        strategy_type = decision.get('strategy_type', 'balanced')
        
        # Check alignment between situation and strategy
        if situation == 'crisis' and strategy_type == 'crisis':
            return 'excellent_fit'
        elif situation == 'opportunity' and strategy_type == 'growth':
            return 'excellent_fit'
        elif situation == 'optimal' and strategy_type == 'optimization':
            return 'excellent_fit'
        elif situation in ['adjusting', 'normal'] and strategy_type in ['optimization', 'defensive']:
            return 'good_fit'
        else:
            return 'adequate_fit'
    
    def get_ceo_insights(self) -> Dict[str, Any]:
        """
        Get detailed insights about CEO performance and strategic effectiveness
        
        Returns:
            Dictionary containing CEO-specific insights and metrics
        """
        # Get base agent statistics
        base_stats = self.get_agent_stats()
        
        # Calculate CEO-specific insights
        ceo_insights = {
            'strategic_performance': {
                'total_strategic_decisions': len(self.strategic_decisions),
                'total_coordinations': len(self.coordination_history),
                'strategy_distribution': self._analyze_strategy_distribution(),
                'coordination_effectiveness': self._analyze_coordination_effectiveness()
            },
            'risk_management': {
                'risks_identified': self._count_risks_identified(),
                'risk_mitigation_effectiveness': self._evaluate_risk_mitigation()
            },
            'business_impact': {
                'average_projected_profit': self._calculate_average_projected_profit(),
                'strategic_consistency': self._measure_strategic_consistency()
            }
        }
        
        # Combine with base stats
        base_stats['ceo_insights'] = ceo_insights
        return base_stats
    
    def _analyze_strategy_distribution(self) -> Dict[str, int]:
        """Analyze distribution of strategy types used"""
        strategy_counts = {}
        for decision in self.strategic_decisions:
            strategy = decision.get('strategy_type', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        return strategy_counts
    
    def _analyze_coordination_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of agent coordination"""
        if not self.coordination_history:
            return {'status': 'no_coordinations'}
        
        agreement_scores = []
        for coord in self.coordination_history:
            if 'coordination_analysis' in coord['final_decision']:
                score = coord['final_decision']['coordination_analysis']['coordination_metrics']['agreement_score']
                agreement_scores.append(score)
        
        avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0
        
        return {
            'average_agreement_score': round(avg_agreement, 2),
            'coordinations_analyzed': len(agreement_scores)
        }
    
    def _count_risks_identified(self) -> int:
        """Count total risks identified across all decisions"""
        total_risks = 0
        for decision in self.strategic_decisions:
            if 'strategic_analysis' in decision and 'strategic_alignment' in decision['strategic_analysis']:
                # This would count risks from strategic analysis
                total_risks += 1  # Simplified for example
        return total_risks
    
    def _evaluate_risk_mitigation(self) -> float:
        """Evaluate average risk mitigation effectiveness"""
        mitigation_scores = []
        for decision in self.strategic_decisions:
            if 'strategic_analysis' in decision and 'strategic_alignment' in decision['strategic_analysis']:
                score = decision['strategic_analysis']['strategic_alignment'].get('risk_mitigation_score', 0.5)
                mitigation_scores.append(score)
        
        return sum(mitigation_scores) / len(mitigation_scores) if mitigation_scores else 0.5
    
    def _calculate_average_projected_profit(self) -> float:
        """Calculate average projected profit from strategic decisions"""
        profits = []
        for decision in self.strategic_decisions:
            if 'strategic_analysis' in decision and 'business_projections' in decision['strategic_analysis']:
                profit = decision['strategic_analysis']['business_projections'].get('projected_profit', 0)
                profits.append(profit)
        
        return sum(profits) / len(profits) if profits else 0
    
    def _measure_strategic_consistency(self) -> float:
        """Measure consistency in strategic approach over time"""
        if len(self.strategic_decisions) < 2:
            return 1.0  # Perfect consistency with only one decision
        
        # Count strategy types used
        strategies = [d.get('strategy_type', 'unknown') for d in self.strategic_decisions]
        unique_strategies = len(set(strategies))
        
        # More consistent = fewer different strategies
        consistency_score = 1.0 - (unique_strategies - 1) / len(strategies)
        return max(0.0, consistency_score)

# Example usage and testing
if __name__ == "__main__":
    """Test the CEO agent independently"""
    import asyncio
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    async def test_ceo_agent():
        """Test CEO agent with different business scenarios"""
        print("Testing CEO Agent...")
        
        # Initialize CEO agent
        api_key = os.getenv("GEMINI_API_KEY")
        ai_helper = GeminiAIHelper(api_key)
        ceo_agent = CEOAgent(ai_helper)
        
        # Test scenarios
        scenarios = [
            {
                'name': 'Crisis Management Scenario',
                'context': {
                    'inventory': 250,
                    'current_price': 11.50,
                    'competitor_prices': [9.00, 10.00, 9.50],
                    'turn': 8,
                    'profit': 95000,
                    'sales_history': [25, 30, 28]
                }
            },
            {
                'name': 'Growth Opportunity Scenario',
                'context': {
                    'inventory': 20,
                    'current_price': 10.00,
                    'competitor_prices': [11.00, 12.00, 10.50],
                    'turn': 5,
                    'profit': 110000,
                    'sales_history': [70, 65, 68]
                }
            },
            {
                'name': 'Agent Coordination Scenario',
                'context': {
                    'inventory': 90,
                    'current_price': 10.25,
                    'competitor_prices': [10.00, 10.50],
                    'turn': 10,
                    'profit': 105000
                },
                'agent_proposals': {
                    'pricing': {'price': 9.75, 'confidence': 0.8},
                    'production': {'production': 80, 'confidence': 0.7},
                    'marketing': {'marketing': 1200, 'confidence': 0.9}
                }
            }
        ]
        
        # Test strategic decisions
        for i, scenario in enumerate(scenarios[:2]):
            print(f"\n=== {scenario['name']} ===")
            
            decision = await ceo_agent.make_decision(
                scenario['context'],
                f"TEST_{scenario['name'].replace(' ', '_').upper()}"
            )
            
            print(f"Strategic Decision: ${decision['price']:.2f}, {decision['production']}u, ${decision['marketing']}m")
            print(f"Strategy Type: {decision.get('strategy_type', 'N/A')}")
            print(f"Risk Mitigation: {decision.get('risk_mitigation', 'N/A')[:60]}...")
            print(f"Performance Projection: {decision.get('performance_projection', 'N/A')[:60]}...")
            print(f"Confidence: {decision['confidence']:.2f}")
        
        # Test coordination capability
        if len(scenarios) > 2:
            print(f"\n=== {scenarios[2]['name']} ===")
            coordination = await ceo_agent.coordinate_decisions(
                scenarios[2]['agent_proposals'],
                scenarios[2]['context'],
                "TEST_COORDINATION"
            )
            
            print(f"Coordinated Decision: ${coordination['price']:.2f}, {coordination['production']}u, ${coordination['marketing']}m")
            print(f"Adjustments Made: {coordination.get('coordination_adjustments', [])}")
        
        # Show CEO insights
        print(f"\n=== CEO Agent Performance ===")
        insights = ceo_agent.get_ceo_insights()
        print(f"Strategic decisions: {insights['decisions_made']}")
        print(f"Average confidence: {insights['average_confidence']:.2f}")
        
        if 'ceo_insights' in insights:
            strategic_perf = insights['ceo_insights']['strategic_performance']
            print(f"Coordinations performed: {strategic_perf['total_coordinations']}")
            print(f"Strategy distribution: {strategic_perf['strategy_distribution']}")
    
    # Run the test
    asyncio.run(test_ceo_agent())