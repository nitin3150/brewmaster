# gemini_helper.py - Centralized Gemini AI integration for all agents

import asyncio  # For asynchronous programming and running blocking calls in threads
import json     # For parsing JSON responses from Gemini AI
import re       # For regular expressions to extract JSON from AI responses
from typing import Dict, Any, Optional  # Type hints for better code documentation

# Try to import Gemini AI library with fallback handling
try:
    import google.generativeai as genai  # Google's Gemini AI library
    GEMINI_AVAILABLE = True              # Flag to track if library is available
    print("Gemini AI library loaded successfully")
except ImportError:
    # If import fails, set flag to False and create a None placeholder
    GEMINI_AVAILABLE = False
    genai = None
    print("Gemini AI library not available - will use fallback logic")

class GeminiAIHelper:
    """
    Centralized helper class for Gemini AI integration
    This class handles all AI communication for all agents
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini AI helper
        
        Args:
            api_key: Google Gemini API key (optional)
        """
        # Initialize instance variables
        self.enabled = False           # Whether AI is successfully enabled
        self.model = None             # The Gemini model instance
        self.model_name = None        # Name of the working model
        
        print("Initializing Gemini AI Helper...")
        
        # Check if we have both an API key and the Gemini library
        if api_key and GEMINI_AVAILABLE:
            self._initialize_gemini(api_key)
        else:
            # Log why AI is disabled
            if not api_key:
                print("Gemini AI disabled: No API key provided")
            if not GEMINI_AVAILABLE:
                print("Gemini AI disabled: Library not installed")
    
    def _initialize_gemini(self, api_key: str) -> None:
        """
        Private method to initialize Gemini AI with error handling
        
        Args:
            api_key: The API key to use for Gemini
        """
        try:
            print(f"Configuring Gemini with API key: {api_key[:10]}...")
            
            # Configure the Gemini library with the API key
            genai.configure(api_key=api_key)
            
            # List of model names to try (in order of preference)
            models_to_try = [
                'gemini-1.5-flash',     # Fastest and latest model
                'gemini-1.5-pro',      # More capable model
                'gemini-1.0-pro'       # Fallback older model
            ]
            
            # Try each model until we find one that works
            for model_name in models_to_try:
                if self._test_model(model_name):
                    # If model test succeeds, use this model
                    self.model = genai.GenerativeModel(model_name)
                    self.model_name = model_name
                    self.enabled = True
                    print(f"Successfully initialized Gemini model: {model_name}")
                    return  # Exit the loop since we found a working model
            
            # If no models worked
            print("No working Gemini models found")
            
        except Exception as e:
            # Handle any unexpected errors during initialization
            print(f"Gemini initialization failed: {e}")
            self.enabled = False
    
    def _test_model(self, model_name: str) -> bool:
        """
        Test if a specific Gemini model works
        
        Args:
            model_name: Name of the model to test
            
        Returns:
            True if model works, False otherwise
        """
        try:
            print(f"Testing model: {model_name}")
            
            # Create a test model instance
            test_model = genai.GenerativeModel(model_name)
            
            # Send a simple test prompt
            test_response = test_model.generate_content("Hello")
            
            # Check if we got a valid response
            if test_response and test_response.text:
                print(f"Model {model_name} test successful")
                return True
            else:
                print(f"Model {model_name} returned empty response")
                return False
                
        except Exception as e:
            # If any error occurs, this model doesn't work
            print(f"Model {model_name} test failed: {e}")
            return False
    
    async def get_ai_decision(self, agent_type: str, context: Dict[str, Any], 
                            framework_context: str = "") -> Dict[str, Any]:
        """
        Get an AI-enhanced decision from Gemini
        
        Args:
            agent_type: Type of agent making the decision ("pricing", "production", etc.)
            context: Market context and data for decision making
            framework_context: Information about which framework is calling
            
        Returns:
            Dictionary containing the AI decision or fallback decision
        """
        # If AI is enabled, try to get AI decision
        if self.enabled:
            try:
                print(f"Getting AI decision for {agent_type} agent...")
                return await self._make_ai_decision(agent_type, context, framework_context)
            except Exception as e:
                # If AI fails, fall back to rule-based decision
                print(f"AI decision failed for {agent_type}: {e}")
                print("Falling back to rule-based decision")
                return self._make_fallback_decision(agent_type, context)
        else:
            # If AI is disabled, use rule-based decision
            print(f"Using rule-based decision for {agent_type} (AI disabled)")
            return self._make_fallback_decision(agent_type, context)
    
    async def _make_ai_decision(self, agent_type: str, context: Dict[str, Any], 
                              framework_context: str) -> Dict[str, Any]:
        """
        Private method to get decision from Gemini AI
        
        Args:
            agent_type: Type of agent
            context: Market context
            framework_context: Framework information
            
        Returns:
            Parsed AI decision
        """
        # Create a specialized prompt for this agent type
        prompt = self._create_agent_prompt(agent_type, context, framework_context)
        
        # Call Gemini API in a separate thread to avoid blocking
        response = await asyncio.get_event_loop().run_in_executor(
            None,  # Use default thread pool
            lambda: self.model.generate_content(prompt)  # The blocking call to Gemini
        )
        
        # Validate that we got a response
        if not response or not response.text:
            raise ValueError("Empty response from Gemini AI")
        
        # Parse and return the AI response
        return self._parse_ai_response(response.text, agent_type)
    
    def _create_agent_prompt(self, agent_type: str, context: Dict[str, Any], 
                           framework_context: str) -> str:
        """
        Create a specialized prompt for each agent type
        
        Args:
            agent_type: Type of agent needing a decision
            context: Market context and data
            framework_context: Which framework is calling
            
        Returns:
            Formatted prompt string for Gemini AI
        """
        # This method is overridden by specific agent classes
        # Each agent creates its own specialized prompts
        return f"Basic prompt for {agent_type} agent"
    
    def _parse_ai_response(self, response_text: str, agent_type: str) -> Dict[str, Any]:
        """
        Parse the AI response text and extract decision data
        
        Args:
            response_text: Raw text response from Gemini AI
            agent_type: Type of agent that made the request
            
        Returns:
            Parsed decision dictionary
        """
        try:
            # Use regex to find JSON in the response text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                # Parse the JSON string into a Python dictionary
                decision_data = json.loads(json_match.group())
            else:
                # If no JSON found, raise an error to trigger fallback
                raise ValueError("No JSON found in AI response")
            
            # Add metadata about this decision
            decision_data['agent_type'] = agent_type
            decision_data['ai_enhanced'] = True
            decision_data['source'] = 'gemini_ai'
            decision_data['model_used'] = self.model_name
            
            return decision_data
            
        except Exception as e:
            # If parsing fails, log the error and raise to trigger fallback
            print(f"Failed to parse AI response for {agent_type}: {e}")
            print(f"Raw response: {response_text[:200]}...")  # Show first 200 chars for debugging
            raise  # Re-raise the exception to trigger fallback logic
    
    def _make_fallback_decision(self, agent_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a rule-based fallback decision when AI is unavailable
        
        Args:
            agent_type: Type of agent needing a decision
            context: Market context for decision making
            
        Returns:
            Rule-based decision dictionary
        """
        # Extract key context information
        inventory = context.get('inventory', 100)
        competitor_prices = context.get('competitor_prices', [10.0])
        
        # Calculate average competitor price for competitive analysis
        avg_competitor_price = sum(competitor_prices) / len(competitor_prices) if competitor_prices else 10.0
        
        # Calculate inventory ratio to determine urgency (1.0 = normal, >1.0 = excess, <1.0 = shortage)
        inventory_ratio = inventory / 100.0  # Normalize against baseline of 100 units
        
        # Create basic fallback decision structure
        fallback_decision = {
            'agent_type': agent_type,
            'ai_enhanced': False,
            'source': 'rule_based_fallback',
            'confidence': 0.6,  # Lower confidence for rule-based decisions
            'reasoning': f'{agent_type.title()} Agent: Rule-based decision (AI unavailable)'
        }
        
        # This is a placeholder - specific agents will override this method
        # with their own specialized fallback logic
        return fallback_decision# gemini_helper.py - Centralized Gemini AI integration for all agents

import asyncio  # For asynchronous programming and running blocking calls in threads
import json     # For parsing JSON responses from Gemini AI
import re       # For regular expressions to extract JSON from AI responses
from typing import Dict, Any, Optional  # Type hints for better code documentation

# Try to import Gemini AI library with fallback handling
try:
    import google.generativeai as genai  # Google's Gemini AI library
    GEMINI_AVAILABLE = True              # Flag to track if library is available
    print("Gemini AI library loaded successfully")
except ImportError:
    # If import fails, set flag to False and create a None placeholder
    GEMINI_AVAILABLE = False
    genai = None
    print("Gemini AI library not available - will use fallback logic")

class GeminiAIHelper:
    """
    Centralized helper class for Gemini AI integration
    This class handles all AI communication for all agents
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Gemini AI helper
        
        Args:
            api_key: Google Gemini API key (optional)
        """
        # Initialize instance variables
        self.enabled = False           # Whether AI is successfully enabled
        self.model = None             # The Gemini model instance
        self.model_name = None        # Name of the working model
        
        print("Initializing Gemini AI Helper...")
        
        # Check if we have both an API key and the Gemini library
        if api_key and GEMINI_AVAILABLE:
            self._initialize_gemini(api_key)
        else:
            # Log why AI is disabled
            if not api_key:
                print("Gemini AI disabled: No API key provided")
            if not GEMINI_AVAILABLE:
                print("Gemini AI disabled: Library not installed")
    
    def _initialize_gemini(self, api_key: str) -> None:
        """
        Private method to initialize Gemini AI with error handling
        
        Args:
            api_key: The API key to use for Gemini
        """
        try:
            print(f"Configuring Gemini with API key: {api_key[:10]}...")
            
            # Configure the Gemini library with the API key
            genai.configure(api_key=api_key)
            
            # List of model names to try (in order of preference)
            models_to_try = [
                'gemini-1.5-flash',     # Fastest and latest model
                'gemini-1.5-pro',      # More capable model
                'gemini-1.0-pro'       # Fallback older model
            ]
            
            # Try each model until we find one that works
            for model_name in models_to_try:
                if self._test_model(model_name):
                    # If model test succeeds, use this model
                    self.model = genai.GenerativeModel(model_name)
                    self.model_name = model_name
                    self.enabled = True
                    print(f"Successfully initialized Gemini model: {model_name}")
                    return  # Exit the loop since we found a working model
            
            # If no models worked
            print("No working Gemini models found")
            
        except Exception as e:
            # Handle any unexpected errors during initialization
            print(f"Gemini initialization failed: {e}")
            self.enabled = False
    
    def _test_model(self, model_name: str) -> bool:
        """
        Test if a specific Gemini model works
        
        Args:
            model_name: Name of the model to test
            
        Returns:
            True if model works, False otherwise
        """
        try:
            print(f"Testing model: {model_name}")
            
            # Create a test model instance
            test_model = genai.GenerativeModel(model_name)
            
            # Send a simple test prompt
            test_response = test_model.generate_content("Hello")
            
            # Check if we got a valid response
            if test_response and test_response.text:
                print(f"Model {model_name} test successful")
                return True
            else:
                print(f"Model {model_name} returned empty response")
                return False
                
        except Exception as e:
            # If any error occurs, this model doesn't work
            print(f"Model {model_name} test failed: {e}")
            return False
    
    async def get_ai_decision(self, agent_type: str, context: Dict[str, Any], 
                            framework_context: str = "") -> Dict[str, Any]:
        """
        Get an AI-enhanced decision from Gemini
        
        Args:
            agent_type: Type of agent making the decision ("pricing", "production", etc.)
            context: Market context and data for decision making
            framework_context: Information about which framework is calling
            
        Returns:
            Dictionary containing the AI decision or fallback decision
        """
        # If AI is enabled, try to get AI decision
        if self.enabled:
            try:
                print(f"Getting AI decision for {agent_type} agent...")
                return await self._make_ai_decision(agent_type, context, framework_context)
            except Exception as e:
                # If AI fails, fall back to rule-based decision
                print(f"AI decision failed for {agent_type}: {e}")
                print("Falling back to rule-based decision")
                return self._make_fallback_decision(agent_type, context)
        else:
            # If AI is disabled, use rule-based decision
            print(f"Using rule-based decision for {agent_type} (AI disabled)")
            return self._make_fallback_decision(agent_type, context)
    
    async def _make_ai_decision(self, agent_type: str, context: Dict[str, Any], 
                              framework_context: str) -> Dict[str, Any]:
        """
        Private method to get decision from Gemini AI
        
        Args:
            agent_type: Type of agent
            context: Market context
            framework_context: Framework information
            
        Returns:
            Parsed AI decision
        """
        # Create a specialized prompt for this agent type
        prompt = self._create_agent_prompt(agent_type, context, framework_context)
        
        # Call Gemini API in a separate thread to avoid blocking
        response = await asyncio.get_event_loop().run_in_executor(
            None,  # Use default thread pool
            lambda: self.model.generate_content(prompt)  # The blocking call to Gemini
        )
        
        # Validate that we got a response
        if not response or not response.text:
            raise ValueError("Empty response from Gemini AI")
        
        # Parse and return the AI response
        return self._parse_ai_response(response.text, agent_type)
    
    def _create_agent_prompt(self, agent_type: str, context: Dict[str, Any], 
                           framework_context: str) -> str:
        """
        Create a specialized prompt for each agent type
        
        Args:
            agent_type: Type of agent needing a decision
            context: Market context and data
            framework_context: Which framework is calling
            
        Returns:
            Formatted prompt string for Gemini AI
        """
        # Extract key information from context for the prompt
        inventory = context.get('inventory', 100)
        current_price = context.get('current_price', 10.0)
        competitor_prices = context.get('competitor_prices', [10.0])
        turn = context.get('turn', 1)
        
        # Create base context that all agents will receive
        base_context = f"""
BREWERY BUSINESS AGENT: {agent_type.upper()}
Framework Context: {framework_context}

CURRENT MARKET SITUATION:
- Current Inventory: {inventory} units
- Current Price: ${current_price:.2f}
- Competitor Prices: {competitor_prices}
- Game Turn: {turn}
- Sales History: {context.get('sales_history', [])[-5:]}

BUSINESS CONSTRAINTS:
- Price Range: ${AgentConfig.MIN_PRICE:.2f} - ${AgentConfig.MAX_PRICE:.2f}
- Production Range: {AgentConfig.MIN_PRODUCTION} - {AgentConfig.MAX_PRODUCTION} units  
- Marketing Range: ${AgentConfig.MIN_MARKETING} - ${AgentConfig.MAX_MARKETING}
- Production Cost: ${AgentConfig.UNIT_PRODUCTION_COST}/unit
- Holding Cost: ${AgentConfig.UNIT_HOLDING_COST}/unit
"""
        
        # Agent-specific prompt additions will be added by individual agent files
        return base_context
    
    def _parse_ai_response(self, response_text: str, agent_type: str) -> Dict[str, Any]:
        """
        Parse the AI response text and extract decision data
        
        Args:
            response_text: Raw text response from Gemini AI
            agent_type: Type of agent that made the request
            
        Returns:
            Parsed decision dictionary
        """
        try:
            # Use regex to find JSON in the response text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                # Parse the JSON string into a Python dictionary
                decision_data = json.loads(json_match.group())
            else:
                # If no JSON found, raise an error to trigger fallback
                raise ValueError("No JSON found in AI response")
            
            # Add metadata about this decision
            decision_data['agent_type'] = agent_type
            decision_data['ai_enhanced'] = True
            decision_data['source'] = 'gemini_ai'
            decision_data['model_used'] = self.model_name
            
            return decision_data
            
        except Exception as e:
            # If parsing fails, log the error and raise to trigger fallback
            print(f"Failed to parse AI response for {agent_type}: {e}")
            print(f"Raw response: {response_text[:200]}...")  # Show first 200 chars for debugging
            raise  # Re-raise the exception to trigger fallback logic
    
    def _make_fallback_decision(self, agent_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a rule-based fallback decision when AI is unavailable
        
        Args:
            agent_type: Type of agent needing a decision
            context: Market context for decision making
            
        Returns:
            Rule-based decision dictionary
        """
        # Extract key context information
        inventory = context.get('inventory', 100)
        competitor_prices = context.get('competitor_prices', [10.0])
        
        # Calculate average competitor price for competitive analysis
        avg_competitor_price = sum(competitor_prices) / len(competitor_prices) if competitor_prices else 10.0
        
        # Calculate inventory ratio to determine urgency (1.0 = normal, >1.0 = excess, <1.0 = shortage)
        inventory_ratio = inventory / 100.0  # Normalize against baseline of 100 units
        
        # Create basic fallback decision structure
        fallback_decision = {
            'agent_type': agent_type,
            'ai_enhanced': False,
            'source': 'rule_based_fallback',
            'confidence': 0.6,  # Lower confidence for rule-based decisions
            'reasoning': f'{agent_type.title()} Agent: Rule-based decision (AI unavailable)'
        }
        
        # This is a placeholder - specific agents will override this method
        # with their own specialized fallback logic
        return fallback_decision