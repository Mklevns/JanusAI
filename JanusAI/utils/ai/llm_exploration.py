# File: JanusAI/utils/ai/llm_exploration.py
"""
LLM-Driven Exploration Utilities

This module provides utilities for using Large Language Models to guide
symbolic discovery through high-level hypothesis generation.
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import asyncio
import aiohttp

# For different LLM backends
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None


@dataclass
class ExplorationContext:
    """Context for LLM-guided exploration."""
    domain: str  # e.g., "mechanics", "thermodynamics"
    variables: List[str]
    variable_descriptions: Dict[str, str]
    discovered_expressions: List[Dict[str, Any]]  # Recent discoveries
    failed_attempts: List[str]  # Recent failed expressions
    performance_history: List[float]
    current_focus: Optional[str] = None  # e.g., "conservation laws"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMGoalGenerator:
    """
    Generates exploration goals using Large Language Models.
    
    Features:
    - Multi-LLM support (OpenAI, Anthropic, local models)
    - Context-aware suggestions
    - Expression validation
    - Diversity enforcement
    - Adaptive prompting based on performance
    """
    
    def __init__(self,
                 model_name: str = "gpt-4",
                 api_type: str = "openai",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7,
                 max_history: int = 20,
                 diversity_penalty: float = 0.5):
        """
        Args:
            model_name: Name of the LLM model
            api_type: Type of API ("openai", "anthropic", "local")
            api_key: API key (if required)
            temperature: Sampling temperature
            max_history: Maximum history to maintain
            diversity_penalty: Penalty for suggesting similar expressions
        """
        self.model_name = model_name
        self.api_type = api_type
        self.temperature = temperature
        self.max_history = max_history
        self.diversity_penalty = diversity_penalty
        
        # Initialize API client
        if api_type == "openai" and openai:
            openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.client = openai
        elif api_type == "anthropic" and anthropic:
            self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        else:
            self.client = None
        
        # Suggestion history for diversity
        self.suggestion_history = deque(maxlen=max_history)
        self.prompt_templates = self._load_prompt_templates()
        
        # Performance tracking
        self.suggestion_performance = {}  # Maps suggestions to their outcomes
        
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load domain-specific prompt templates."""
        return {
            'mechanics': """You are an expert physicist helping discover fundamental laws of mechanics.

Context:
- Physical system: {context}
- Variables: {variables}
- Variable meanings: {variable_descriptions}
- Recent successful discoveries: {discoveries}
- Recent failed attempts: {failures}

Task: Suggest a new mathematical expression that could represent a physical law or relationship.

Guidelines:
1. Consider conservation laws (energy, momentum, angular momentum)
2. Look for relationships involving powers, products, and ratios
3. Include trigonometric functions for oscillatory systems
4. Consider both simple and composite relationships
5. The expression should be structurally different from recent attempts

Respond with ONLY the mathematical expression using the exact variable names provided.
Example format: m1*v1**2 + m2*v2**2""",

            'thermodynamics': """You are an expert thermodynamicist helping discover fundamental laws.

Context:
- System: {context}
- Variables: {variables}
- Variable meanings: {variable_descriptions}
- Recent discoveries: {discoveries}

Task: Suggest a mathematical relationship between the variables.

Consider:
1. Ideal gas relationships
2. Entropy and energy relationships
3. Logarithmic and exponential dependencies
4. Products and ratios of variables

Respond with ONLY the mathematical expression.""",

            'electromagnetism': """You are an expert in electromagnetism helping discover field relationships.

Context:
- System: {context}
- Variables: {variables}
- Recent patterns: {discoveries}

Suggest a new expression considering:
1. Inverse square laws
2. Cross products for magnetic fields
3. Symmetries and conservation laws

Respond with ONLY the mathematical expression.""",

            'general': """You are a mathematical physicist discovering symbolic relationships in data.

Variables: {variables}
Previous successful patterns: {discoveries}
Failed attempts: {failures}

Suggest a new mathematical expression that:
1. Uses the given variables
2. Is structurally different from previous attempts
3. Could represent a meaningful physical relationship

Respond with ONLY the expression."""
        }
    
    async def suggest_next_goal_async(self, context: ExplorationContext) -> str:
        """Asynchronously generate a goal expression."""
        prompt = self._build_prompt(context)
        
        try:
            if self.api_type == "openai":
                response = await self._call_openai_async(prompt)
            elif self.api_type == "anthropic":
                response = await self._call_anthropic_async(prompt)
            else:
                response = await self._call_local_model_async(prompt)
            
            # Extract and validate expression
            expression = self._extract_expression(response)
            
            # Check diversity
            if self._is_too_similar(expression):
                # Request another suggestion with explicit diversity instruction
                prompt += "\n\nIMPORTANT: The expression must be structurally very different from: " + \
                         ", ".join(list(self.suggestion_history)[-3:])
                response = await self._call_model_async(prompt)
                expression = self._extract_expression(response)
            
            # Validate expression
            if self._validate_expression(expression, context.variables):
                self.suggestion_history.append(expression)
                return expression
            else:
                # Fallback to a template-based suggestion
                return self._generate_fallback_expression(context)
                
        except Exception as e:
            print(f"Error in LLM goal generation: {e}")
            return self._generate_fallback_expression(context)
    
    def suggest_next_goal(self, context: ExplorationContext) -> str:
        """Synchronous wrapper for goal suggestion."""
        return asyncio.run(self.suggest_next_goal_async(context))
    
    def _build_prompt(self, context: ExplorationContext) -> str:
        """Build a context-aware prompt."""
        # Select appropriate template
        template = self.prompt_templates.get(
            context.domain, 
            self.prompt_templates['general']
        )
        
        # Prepare context components
        discoveries_str = self._format_discoveries(context.discovered_expressions)
        failures_str = ", ".join(context.failed_attempts[-5:]) if context.failed_attempts else "None"
        var_desc_str = "\n".join(f"- {var}: {desc}" for var, desc in context.variable_descriptions.items())
        
        # Fill template
        prompt = template.format(
            context=context.metadata.get('system_description', 'Unknown system'),
            variables=", ".join(context.variables),
            variable_descriptions=var_desc_str,
            discoveries=discoveries_str,
            failures=failures_str
        )
        
        # Add performance-based guidance
        if context.performance_history:
            recent_performance = np.mean(context.performance_history[-10:])
            if recent_performance < 0.3:
                prompt += "\n\nNote: Recent performance is low. Consider simpler expressions."
            elif recent_performance > 0.7:
                prompt += "\n\nNote: Recent performance is high. Try more complex, composite expressions."
        
        # Add focus area if specified
        if context.current_focus:
            prompt += f"\n\nCurrent focus: {context.current_focus}"
        
        return prompt
    
    def _format_discoveries(self, discoveries: List[Dict[str, Any]]) -> str:
        """Format discovered expressions for the prompt."""
        if not discoveries:
            return "None yet"
        
        formatted = []
        for disc in discoveries[-5:]:  # Last 5 discoveries
            expr = disc.get('expression', 'Unknown')
            reward = disc.get('reward', 0.0)
            formatted.append(f"{expr} (reward: {reward:.3f})")
        
        return "\n".join(formatted)
    
    async def _call_openai_async(self, prompt: str) -> str:
        """Call OpenAI API asynchronously."""
        if not self.client:
            raise ValueError("OpenAI client not initialized")
        
        response = await self.client.ChatCompletion.acreate(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a physics expert helping discover mathematical laws."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature,
            max_tokens=100
        )
        
        return response.choices[0].message.content
    
    async def _call_anthropic_async(self, prompt: str) -> str:
        """Call Anthropic API asynchronously."""
        if not self.client:
            raise ValueError("Anthropic client not initialized")
        
        response = await self.client.messages.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=100
        )
        
        return response.content[0].text
    
    async def _call_local_model_async(self, prompt: str) -> str:
        """Call a local model API (e.g., llama.cpp server)."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:8080/completion",  # Adjust URL as needed
                json={
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": 100
                }
            ) as response:
                result = await response.json()
                return result.get("content", "")
    
    def _extract_expression(self, response: str) -> str:
        """Extract mathematical expression from LLM response."""
        # Remove any explanatory text
        lines = response.strip().split('\n')
        
        # Look for lines that look like expressions
        for line in lines:
            # Remove common prefixes
            line = re.sub(r'^(Expression:|Answer:|Result:|Output:)\s*', '', line, flags=re.IGNORECASE)
            line = line.strip()
            
            # Check if it looks like a mathematical expression
            if re.search(r'[a-zA-Z_]\w*\s*[\+\-\*/\^]', line) or \
               re.search(r'(sin|cos|tan|log|exp|sqrt)\s*\(', line):
                # Clean up the expression
                expression = line.strip('`"\'')
                expression = expression.replace('^', '**')  # Convert to Python syntax
                return expression
        
        # If no clear expression found, return the cleaned response
        return response.strip().split('\n')[0].strip('`"\'')
    
    def _validate_expression(self, expression: str, variables: List[str]) -> bool:
        """Validate that the expression is syntactically correct."""
        try:
            # Check for basic syntax
            if not expression or len(expression) < 3:
                return False
            
            # Check that it contains at least one variable
            contains_var = any(var in expression for var in variables)
            if not contains_var:
                return False
            
            # Check for balanced parentheses
            paren_count = 0
            for char in expression:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                if paren_count < 0:
                    return False
            
            if paren_count != 0:
                return False
            
            # Check for valid operators and functions
            valid_patterns = [
                r'[\+\-\*/]',  # Basic operators
                r'\*\*',       # Power operator
                r'(sin|cos|tan|log|exp|sqrt|abs)\s*\(',  # Functions
                r'\d+\.?\d*',  # Numbers
            ]
            
            # Remove all valid patterns and variables, should be mostly empty
            test_expr = expression
            for var in variables:
                test_expr = test_expr.replace(var, '')
            for pattern in valid_patterns:
                test_expr = re.sub(pattern, '', test_expr)
            
            # Remove whitespace and parentheses
            test_expr = re.sub(r'[\s\(\)]', '', test_expr)
            
            # If much remains, it's likely invalid
            return len(test_expr) < len(expression) * 0.1
            
        except Exception:
            return False
    
    def _is_too_similar(self, expression: str) -> bool:
        """Check if expression is too similar to recent suggestions."""
        if len(self.suggestion_history) < 3:
            return False
        
        # Simple similarity check based on structure
        for recent in list(self.suggestion_history)[-3:]:
            # Calculate structural similarity
            similarity = self._calculate_similarity(expression, recent)
            if similarity > 0.8:
                return True
        
        return False
    
    def _calculate_similarity(self, expr1: str, expr2: str) -> float:
        """Calculate structural similarity between expressions."""
        # Extract operators and functions
        ops1 = re.findall(r'[\+\-\*/]|\*\*|sin|cos|tan|log|exp|sqrt', expr1)
        ops2 = re.findall(r'[\+\-\*/]|\*\*|sin|cos|tan|log|exp|sqrt', expr2)
        
        # Calculate Jaccard similarity
        if not ops1 and not ops2:
            return 1.0 if expr1 == expr2 else 0.0
        
        set1, set2 = set(ops1), set(ops2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _generate_fallback_expression(self, context: ExplorationContext) -> str:
        """Generate a fallback expression using templates."""
        templates = [
            "{v1} + {v2}",
            "{v1} * {v2}",
            "{v1}**2 + {v2}**2",
            "{v1} / {v2}",
            "sin({v1}) + cos({v2})",
            "log({v1}) * {v2}",
            "sqrt({v1}**2 + {v2}**2)",
            "{v1} * {v2} / {v3}" if len(context.variables) > 2 else "{v1} * {v2}",
        ]
        
        # Select a template that hasn't been used recently
        for template in templates:
            # Fill template with random variables
            var_mapping = {}
            vars_used = list(context.variables)
            np.random.shuffle(vars_used)
            
            for i, var_placeholder in enumerate(['v1', 'v2', 'v3']):
                if i < len(vars_used):
                    var_mapping[var_placeholder] = vars_used[i]
            
            try:
                expression = template.format(**var_mapping)
                if expression not in self.suggestion_history:
                    return expression
            except:
                continue
        
        # Ultimate fallback
        return f"{context.variables[0]} + {context.variables[1]}"
    
    def update_performance(self, expression: str, performance: float):
        """Update performance tracking for suggested expressions."""
        self.suggestion_performance[expression] = performance
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about suggestions and performance."""
        if not self.suggestion_performance:
            return {}
        
        performances = list(self.suggestion_performance.values())
        
        return {
            'total_suggestions': len(self.suggestion_history),
            'unique_suggestions': len(set(self.suggestion_history)),
            'mean_performance': np.mean(performances),
            'best_suggestion': max(self.suggestion_performance.items(), 
                                  key=lambda x: x[1])[0] if self.suggestion_performance else None,
            'diversity_score': len(set(self.suggestion_history)) / max(len(self.suggestion_history), 1)
        }


class AdaptiveLLMExploration:
    """
    Adaptive exploration strategy that combines LLM guidance with learned patterns.
    """
    
    def __init__(self,
                 goal_generator: LLMGoalGenerator,
                 grammar,
                 initial_exploration_rate: float = 0.3,
                 min_exploration_rate: float = 0.1,
                 decay_factor: float = 0.995):
        """
        Args:
            goal_generator: LLM goal generator instance
            grammar: Expression grammar for parsing
            initial_exploration_rate: Initial probability of using LLM goals
            min_exploration_rate: Minimum exploration rate
            decay_factor: Decay factor for exploration rate
        """
        self.goal_generator = goal_generator
        self.grammar = grammar
        self.exploration_rate = initial_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.decay_factor = decay_factor
        
        # Current goal tracking
        self.current_goal = None
        self.goal_steps_remaining = 0
        self.goal_start_time = None
        
        # Performance tracking
        self.goal_achievements = []
        self.random_achievements = []
        
    def should_use_llm_goal(self) -> bool:
        """Decide whether to use LLM guidance for this episode."""
        return np.random.random() < self.exploration_rate
    
    def set_new_goal(self, context: ExplorationContext, duration: int = 100):
        """Set a new LLM-generated goal."""
        goal_expr_str = self.goal_generator.suggest_next_goal(context)
        
        try:
            self.current_goal = self.grammar.parse(goal_expr_str)
            self.goal_steps_remaining = duration
            self.goal_start_time = time.time()
            return True
        except Exception as e:
            print(f"Failed to set goal: {e}")
            self.current_goal = None
            return False
    
    def update_exploration_rate(self):
        """Decay exploration rate over time."""
        self.exploration_rate = max(
            self.min_exploration_rate,
            self.exploration_rate * self.decay_factor
        )
    
    def record_achievement(self, reward: float, used_llm_goal: bool):
        """Record achievement for adaptive strategy."""
        if used_llm_goal:
            self.goal_achievements.append(reward)
        else:
            self.random_achievements.append(reward)
        
        # Adapt exploration rate based on relative performance
        if len(self.goal_achievements) > 10 and len(self.random_achievements) > 10:
            llm_mean = np.mean(self.goal_achievements[-20:])
            random_mean = np.mean(self.random_achievements[-20:])
            
            # Increase exploration if LLM is significantly better
            if llm_mean > random_mean * 1.2:
                self.exploration_rate = min(0.5, self.exploration_rate * 1.1)
            elif random_mean > llm_mean * 1.2:
                self.exploration_rate = max(self.min_exploration_rate, 
                                          self.exploration_rate * 0.9)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        stats = {
            'exploration_rate': self.exploration_rate,
            'total_goals_set': len(self.goal_achievements),
            'llm_goal_performance': np.mean(self.goal_achievements) if self.goal_achievements else 0,
            'random_performance': np.mean(self.random_achievements) if self.random_achievements else 0,
        }
        
        # Add goal generator stats
        stats.update(self.goal_generator.get_statistics())
        
        return stats