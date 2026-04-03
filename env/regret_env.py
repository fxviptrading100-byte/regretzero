import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
from gymnasium import spaces
import random


class RegretZeroEnv(gym.Env):
    """
    A Gymnasium-style OpenEnv environment for regret minimization in decision-making.
    
    This environment simulates realistic decision scenarios where agents must choose
    appropriate actions to minimize long-term regret. Each decision point includes
    contextual information and emotional states that influence outcomes.
    
    The environment models complex decision-making with 32-dimensional observations
    and 8 discrete actions representing different decision-making strategies.
    """
    
    # Action definitions
    ACTIONS = {
        0: "suggest_action",      # Propose a specific course of action
        1: "wait",               # Delay decision for more information
        2: "research",           # Gather more data/information
        3: "talk_to_someone",    # Consult with others for advice
        4: "reflect",            # Deep contemplation of options
        5: "act_immediately",    # Make immediate decision
        6: "delegate",           # Pass decision to someone else
        7: "gather_team"         # Bring stakeholders together
    }
    
    def __init__(self, max_steps: int = 50, seed: Optional[int] = None):
        """
        Initialize the RegretZero environment.
        
        Args:
            max_steps: Maximum number of steps per episode
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Environment parameters
        self.max_steps = max_steps
        self.current_step = 0
        
        # Define action space: 8 discrete decision-making actions
        self.action_space = spaces.Discrete(8)
        
        # Define observation space: 64-dimensional vector for richer representation
        # Decision context (24 dims): situational factors and scenario details
        # Emotional state (16 dims): internal states and psychological factors
        # Decision history (16 dims): past actions and their effectiveness
        # Progress indicators (8 dims): episode progress and goal achievement
        self.observation_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(64,), 
            dtype=np.float32
        )
        
        # Initialize state variables
        self.state = None
        self.decision_history = []
        self.regret_accumulator = 0.0
        self.base_regret_rate = 0.1
        self.optimal_regret_threshold = 0.5  # Target regret per action
        
        # Decision scenario parameters
        self.scenario_type = None
        self.difficulty_level = None
        self.stakes_level = None
        self.scenario_complexity = None
        
        # Decision quality tracking
        self.decision_quality_score = 0.0
        self.action_effectiveness = {i: 0.0 for i in range(8)}
        
        # Set seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options for reset
            
        Returns:
            observation: Initial observation (32-dim vector)
            info: Additional information dictionary
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Reset counters and history
        self.current_step = 0
        self.decision_history = []
        self.regret_accumulator = 0.0
        
        # Generate new decision scenario
        self._generate_scenario()
        
        # Generate initial state
        self.state = self._generate_state()
        
        # Prepare info dictionary
        info = {
            "step": self.current_step,
            "total_regret": self.regret_accumulator,
            "decision_count": len(self.decision_history),
            "scenario_type": self.scenario_type,
            "difficulty_level": self.difficulty_level,
            "stakes_level": self.stakes_level,
            "available_actions": list(self.ACTIONS.values())
        }
        
        return self.state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-7)
            
        Returns:
            observation: Next state observation
            reward: Reward signal (negative regret)
            terminated: Whether episode ended naturally
            truncated: Whether episode ended due to step limit
            info: Additional information
        """
        if self.state is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}. Must be in range [0, 7]")
        
        # Calculate regret and reward for this action
        immediate_regret = self._calculate_immediate_regret(action)
        long_term_regret = self._calculate_long_term_regret(action)
        total_regret = immediate_regret + long_term_regret
        
        # Reward is negative regret (we want to minimize regret)
        reward = -total_regret
        
        # Update state based on action and its consequences
        self.state = self._update_state(action)
        
        # Record decision in history
        decision_record = {
            "step": self.current_step,
            "action": action,
            "action_name": self.ACTIONS[action],
            "immediate_regret": immediate_regret,
            "long_term_regret": long_term_regret,
            "total_regret": total_regret,
            "state_snapshot": self.state.copy()
        }
        self.decision_history.append(decision_record)
        
        # Update accumulators
        self.regret_accumulator += total_regret
        self.current_step += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # Prepare info dictionary
        info = {
            "step": self.current_step,
            "total_regret": self.regret_accumulator,
            "decision_count": len(self.decision_history),
            "current_regret": total_regret,
            "avg_regret": self.regret_accumulator / max(1, len(self.decision_history)),
            "action_taken": self.ACTIONS[action],
            "scenario_progress": self._get_scenario_progress(),
            "termination_reason": self._get_termination_reason() if terminated else None
        }
        
        return self.state, reward, terminated, truncated, info
    
    def _generate_scenario(self):
        """Generate a random decision scenario with realistic parameters."""
        scenarios = [
            "career_decision", "financial_investment", "relationship_choice", 
            "business_strategy", "personal_health", "educational_path",
            "relocation_decision", "major_purchase", "ethical_dilemma",
            "team_conflict", "project_deadline", "resource_allocation"
        ]
        
        self.scenario_type = random.choice(scenarios)
        self.difficulty_level = np.random.choice(["easy", "medium", "hard"], p=[0.3, 0.5, 0.2])
        self.stakes_level = np.random.choice(["low", "medium", "high"], p=[0.4, 0.4, 0.2])
        
        # Adjust base regret rate based on scenario parameters
        difficulty_multiplier = {"easy": 0.5, "medium": 1.0, "hard": 1.5}[self.difficulty_level]
        stakes_multiplier = {"low": 0.7, "medium": 1.0, "high": 1.3}[self.stakes_level]
        self.base_regret_rate = 0.1 * difficulty_multiplier * stakes_multiplier
    
    def _generate_state(self) -> np.ndarray:
        """
        Generate a realistic 64-dimensional state vector.
        
        Returns:
            state: 64-dimensional observation vector
        """
        state = np.zeros(64, dtype=np.float32)
        
        # Decision context (24 dimensions)
        decision_context = self._generate_decision_context()
        state[0:24] = decision_context
        
        # Emotional state (16 dimensions)
        emotional_state = self._generate_emotional_state(decision_context)
        state[24:40] = emotional_state
        
        # Decision history and effectiveness (16 dimensions)
        history_context = self._generate_history_context()
        state[40:56] = history_context
        
        # Progress indicators (8 dimensions)
        progress_indicators = self._generate_progress_indicators()
        state[56:64] = progress_indicators
        
        return state
    
    def _generate_decision_context(self) -> np.ndarray:
        """Generate 24-dimensional decision context vector."""
        context = np.random.uniform(-0.3, 0.3, 24)
        
        # Core decision factors (8 dims)
        context[0] = np.random.uniform(0.2, 0.9)  # importance
        context[1] = np.random.uniform(0.1, 0.8)  # urgency
        context[2] = np.random.uniform(0.1, 0.9)  # complexity
        context[3] = np.random.uniform(0.1, 0.8)  # risk
        context[4] = np.random.uniform(-0.2, 0.8)  # opportunity
        context[5] = np.random.uniform(0.0, 0.7)  # resource_availability
        context[6] = np.random.uniform(0.0, 0.8)  # social_pressure
        context[7] = np.random.uniform(0.1, 0.9)  # uncertainty
        
        # Scenario-specific factors (8 dims)
        if self.scenario_type == "career_decision":
            context[8] = np.random.uniform(0.4, 0.9)  # growth_potential
            context[9] = np.random.uniform(0.2, 0.7)  # work_life_balance
            context[10] = np.random.uniform(0.3, 0.8)  # skill_match
            context[11] = np.random.uniform(0.1, 0.6)  # market_demand
        elif self.scenario_type == "financial_investment":
            context[8] = np.random.uniform(0.3, 0.9)  # return_potential
            context[9] = np.random.uniform(0.2, 0.8)  # liquidity
            context[10] = np.random.uniform(0.4, 0.9)  # time_horizon
            context[11] = np.random.uniform(0.1, 0.7)  # diversification
        elif self.scenario_type == "relationship_choice":
            context[8] = np.random.uniform(0.3, 0.8)  # compatibility
            context[9] = np.random.uniform(0.2, 0.7)  # trust_level
            context[10] = np.random.uniform(0.4, 0.8)  # shared_values
            context[11] = np.random.uniform(0.1, 0.6)  # future_potential
        
        # Environmental factors (8 dims)
        context[12] = np.random.uniform(0.1, 0.7)  # support_system
        context[13] = np.random.uniform(0.0, 0.6)  # external_pressure
        context[14] = np.random.uniform(0.2, 0.8)  # information_quality
        context[15] = np.random.uniform(0.1, 0.7)  # time_constraints
        context[16] = np.random.uniform(0.0, 0.5)  # emotional_load
        context[17] = np.random.uniform(0.2, 0.7)  # consequences
        context[18] = np.random.uniform(0.1, 0.6)  # reversibility
        context[19] = np.random.uniform(0.2, 0.8)  # learning_opportunity
        
        # Apply difficulty and stakes adjustments
        if self.difficulty_level == "hard":
            context[2] *= 1.3  # Increase complexity
            context[7] *= 1.4  # Increase uncertainty
            context[3] *= 1.2  # Increase risk
        
        if self.stakes_level == "high":
            context[0] *= 1.3  # Increase importance
            context[17] *= 1.4  # Increase consequences
            context[1] *= 1.2  # Increase urgency
        
        # Clip to valid range
        return np.clip(context, -1.0, 1.0)
    
    def _generate_emotional_state(self, decision_context: np.ndarray) -> np.ndarray:
        """Generate 16-dimensional emotional state influenced by decision context."""
        emotions = np.zeros(16, dtype=np.float32)
        
        # Core emotions (8 dims): [anxiety, confidence, clarity, motivation, stress, hope, fear, satisfaction]
        emotions[0] = (decision_context[1] + decision_context[3]) * 0.3 + np.random.normal(0, 0.15)
        emotions[1] = -(decision_context[2] + decision_context[7]) * 0.25 + np.random.normal(0, 0.25)
        emotions[2] = -decision_context[2] * 0.4 + np.random.normal(0, 0.15)
        emotions[3] = (decision_context[0] + decision_context[4]) * 0.25 + np.random.normal(0, 0.15)
        emotions[4] = (decision_context[1] + decision_context[6] + decision_context[16]) * 0.3 + np.random.normal(0, 0.15)
        emotions[5] = (decision_context[4] + emotions[1]) * 0.25 + np.random.normal(0, 0.15)
        emotions[6] = (decision_context[3] + emotions[0]) * 0.3 + np.random.normal(0, 0.15)
        emotions[7] = np.random.normal(0.1, 0.2)
        
        # Advanced psychological states (8 dims): [resilience, focus, creativity, patience, decisiveness, adaptability, wisdom, intuition]
        emotions[8] = -(emotions[0] + emotions[4]) * 0.2 + np.random.normal(0, 0.2)  # resilience
        emotions[9] = emotions[2] * 0.3 - emotions[0] * 0.2 + np.random.normal(0, 0.2)  # focus
        emotions[10] = emotions[3] * 0.2 + emotions[5] * 0.2 + np.random.normal(0, 0.2)  # creativity
        emotions[11] = -emotions[1] * 0.2 + np.random.normal(0, 0.2)  # patience
        emotions[12] = emotions[1] * 0.3 - emotions[7] * 0.1 + np.random.normal(0, 0.2)  # decisiveness
        emotions[13] = -decision_context[2] * 0.2 + emotions[8] * 0.2 + np.random.normal(0, 0.2)  # adaptability
        emotions[14] = emotions[2] * 0.3 + emotions[8] * 0.2 + np.random.normal(0, 0.2)  # wisdom
        emotions[15] = emotions[5] * 0.2 + emotions[10] * 0.2 + np.random.normal(0, 0.2)  # intuition
        
        return np.clip(emotions, -1.0, 1.0)
    
    def _generate_history_context(self) -> np.ndarray:
        """Generate 16-dimensional history context vector."""
        history = np.zeros(16, dtype=np.float32)
        
        if len(self.decision_history) == 0:
            # Initial state - no history
            history[0:8] = 0.0  # Action frequencies
            history[8:12] = 0.0  # Recent action effectiveness
            history[12:16] = 0.0  # Learning progress
        else:
            # Action frequencies (8 dims)
            recent_actions = [d["action"] for d in self.decision_history[-10:]]
            action_counts = [recent_actions.count(i) / max(1, len(recent_actions)) for i in range(8)]
            history[0:8] = action_counts
            
            # Recent action effectiveness (4 dims)
            if len(self.decision_history) >= 4:
                recent_regrets = [d["total_regret"] for d in self.decision_history[-4:]]
                avg_recent_regret = np.mean(recent_regrets)
                # Lower regret = higher effectiveness
                effectiveness = max(0, 1.0 - avg_recent_regret / self.optimal_regret_threshold)
                history[8] = effectiveness
                history[9] = 1.0 - np.std(recent_regrets) / (avg_recent_regret + 0.1)  # consistency
                history[10] = len([r for r in recent_regrets if r < self.optimal_regret_threshold]) / len(recent_regrets)  # success_rate
                history[11] = min(1.0, len(self.decision_history) / 10.0)  # experience_level
            
            # Learning progress (4 dims)
            history[12] = self.decision_quality_score
            history[13] = np.mean(list(self.action_effectiveness.values()))
            history[14] = len(self.decision_history) / self.max_steps  # episode_progress
            history[15] = max(0, 1.0 - self.regret_accumulator / (self.current_step + 1))  # overall_performance
        
        return np.clip(history, -1.0, 1.0)
    
    def _generate_progress_indicators(self) -> np.ndarray:
        """Generate 8-dimensional progress indicators vector."""
        progress = np.zeros(8, dtype=np.float32)
        
        # Episode progress
        progress[0] = self.current_step / self.max_steps
        progress[1] = min(1.0, len(self.decision_history) / 5.0)  # decision_made_progress
        
        # Decision quality indicators
        if len(self.decision_history) > 0:
            avg_regret = self.regret_accumulator / len(self.decision_history)
            progress[2] = max(0, 1.0 - avg_regret / self.optimal_regret_threshold)  # regret_minimization_progress
            progress[3] = self.decision_quality_score
        
        # Scenario resolution progress
        if self.state is not None:
            confidence = self.state[25]  # confidence index in emotional state
            uncertainty = self.state[31]  # uncertainty index in decision context
            clarity = self.state[26]  # clarity index in emotional state
            
            progress[4] = confidence
            progress[5] = 1.0 - uncertainty
            progress[6] = clarity
            progress[7] = (confidence + (1.0 - uncertainty) + clarity) / 3.0  # overall_readiness
        
        return np.clip(progress, -1.0, 1.0)
    
    def _calculate_immediate_regret(self, action: int) -> float:
        """Calculate immediate regret based on action appropriateness."""
        if self.state is None:
            return 0.0
        
        decision_context = self.state[0:24]
        emotional_state = self.state[24:40]
        
        # Extract key factors from enhanced context
        urgency = decision_context[1]
        importance = decision_context[0]
        complexity = decision_context[2]
        risk = decision_context[3]
        opportunity = decision_context[4]
        social_pressure = decision_context[6]
        uncertainty = decision_context[7]
        confidence = emotional_state[1]
        stress = emotional_state[4]
        clarity = emotional_state[2]
        wisdom = emotional_state[14]
        
        # Base regret calculation
        regret = self.base_regret_rate
        
        # Enhanced action-specific regret calculations with more nuanced logic
        if action == 0:  # suggest_action
            if urgency > 0.6 and complexity > 0.5:
                regret += 0.4
            if confidence < -0.2:
                regret += 0.3
            if risk > 0.7 and uncertainty > 0.5:
                regret += 0.5
            if clarity < 0.3:
                regret += 0.2
            if wisdom > 0.5:
                regret -= 0.2  # Wise suggestions reduce regret
                
        elif action == 1:  # wait
            if urgency > 0.7:
                regret += 0.6
            if uncertainty < 0.2 and importance > 0.6:
                regret += 0.4
            if uncertainty > 0.6 and urgency < 0.3:
                regret -= 0.3
            if stress > 0.6:
                regret -= 0.2  # Waiting reduces stress
                
        elif action == 2:  # research
            if urgency > 0.8:
                regret += 0.5
            if complexity > 0.6 and uncertainty > 0.4:
                regret -= 0.4
            if risk > 0.7:
                regret -= 0.3
            if clarity > 0.5:
                regret += 0.2  # Over-researching when clear
                
        elif action == 3:  # talk_to_someone
            if social_pressure > 0.6:
                regret -= 0.4
            if stress > 0.7:
                regret -= 0.3
            if confidence > 0.5 and risk < 0.3:
                regret += 0.3
            if wisdom < 0.2:
                regret -= 0.2  # Seeking help when not wise
                
        elif action == 4:  # reflect
            if urgency > 0.7:
                regret += 0.5
            if stress > 0.5 and complexity > 0.4:
                regret -= 0.4
            if confidence < -0.3:
                regret -= 0.3
            if wisdom > 0.6:
                regret -= 0.2  # Reflection is valuable for wise people
                
        elif action == 5:  # act_immediately
            if uncertainty > 0.6:
                regret += 0.7
            if complexity > 0.7:
                regret += 0.6
            if confidence > 0.6 and urgency > 0.5:
                regret -= 0.5
            if clarity > 0.7 and wisdom > 0.5:
                regret -= 0.3  # Clear, wise immediate action
                
        elif action == 6:  # delegate
            if importance > 0.7 and confidence < 0:
                regret -= 0.4
            if social_pressure > 0.8:
                regret -= 0.3
            if opportunity > 0.6 and confidence > 0.4:
                regret += 0.4
            if wisdom > 0.5:
                regret -= 0.2  # Wise delegation
                
        elif action == 7:  # gather_team
            if social_pressure > 0.6 and complexity > 0.5:
                regret -= 0.5
            if urgency > 0.8:
                regret += 0.4
            if risk > 0.7:
                regret -= 0.3
            if importance > 0.8:
                regret -= 0.2  # Important decisions benefit from teams
        
        # Add contextual modifiers
        if self.stakes_level == "high":
            regret *= 1.3
        if self.difficulty_level == "hard":
            regret *= 1.2
            
        # Add noise for realism
        regret += np.random.normal(0, 0.03)
        
        return max(0.0, regret)
    
    def _calculate_long_term_regret(self, action: int) -> float:
        """Calculate long-term regret consequences of actions."""
        long_term_regret = 0.0
        
        # Enhanced long-term consequences based on scenario and action patterns
        if action == 5:  # act_immediately - can cause future regret if rushed
            if self.current_step < 3:  # Very early in decision process
                long_term_regret += 0.3
            if self.scenario_type in ["career_decision", "financial_investment"]:
                long_term_regret += 0.2  # High-stakes decisions
                
        elif action == 6:  # delegate - might regret not taking ownership
            if self.stakes_level == "high":
                long_term_regret += 0.15
            if self.scenario_type == "relationship_choice":
                long_term_regret += 0.2  # Personal decisions
                
        elif action == 1:  # wait - might regret missed opportunities
            if self.state is not None and self.state[4] > 0.6:  # High opportunity
                long_term_regret += 0.2
            if self.scenario_type == "financial_investment":
                long_term_regret += 0.15  # Time-sensitive opportunities
        
        # Pattern-based long-term regret
        if len(self.decision_history) > 5:
            recent_actions = [d["action"] for d in self.decision_history[-5:]]
            # Too much waiting
            if recent_actions.count(1) >= 3 and action == 1:
                long_term_regret += 0.1
            # Too much delegation
            if recent_actions.count(6) >= 2 and action == 6:
                long_term_regret += 0.15
        
        return long_term_regret
    
    def _update_state(self, action: int) -> np.ndarray:
        """Update state based on action taken."""
        new_state = self.state.copy()
        decision_context = new_state[0:24]
        emotional_state = new_state[24:40]
        history_context = new_state[40:56]
        progress_indicators = new_state[56:64]
        
        # Update emotional state based on action
        if action == 0:  # suggest_action
            emotional_state[1] += 0.1  # Increase confidence
            emotional_state[4] += 0.1  # Increase stress
            emotional_state[12] += 0.05  # Increase decisiveness
        elif action == 1:  # wait
            emotional_state[4] -= 0.15  # Decrease stress
            emotional_state[2] += 0.15  # Increase clarity
            emotional_state[11] += 0.1  # Increase patience
        elif action == 2:  # research
            emotional_state[2] += 0.2  # Increase clarity
            decision_context[7] -= 0.2  # Decrease uncertainty
            emotional_state[9] += 0.1  # Increase focus
        elif action == 3:  # talk_to_someone
            emotional_state[6] -= 0.2  # Decrease fear
            decision_context[6] -= 0.2  # Decrease social pressure
            emotional_state[8] += 0.1  # Increase resilience
        elif action == 4:  # reflect
            emotional_state[2] += 0.25  # Increase clarity
            emotional_state[0] -= 0.2  # Decrease anxiety
            emotional_state[14] += 0.15  # Increase wisdom
        elif action == 5:  # act_immediately
            emotional_state[1] += 0.2  # Increase confidence
            emotional_state[4] += 0.25  # Increase stress
            emotional_state[12] += 0.15  # Increase decisiveness
        elif action == 6:  # delegate
            emotional_state[4] -= 0.2  # Decrease stress
            decision_context[6] -= 0.3  # Decrease social pressure
            emotional_state[13] += 0.1  # Increase adaptability
        elif action == 7:  # gather_team
            decision_context[6] -= 0.4  # Decrease social pressure
            emotional_state[3] += 0.2  # Increase motivation
            emotional_state[10] += 0.1  # Increase creativity
        
        # Natural evolution of decision context
        decision_context[1] += 0.02  # Urgency naturally increases
        decision_context[4] -= 0.01  # Opportunities may decrease
        decision_context[7] *= 0.98  # Uncertainty slowly decreases
        decision_context[15] += 0.01  # Time constraints increase
        
        # Update action effectiveness tracking
        if len(self.decision_history) > 0:
            last_regret = self.decision_history[-1]["total_regret"]
            effectiveness = max(0, 1.0 - last_regret / self.optimal_regret_threshold)
            self.action_effectiveness[action] = 0.7 * self.action_effectiveness[action] + 0.3 * effectiveness
        
        # Update decision quality score
        if len(self.decision_history) > 0:
            recent_regrets = [d["total_regret"] for d in self.decision_history[-5:]]
            avg_recent_regret = np.mean(recent_regrets)
            self.decision_quality_score = max(0, 1.0 - avg_recent_regret / self.optimal_regret_threshold)
        
        # Add some random evolution
        decision_context += np.random.normal(0, 0.008, 24)
        emotional_state += np.random.normal(0, 0.015, 16)
        
        # Clip to valid range
        new_state[0:24] = np.clip(decision_context, -1.0, 1.0)
        new_state[24:40] = np.clip(emotional_state, -1.0, 1.0)
        new_state[40:56] = np.clip(history_context, -1.0, 1.0)
        new_state[56:64] = np.clip(progress_indicators, -1.0, 1.0)
        
        return new_state
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate naturally."""
        # Terminate if regret gets too high (decision maker gives up)
        if self.regret_accumulator > 6.0:
            return True
        
        # Enhanced termination conditions based on decision readiness
        if len(self.decision_history) >= 3:
            if self.state is not None:
                confidence = self.state[25]  # confidence index in emotional state
                uncertainty = self.state[31]  # uncertainty index in decision context
                clarity = self.state[26]  # clarity index in emotional state
                wisdom = self.state[38]  # wisdom index in emotional state
                
                # Strong decision readiness
                readiness_score = (confidence + (1.0 - uncertainty) + clarity + wisdom) / 4.0
                if readiness_score > 0.8:
                    return True
                
                # Good decision quality with sufficient experience
                if (self.decision_quality_score > 0.7 and 
                    len(self.decision_history) >= 5 and 
                    uncertainty < 0.3):
                    return True
        
        # Terminate if optimal decision path found (low regret over multiple steps)
        if len(self.decision_history) >= 4:
            recent_regrets = [d["total_regret"] for d in self.decision_history[-4:]]
            if all(r < self.optimal_regret_threshold * 0.8 for r in recent_regrets):
                return True
        
        return False
    
    def _get_termination_reason(self) -> str:
        """Get reason for episode termination."""
        if self.regret_accumulator > 6.0:
            return "high_regret"
        elif len(self.decision_history) >= 3 and self.state is not None:
            confidence = self.state[25]
            uncertainty = self.state[31]
            clarity = self.state[26]
            wisdom = self.state[38]
            
            readiness_score = (confidence + (1.0 - uncertainty) + clarity + wisdom) / 4.0
            if readiness_score > 0.8:
                return "decision_resolved_ready"
            
            if (self.decision_quality_score > 0.7 and 
                len(self.decision_history) >= 5 and 
                uncertainty < 0.3):
                return "decision_resolved_quality"
                
            if len(self.decision_history) >= 4:
                recent_regrets = [d["total_regret"] for d in self.decision_history[-4:]]
                if all(r < self.optimal_regret_threshold * 0.8 for r in recent_regrets):
                    return "decision_resolved_optimal"
        return "unknown"
    
    def render(self, mode: str = 'human'):
        """Render the environment state."""
        if self.state is None:
            print("Environment not initialized. Call reset() first.")
            return
        
        decision_context = self.state[0:24]
        emotional_state = self.state[24:40]
        progress_indicators = self.state[56:64]
        
        print(f"\n{'='*60}")
        print(f"STEP {self.current_step} - {self.scenario_type.upper()} ({self.difficulty_level}, {self.stakes_level} stakes)")
        print(f"{'='*60}")
        print(f"Total Regret: {self.regret_accumulator:.3f} | Decision Quality: {self.decision_quality_score:.3f}")
        print(f"Decision Context: urgency={decision_context[1]:.2f}, "
              f"importance={decision_context[0]:.2f}, complexity={decision_context[2]:.2f}, "
              f"uncertainty={decision_context[7]:.2f}")
        print(f"Emotional State: confidence={emotional_state[1]:.2f}, "
              f"stress={emotional_state[4]:.2f}, anxiety={emotional_state[0]:.2f}, "
              f"wisdom={emotional_state[14]:.2f}")
        print(f"Progress: episode={progress_indicators[0]:.2f}, "
              f"readiness={progress_indicators[7]:.2f}, "
              f"regret_min={progress_indicators[2]:.2f}")
        
        if len(self.decision_history) > 0:
            last_action = self.decision_history[-1]
            print(f"Last Action: {last_action['action_name']} (regret: {last_action['total_regret']:.3f})")
    
    def get_action_meanings(self) -> List[str]:
        """Get human-readable descriptions of actions."""
        return list(self.ACTIONS.values())
    
    def _get_scenario_progress(self) -> float:
        """
        Calculate scenario progress based on current state.
        
        Returns:
            Progress value between 0.0 and 1.0
        """
        # Progress based on episode completion and decision quality
        episode_progress = self.current_step / self.max_steps
        
        # Consider regret minimization progress
        if len(self.decision_history) > 0:
            avg_regret = np.mean([d['total_regret'] for d in self.decision_history])
            regret_progress = max(0.0, 1.0 - avg_regret)  # Lower regret = higher progress
        else:
            regret_progress = 0.5  # Neutral start
        
        # Combine progress metrics
        total_progress = (episode_progress * 0.6 + regret_progress * 0.4)
        return np.clip(total_progress, 0.0, 1.0)
    
    def _get_termination_reason(self) -> str:
        """
        Get the reason for episode termination.
        
        Returns:
            String describing termination reason
        """
        if self.current_step >= self.max_steps:
            return "Maximum steps reached"
        elif hasattr(self, '_terminated_early'):
            return "Early termination - decision resolved"
        else:
            return "Unknown reason"
    
    def close(self):
        """Clean up the environment."""
        pass
