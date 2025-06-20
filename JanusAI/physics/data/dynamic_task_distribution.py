# File: JanusAI/physics/data/dynamic_task_distribution.py
"""
Enhanced PhysicsTaskDistribution with dynamic task generation capabilities
for self-play curriculum learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib
import json

from janus.physics.data.task_distribution import PhysicsTask, PhysicsTaskDistribution


@dataclass
class DynamicPhysicsTask(PhysicsTask):
    """Extended PhysicsTask that supports parameterized generation"""
    
    # Generator function that creates data given parameters
    data_generator: Optional[Callable] = None
    
    # Parameter schema defining valid ranges
    parameter_schema: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Unique ID for tracking
    task_id: str = ""
    
    def __post_init__(self):
        if not self.task_id:
            # Generate unique ID from parameters
            param_str = json.dumps(self.parameters, sort_keys=True)
            self.task_id = hashlib.md5(param_str.encode()).hexdigest()[:8]
    
    def generate_data_with_params(self, n_samples: int, params: Dict[str, float], 
                                 noise: bool = True) -> np.ndarray:
        """Generate data with custom parameters"""
        if self.data_generator:
            return self.data_generator(n_samples, params, noise)
        else:
            # Fall back to parent class method
            return super().generate_data(n_samples, noise)


class DynamicPhysicsTaskDistribution(PhysicsTaskDistribution):
    """
    Enhanced task distribution that supports dynamic task generation
    for self-play curriculum learning.
    """
    
    def __init__(self, include_noise: bool = True):
        super().__init__(include_noise)
        
        # Task generators for different physics domains
        self.task_generators = {
            'harmonic_oscillator': self._generate_harmonic_oscillator_task,
            'pendulum': self._generate_pendulum_task,
            'double_pendulum': self._generate_double_pendulum_task,
            'spring_pendulum': self._generate_spring_pendulum_task,
            'coupled_oscillators': self._generate_coupled_oscillators_task
        }
        
        # Cache for generated tasks
        self.generated_tasks_cache = {}
        
        # Performance tracking for adaptive difficulty
        self.task_performance_history = defaultdict(list)
        
    def create_custom_task(self, task_type: str, parameters: Dict[str, float]) -> DynamicPhysicsTask:
        """
        Create a custom physics task with specified parameters.
        
        Args:
            task_type: Type of physics system
            parameters: Dictionary of physical parameters
            
        Returns:
            DynamicPhysicsTask instance
        """
        if task_type not in self.task_generators:
            raise ValueError(f"Unknown task type: {task_type}")
        
        return self.task_generators[task_type](parameters)
    
    def _generate_harmonic_oscillator_task(self, params: Dict[str, float]) -> DynamicPhysicsTask:
        """Generate a harmonic oscillator task with custom parameters"""
        
        def data_generator(n_samples: int, p: Dict[str, float], noise: bool) -> np.ndarray:
            m = p.get('m1', 1.0)
            k = p.get('k', 10.0)
            b = p.get('b', 0.0)  # Damping
            
            omega = np.sqrt(k / m)
            
            # Generate time series
            t_max = 4 * np.pi / omega
            t = np.linspace(0, t_max, n_samples)
            
            # Initial conditions
            x0 = np.random.uniform(-2, 2)
            v0 = np.random.uniform(-5, 5)
            
            if b == 0:  # Undamped
                x = x0 * np.cos(omega * t) + (v0 / omega) * np.sin(omega * t)
                v = -x0 * omega * np.sin(omega * t) + v0 * np.cos(omega * t)
            else:  # Damped
                gamma = b / (2 * m)
                omega_d = np.sqrt(omega**2 - gamma**2)
                
                if omega_d > 0:  # Underdamped
                    A = x0
                    B = (v0 + gamma * x0) / omega_d
                    x = np.exp(-gamma * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
                    v = np.gradient(x, t)
                else:  # Overdamped
                    r1 = -gamma + np.sqrt(gamma**2 - omega**2)
                    r2 = -gamma - np.sqrt(gamma**2 - omega**2)
                    A = (v0 - r2 * x0) / (r1 - r2)
                    B = x0 - A
                    x = A * np.exp(r1 * t) + B * np.exp(r2 * t)
                    v = np.gradient(x, t)
            
            # Calculate derived quantities
            a = np.gradient(v, t)
            KE = 0.5 * m * v**2
            PE = 0.5 * k * x**2
            E_total = KE + PE
            
            data = np.column_stack([x, v, a, t, m, k, b, E_total])
            
            if noise and self.include_noise:
                noise_level = 0.02 * np.std(data, axis=0)
                data += np.random.normal(0, noise_level, data.shape)
            
            return data
        
        # Calculate difficulty based on parameters
        m = params.get('m1', 1.0)
        k = params.get('k', 10.0)
        b = params.get('b', 0.0)
        
        # Difficulty increases with damping and extreme parameter values
        difficulty = 0.3 + 0.2 * (b / 2.0) + 0.1 * abs(np.log10(k/10)) + 0.1 * abs(np.log10(m))
        difficulty = np.clip(difficulty, 0.1, 0.9)
        
        return DynamicPhysicsTask(
            name=f"harmonic_oscillator_m{m:.1f}_k{k:.1f}_b{b:.2f}",
            domain="mechanics",
            difficulty=difficulty,
            variables=["x", "v", "a", "t", "m", "k", "b", "E_total"],
            parameters=params,
            conserved_quantities=["energy"] if b == 0 else [],
            symmetries=["time_translation"],
            data_generator=data_generator,
            parameter_schema={
                'm1': (0.1, 10.0),
                'k': (0.5, 50.0),
                'b': (0.0, 2.0)
            }
        )
    
    def _generate_pendulum_task(self, params: Dict[str, float]) -> DynamicPhysicsTask:
        """Generate a pendulum task with custom parameters"""
        
        def data_generator(n_samples: int, p: Dict[str, float], noise: bool) -> np.ndarray:
            m = p.get('m1', 1.0)
            L = p.get('L1', 1.0)
            g = p.get('g', 9.81)
            b = p.get('b', 0.0)  # Damping
            
            # Time array
            t_max = 10 * np.sqrt(L / g)
            t = np.linspace(0, t_max, n_samples)
            dt = t[1] - t[0]
            
            # Initial conditions
            theta0 = np.random.uniform(-np.pi/3, np.pi/3)
            omega0 = np.random.uniform(-2, 2)
            
            # Integrate equations of motion
            theta = np.zeros(n_samples)
            omega = np.zeros(n_samples)
            theta[0] = theta0
            omega[0] = omega0
            
            for i in range(1, n_samples):
                # Angular acceleration with damping
                alpha = -(g/L) * np.sin(theta[i-1]) - (b/(m*L**2)) * omega[i-1]
                
                # Update using Euler method
                omega[i] = omega[i-1] + alpha * dt
                theta[i] = theta[i-1] + omega[i-1] * dt
            
            # Calculate Cartesian coordinates and energy
            x = L * np.sin(theta)
            y = -L * np.cos(theta)
            
            KE = 0.5 * m * L**2 * omega**2
            PE = m * g * L * (1 - np.cos(theta))
            E_total = KE + PE
            
            data = np.column_stack([theta, omega, x, y, t, m, L, g, b, E_total])
            
            if noise and self.include_noise:
                noise_level = 0.02 * np.std(data, axis=0)
                data += np.random.normal(0, noise_level, data.shape)
            
            return data
        
        # Calculate difficulty
        L = params.get('L1', 1.0)
        g = params.get('g', 9.81)
        b = params.get('b', 0.0)
        
        difficulty = 0.4 + 0.1 * abs(np.log10(L)) + 0.1 * abs(np.log10(g/9.81)) + 0.2 * (b / 1.0)
        difficulty = np.clip(difficulty, 0.2, 0.9)
        
        return DynamicPhysicsTask(
            name=f"pendulum_L{L:.1f}_g{g:.1f}_b{b:.2f}",
            domain="mechanics",
            difficulty=difficulty,
            variables=["theta", "omega", "x", "y", "t", "m", "L", "g", "b", "E_total"],
            parameters=params,
            conserved_quantities=["energy", "angular_momentum"] if b == 0 else [],
            symmetries=["time_translation"],
            data_generator=data_generator,
            parameter_schema={
                'm1': (0.1, 10.0),
                'L1': (0.1, 5.0),
                'g': (1.0, 20.0),
                'b': (0.0, 2.0)
            }
        )
    
    def _generate_double_pendulum_task(self, params: Dict[str, float]) -> DynamicPhysicsTask:
        """Generate a double pendulum task with custom parameters"""
        
        def data_generator(n_samples: int, p: Dict[str, float], noise: bool) -> np.ndarray:
            m1 = p.get('m1', 1.0)
            m2 = p.get('m2', 1.0)
            L1 = p.get('L1', 1.0)
            L2 = p.get('L2', 1.0)
            g = p.get('g', 9.81)
            
            # This is a simplified version - full implementation would use RK4
            # For now, generate synthetic data that captures key features
            t = np.linspace(0, 20, n_samples)
            
            # Generate chaotic-looking trajectories
            freq1 = np.sqrt(g/L1) * (1 + 0.1 * np.random.randn())
            freq2 = np.sqrt(g/L2) * (1 + 0.1 * np.random.randn())
            
            theta1 = 0.5 * np.sin(freq1 * t) * np.exp(-0.01 * t)
            theta2 = 0.3 * np.sin(freq2 * t + np.pi/4) * np.exp(-0.01 * t)
            
            omega1 = np.gradient(theta1, t)
            omega2 = np.gradient(theta2, t)
            
            # Energy (approximate)
            E_total = 0.5 * (m1 + m2) * L1**2 * omega1**2 + 0.5 * m2 * L2**2 * omega2**2
            
            data = np.column_stack([theta1, theta2, omega1, omega2, m1, m2, L1, L2, g, E_total])
            
            if noise and self.include_noise:
                noise_level = 0.02 * np.std(data, axis=0)
                data += np.random.normal(0, noise_level, data.shape)
            
            return data
        
        # Double pendulum is inherently difficult due to chaos
        difficulty = 0.7 + 0.1 * np.random.rand()
        
        return DynamicPhysicsTask(
            name=f"double_pendulum_custom_{params.get('m1', 1):.1f}",
            domain="mechanics",
            difficulty=difficulty,
            variables=["theta1", "theta2", "omega1", "omega2", "m1", "m2", "L1", "L2", "g", "E_total"],
            parameters=params,
            conserved_quantities=["energy"],
            symmetries=[],
            data_generator=data_generator,
            parameter_schema={
                'm1': (0.1, 10.0),
                'm2': (0.1, 10.0),
                'L1': (0.1, 5.0),
                'L2': (0.1, 5.0),
                'g': (1.0, 20.0)
            }
        )
    
    def _generate_spring_pendulum_task(self, params: Dict[str, float]) -> DynamicPhysicsTask:
        """Generate a spring pendulum task with custom parameters"""
        
        # Similar implementation pattern...
        difficulty = 0.6 + 0.2 * np.random.rand()
        
        return DynamicPhysicsTask(
            name=f"spring_pendulum_custom",
            domain="mechanics",
            difficulty=difficulty,
            variables=["r", "theta", "v_r", "omega", "m", "k", "L0", "g", "E_total"],
            parameters=params,
            conserved_quantities=["energy"],
            symmetries=["time_translation"],
            parameter_schema={
                'm1': (0.1, 10.0),
                'k': (0.5, 50.0),
                'L0': (0.5, 5.0),
                'g': (1.0, 20.0)
            }
        )
    
    def _generate_coupled_oscillators_task(self, params: Dict[str, float]) -> DynamicPhysicsTask:
        """Generate coupled oscillators task with custom parameters"""
        
        # Similar implementation pattern...
        difficulty = 0.5 + 0.3 * np.random.rand()
        
        return DynamicPhysicsTask(
            name=f"coupled_oscillators_custom",
            domain="mechanics",
            difficulty=difficulty,
            variables=["x1", "x2", "v1", "v2", "m1", "m2", "k1", "k2", "k_coupling", "E_total"],
            parameters=params,
            conserved_quantities=["energy"],
            symmetries=["time_translation"],
            parameter_schema={
                'm1': (0.1, 10.0),
                'm2': (0.1, 10.0),
                'k1': (0.5, 50.0),
                'k2': (0.5, 50.0),
                'k_coupling': (0.1, 10.0)
            }
        )
    
    def update_task_performance(self, task_id: str, performance: Dict[str, float]):
        """
        Update performance history for a task.
        Used by the curriculum to adapt difficulty.
        """
        self.task_performance_history[task_id].append(performance)
    
    def get_task_difficulty_estimate(self, task_id: str) -> float:
        """
        Estimate task difficulty based on historical performance.
        """
        if task_id not in self.task_performance_history:
            return 0.5  # Default middle difficulty
        
        performances = self.task_performance_history[task_id]
        if len(performances) < 3:
            return 0.5
        
        # Average success rate inversely correlates with difficulty
        avg_success = np.mean([p.get('success_rate', 0.5) for p in performances[-10:]])
        estimated_difficulty = 1.0 - avg_success
        
        return np.clip(estimated_difficulty, 0.1, 0.9)
    
    def sample_task_batch_dynamic(self, n_tasks: int, 
                                 setter_agent=None,
                                 performance_history=None) -> List[DynamicPhysicsTask]:
        """
        Sample tasks using the TaskSetter agent if available,
        otherwise fall back to standard sampling.
        """
        if setter_agent and performance_history:
            # Use TaskSetter to generate tasks
            tasks = []
            for _ in range(n_tasks):
                # Get task parameters from setter
                observation = self._create_setter_observation(performance_history)
                task_params = setter_agent.propose_task(observation)
                
                # Create task
                task_type = self._get_task_type_from_params(task_params)
                task = self.create_custom_task(task_type, task_params)
                tasks.append(task)
            
            return tasks
        else:
            # Fall back to parent class sampling
            return super().sample_task_batch(n_tasks, curriculum=True)
    
    def _create_setter_observation(self, performance_history: List[Dict]) -> np.ndarray:
        """Create observation for TaskSetter from performance history"""
        # This would match the observation space of TaskSetterEnv
        # Simplified version here
        if performance_history:
            recent = performance_history[-20:]
            success_rates = [p.get('success_rate', 0.5) for p in recent]
            obs = [
                np.mean(success_rates),
                np.std(success_rates),
                len(recent) / 20.0
            ]
        else:
            obs = [0.5, 0.0, 0.0]
        
        # Pad to expected size
        while len(obs) < 50:
            obs.append(0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_task_type_from_params(self, params: Dict[str, float]) -> str:
        """Determine task type from parameters"""
        task_types = [
            "harmonic_oscillator",
            "pendulum", 
            "double_pendulum",
            "spring_pendulum",
            "coupled_oscillators"
        ]
        
        # Use task_type parameter if available
        if 'task_type' in params:
            idx = int(params['task_type'])
            return task_types[idx % len(task_types)]
        
        # Otherwise, make an educated guess based on parameters
        if 'm2' in params and 'L2' in params:
            return "double_pendulum"
        elif 'k' in params and 'L1' in params:
            return "spring_pendulum"
        elif 'k_coupling' in params:
            return "coupled_oscillators"
        elif 'L1' in params:
            return "pendulum"
        else:
            return "harmonic_oscillator"