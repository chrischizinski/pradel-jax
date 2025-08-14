"""
Optimization Orchestrator - High-level coordination of optimization strategies.

Provides the main interface for the optimization framework, coordinating:
- Strategy selection and execution
- Performance monitoring and logging
- Fallback and recovery mechanisms
- Experiment tracking and analysis

Follows enterprise patterns from:
- Apache Airflow for workflow orchestration
- Kubernetes for resource management
- Circuit breaker patterns for resilience
- Event-driven architecture for monitoring
"""

import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path

from .strategy import (
    StrategySelector, OptimizationStrategy, OptimizationConfig,
    StrategyRecommendation, ModelContext
)
from .optimizers import (
    create_optimizer, OptimizationResult, minimize_with_strategy
)
from .monitoring import (
    PerformanceMonitor, ExperimentTracker, OptimizationProfiler,
    OptimizationMetrics, start_monitoring_session, log_optimization_metrics,
    end_monitoring_session, optimization_experiment
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizationRequest:
    """Complete specification for an optimization request."""
    objective_function: Callable
    initial_parameters: np.ndarray
    bounds: Optional[List[tuple]] = None
    gradient_function: Optional[Callable] = None
    hessian_function: Optional[Callable] = None
    
    # Strategy preferences
    preferred_strategy: Optional[OptimizationStrategy] = None
    user_preferences: Optional[Dict[str, Any]] = None
    config_overrides: Optional[Dict[str, Any]] = None
    
    # Monitoring options
    enable_monitoring: bool = True
    enable_profiling: bool = False
    experiment_name: Optional[str] = None


@dataclass
class OptimizationResponse:
    """Complete response from optimization orchestrator."""
    success: bool
    result: OptimizationResult
    recommendation: StrategyRecommendation
    session_summary: Dict[str, Any]
    total_time: float
    strategy_used: str
    confidence_score: float
    convergence_quality: str
    recommendations: List[str]
    
    # Optional fields with defaults
    fallback_used: bool = False
    profiling_data: Optional[Dict[str, Any]] = None


class CircuitBreaker:
    """
    Circuit breaker pattern for optimization strategy failures.
    
    Prevents cascading failures by temporarily disabling failing strategies.
    """
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_counts = {}
        self.last_failure_times = {}
        self.disabled_strategies = set()
    
    def can_execute(self, strategy: OptimizationStrategy) -> bool:
        """Check if strategy can be executed (circuit is closed)."""
        if strategy not in self.disabled_strategies:
            return True
        
        # Check if recovery timeout has passed
        last_failure = self.last_failure_times.get(strategy, 0)
        if time.time() - last_failure > self.recovery_timeout:
            logger.info(f"Circuit breaker recovery: re-enabling {strategy.value}")
            self.disabled_strategies.discard(strategy)
            self.failure_counts[strategy] = 0
            return True
        
        return False
    
    def record_success(self, strategy: OptimizationStrategy):
        """Record successful execution."""
        if strategy in self.failure_counts:
            self.failure_counts[strategy] = 0
        if strategy in self.disabled_strategies:
            self.disabled_strategies.discard(strategy)
    
    def record_failure(self, strategy: OptimizationStrategy):
        """Record failed execution."""
        self.failure_counts[strategy] = self.failure_counts.get(strategy, 0) + 1
        self.last_failure_times[strategy] = time.time()
        
        if self.failure_counts[strategy] >= self.failure_threshold:
            logger.warning(f"Circuit breaker opened: disabling {strategy.value} "
                          f"after {self.failure_counts[strategy]} failures")
            self.disabled_strategies.add(strategy)


class OptimizationOrchestrator:
    """
    Main orchestrator for optimization operations.
    
    Coordinates strategy selection, execution, monitoring, and recovery
    following enterprise patterns for reliability and observability.
    """
    
    def __init__(
        self,
        tracking_dir: Optional[Union[str, Path]] = None,
        enable_circuit_breaker: bool = True
    ):
        # Core components
        self.strategy_selector = StrategySelector()
        self.performance_monitor = PerformanceMonitor()
        self.experiment_tracker = ExperimentTracker(tracking_dir)
        self.profiler = OptimizationProfiler()
        
        # Resilience components
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        
        # State tracking
        self.optimization_history = []
        self.active_sessions = {}
        
        logger.info("Optimization orchestrator initialized")
    
    def optimize(
        self,
        request: OptimizationRequest,
        context: ModelContext
    ) -> OptimizationResponse:
        """
        Main optimization entry point.
        
        Coordinates the complete optimization workflow with monitoring,
        fallback strategies, and comprehensive result analysis.
        """
        start_time = time.time()
        session_id = None
        
        try:
            # Strategy selection
            if request.preferred_strategy is not None:
                recommendation = self._create_manual_recommendation(
                    request.preferred_strategy, context
                )
            else:
                recommendation = self.strategy_selector.select_strategy(
                    context, request.user_preferences
                )
            
            logger.info(f"Selected {recommendation.strategy.value} strategy "
                       f"(confidence: {recommendation.confidence:.1%})")
            
            # Start monitoring
            if request.enable_monitoring:
                difficulty = self._assess_difficulty(context)
                session_id = self.performance_monitor.start_session(
                    recommendation.strategy,
                    context.n_parameters,
                    difficulty
                )
            
            # Start profiling if requested
            profile_id = None
            if request.enable_profiling:
                profile_id = f"opt_{int(time.time())}"
                self.profiler.start_profiling(profile_id)
            
            # Execute optimization with monitoring
            result = self._execute_with_monitoring(
                request, recommendation, context, session_id, profile_id
            )
            
            # End monitoring
            if session_id:
                self.performance_monitor.end_session(result)
            
            # Generate response
            response = self._generate_response(
                request, result, recommendation, session_id, start_time, profile_id
            )
            
            # Record success for circuit breaker
            if self.circuit_breaker and result.success:
                self.circuit_breaker.record_success(recommendation.strategy)
            
            # Log to experiment tracker
            if request.experiment_name and request.enable_monitoring:
                self._log_experiment_run(request.experiment_name, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Optimization orchestration failed: {e}")
            
            # Record failure for circuit breaker
            if self.circuit_breaker and 'recommendation' in locals():
                self.circuit_breaker.record_failure(recommendation.strategy)
            
            # Return error response
            return self._create_error_response(str(e), start_time)
    
    def optimize_with_experiment(
        self,
        request: OptimizationRequest,
        context: ModelContext,
        experiment_name: str,
        description: str = ""
    ) -> OptimizationResponse:
        """Optimize within an experiment context."""
        with optimization_experiment(experiment_name, description):
            request.experiment_name = experiment_name
            return self.optimize(request, context)
    
    def batch_optimize(
        self,
        requests: List[OptimizationRequest],
        context: ModelContext,
        experiment_name: str = "batch_optimization"
    ) -> List[OptimizationResponse]:
        """
        Run multiple optimization requests in batch.
        
        Useful for hyperparameter sweeps, strategy comparisons, etc.
        """
        with optimization_experiment(experiment_name, f"Batch of {len(requests)} optimizations"):
            responses = []
            
            for i, request in enumerate(requests):
                logger.info(f"Running batch optimization {i+1}/{len(requests)}")
                request.experiment_name = experiment_name
                
                try:
                    response = self.optimize(request, context)
                    responses.append(response)
                except Exception as e:
                    logger.error(f"Batch optimization {i+1} failed: {e}")
                    error_response = self._create_error_response(str(e), time.time())
                    responses.append(error_response)
            
            return responses
    
    def compare_strategies(
        self,
        request: OptimizationRequest,
        context: ModelContext,
        strategies: Optional[List[OptimizationStrategy]] = None
    ) -> Dict[str, OptimizationResponse]:
        """
        Compare multiple optimization strategies on the same problem.
        
        Returns performance comparison across strategies.
        """
        if strategies is None:
            # Use all available strategies
            strategies = list(OptimizationStrategy)
        
        experiment_name = "strategy_comparison"
        results = {}
        
        with optimization_experiment(experiment_name, f"Comparing {len(strategies)} strategies"):
            for strategy in strategies:
                # Skip disabled strategies
                if self.circuit_breaker and not self.circuit_breaker.can_execute(strategy):
                    logger.info(f"Skipping {strategy.value} (circuit breaker open)")
                    continue
                
                logger.info(f"Testing strategy: {strategy.value}")
                
                # Create request for this strategy
                strategy_request = OptimizationRequest(
                    objective_function=request.objective_function,
                    initial_parameters=request.initial_parameters.copy(),
                    bounds=request.bounds,
                    gradient_function=request.gradient_function,
                    hessian_function=request.hessian_function,
                    preferred_strategy=strategy,
                    user_preferences=request.user_preferences,
                    enable_monitoring=True,
                    experiment_name=experiment_name
                )
                
                try:
                    response = self.optimize(strategy_request, context)
                    results[strategy.value] = response
                except Exception as e:
                    logger.error(f"Strategy {strategy.value} failed: {e}")
                    results[strategy.value] = self._create_error_response(str(e), time.time())
        
        # Analyze comparison
        self._log_strategy_comparison(results)
        
        return results
    
    def _execute_with_monitoring(
        self,
        request: OptimizationRequest,
        recommendation: StrategyRecommendation,
        context: ModelContext,
        session_id: Optional[str],
        profile_id: Optional[str]
    ) -> OptimizationResult:
        """Execute optimization with comprehensive monitoring."""
        
        # Apply config overrides
        config = recommendation.config
        if request.config_overrides:
            config = config.copy_with_overrides(**request.config_overrides)
        
        # Create monitored objective function
        objective_with_monitoring = self._wrap_objective_with_monitoring(
            request.objective_function, session_id, profile_id
        )
        
        # Primary strategy execution
        primary_success = False
        result = None
        
        if self.circuit_breaker is None or self.circuit_breaker.can_execute(recommendation.strategy):
            try:
                with self.profiler.profile_section("primary_optimization"):
                    result = minimize_with_strategy(
                        recommendation.strategy,
                        objective_with_monitoring,
                        request.initial_parameters,
                        config=config,
                        bounds=request.bounds,
                        gradient=request.gradient_function
                    )
                
                primary_success = result.success
                
            except Exception as e:
                logger.warning(f"Primary strategy {recommendation.strategy.value} failed: {e}")
                if self.circuit_breaker:
                    self.circuit_breaker.record_failure(recommendation.strategy)
        
        # Fallback strategies if primary failed
        if not primary_success and recommendation.fallback_strategies:
            logger.info("Attempting fallback strategies")
            
            for fallback_strategy in recommendation.fallback_strategies:
                if self.circuit_breaker and not self.circuit_breaker.can_execute(fallback_strategy):
                    continue
                
                try:
                    logger.info(f"Trying fallback: {fallback_strategy.value}")
                    
                    with self.profiler.profile_section(f"fallback_{fallback_strategy.value}"):
                        fallback_config = self.strategy_selector._generate_config(
                            fallback_strategy,
                            self.strategy_selector.analyzer.analyze_characteristics(context)
                        )
                        
                        result = minimize_with_strategy(
                            fallback_strategy,
                            objective_with_monitoring,
                            request.initial_parameters,
                            config=fallback_config,
                            bounds=request.bounds,
                            gradient=request.gradient_function
                        )
                    
                    if result.success:
                        result.strategy_used = fallback_strategy.value
                        logger.info(f"Fallback strategy {fallback_strategy.value} succeeded")
                        break
                        
                except Exception as e:
                    logger.warning(f"Fallback strategy {fallback_strategy.value} failed: {e}")
                    if self.circuit_breaker:
                        self.circuit_breaker.record_failure(fallback_strategy)
                    continue
        
        if result is None:
            raise RuntimeError("All optimization strategies failed")
        
        return result
    
    def _wrap_objective_with_monitoring(
        self,
        objective: Callable,
        session_id: Optional[str],
        profile_id: Optional[str]
    ) -> Callable:
        """Wrap objective function with monitoring capabilities."""
        
        call_count = [0]  # Mutable counter
        
        def monitored_objective(x):
            call_count[0] += 1
            
            if profile_id:
                with self.profiler.profile_section("objective_evaluation"):
                    result = objective(x)
            else:
                result = objective(x)
            
            # Log metrics if monitoring is enabled
            if session_id:
                try:
                    # Estimate gradient norm if possible
                    gradient_norm = None
                    if hasattr(objective, '__gradient__'):
                        try:
                            grad = objective.__gradient__(x)
                            gradient_norm = float(np.linalg.norm(grad))
                        except:
                            pass
                    
                    metrics = OptimizationMetrics(
                        iteration=call_count[0],
                        objective_value=float(result),
                        gradient_norm=gradient_norm,
                        parameter_norm=float(np.linalg.norm(x))
                    )
                    
                    self.performance_monitor.log_metrics(metrics)
                    
                except Exception as e:
                    logger.debug(f"Metrics logging failed: {e}")
            
            return result
        
        return monitored_objective
    
    def _generate_response(
        self,
        request: OptimizationRequest,
        result: OptimizationResult,
        recommendation: StrategyRecommendation,
        session_id: Optional[str],
        start_time: float,
        profile_id: Optional[str]
    ) -> OptimizationResponse:
        """Generate comprehensive optimization response."""
        
        total_time = time.time() - start_time
        
        # Get session summary
        session_summary = {}
        if session_id:
            session_summary = self.performance_monitor.get_session_summary(session_id)
        
        # Get profiling data
        profiling_data = None
        if profile_id:
            profiling_data = self.profiler.end_profiling()
        
        # Assess convergence quality
        convergence_quality = self._assess_convergence_quality(result, recommendation)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result, recommendation, session_summary)
        
        # Determine if fallback was used
        fallback_used = result.strategy_used != recommendation.strategy.value
        
        return OptimizationResponse(
            success=result.success,
            result=result,
            recommendation=recommendation,
            session_summary=session_summary,
            total_time=total_time,
            strategy_used=result.strategy_used,
            fallback_used=fallback_used,
            profiling_data=profiling_data,
            confidence_score=recommendation.confidence,
            convergence_quality=convergence_quality,
            recommendations=recommendations
        )
    
    def _assess_difficulty(self, context: ModelContext) -> str:
        """Assess optimization difficulty for monitoring."""
        characteristics = self.strategy_selector.analyzer.analyze_characteristics(context)
        difficulty = self.strategy_selector.analyzer.classify_difficulty(characteristics)
        return difficulty.value
    
    def _assess_convergence_quality(
        self, 
        result: OptimizationResult, 
        recommendation: StrategyRecommendation
    ) -> str:
        """Assess quality of convergence."""
        if not result.success:
            return "failed"
        
        # Consider various quality indicators
        if result.nit < recommendation.config.max_iter * 0.1:
            return "excellent"  # Very fast convergence
        elif result.nit < recommendation.config.max_iter * 0.5:
            return "good"
        elif result.nit < recommendation.config.max_iter * 0.8:
            return "adequate"
        else:
            return "poor"  # Slow convergence
    
    def _generate_recommendations(
        self,
        result: OptimizationResult,
        recommendation: StrategyRecommendation,
        session_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate post-optimization recommendations."""
        recommendations = []
        
        if not result.success:
            recommendations.append("Optimization failed - consider preprocessing data or simplifying model")
            recommendations.extend(recommendation.preprocessing_recommendations)
        
        if result.nit >= recommendation.config.max_iter * 0.9:
            recommendations.append("Reached iteration limit - consider increasing max_iter or improving initialization")
        
        if 'convergence_stats' in session_summary:
            conv_stats = session_summary['convergence_stats']
            if conv_stats.get('total_improvement', 0) < 1e-6:
                recommendations.append("Very small improvement - check for numerical precision issues")
        
        return recommendations
    
    def _create_manual_recommendation(
        self, 
        strategy: OptimizationStrategy, 
        context: ModelContext
    ) -> StrategyRecommendation:
        """Create recommendation for manually specified strategy."""
        characteristics = self.strategy_selector.analyzer.analyze_characteristics(context)
        config = self.strategy_selector._generate_config(strategy, characteristics)
        
        return StrategyRecommendation(
            strategy=strategy,
            config=config,
            confidence=0.8,  # Lower confidence for manual selection
            rationale=f"Manually selected {strategy.value}",
            expected_success_rate=0.8,
            estimated_time_seconds=10.0,
            fallback_strategies=[],
            preprocessing_recommendations=[]
        )
    
    def _create_error_response(self, error_message: str, start_time: float) -> OptimizationResponse:
        """Create error response."""
        return OptimizationResponse(
            success=False,
            result=OptimizationResult(
                success=False,
                x=np.array([]),
                fun=float('inf'),
                nit=0,
                nfev=0,
                message=error_message
            ),
            recommendation=None,
            session_summary={},
            total_time=time.time() - start_time,
            strategy_used="none",
            confidence_score=0.0,
            convergence_quality="failed",
            recommendations=[f"Error occurred: {error_message}"]
        )
    
    def _log_experiment_run(self, experiment_name: str, response: OptimizationResponse):
        """Log optimization run to experiment tracker."""
        try:
            # Create or get experiment
            if not hasattr(self, '_active_experiments'):
                self._active_experiments = {}
            
            if experiment_name not in self._active_experiments:
                self.experiment_tracker.start_experiment(experiment_name)
                self._active_experiments[experiment_name] = True
            
            # Extract session from response
            if 'session' in response.session_summary:
                session_data = response.session_summary['session']
                from .monitoring import OptimizationSession
                
                session = OptimizationSession(**session_data)
                
                # Log the run
                self.experiment_tracker.log_run(
                    session=session,
                    parameters={
                        'strategy': response.strategy_used,
                        'n_parameters': session.n_parameters,
                        'fallback_used': response.fallback_used
                    },
                    metrics={
                        'final_objective': response.result.fun,
                        'total_time': response.total_time,
                        'confidence_score': response.confidence_score,
                        'success': 1.0 if response.success else 0.0
                    }
                )
        except Exception as e:
            logger.warning(f"Experiment logging failed: {e}")
    
    def _log_strategy_comparison(self, results: Dict[str, OptimizationResponse]):
        """Log strategy comparison results."""
        logger.info("Strategy comparison results:")
        
        successful_results = {k: v for k, v in results.items() if v.success}
        
        if successful_results:
            # Sort by objective value
            sorted_results = sorted(
                successful_results.items(),
                key=lambda x: x[1].result.fun
            )
            
            logger.info("Ranking (by objective value):")
            for i, (strategy, response) in enumerate(sorted_results, 1):
                logger.info(f"  {i}. {strategy}: {response.result.fun:.6f} "
                           f"({response.total_time:.2f}s, {response.result.nit} iter)")
        
        failure_count = len(results) - len(successful_results)
        if failure_count > 0:
            logger.warning(f"{failure_count} strategies failed")


# Convenience functions for common use cases

def optimize_model(
    objective_function: Callable,
    initial_parameters: np.ndarray,
    context: ModelContext,
    bounds: Optional[List[tuple]] = None,
    preferred_strategy: Optional[OptimizationStrategy] = None,
    **kwargs
) -> OptimizationResponse:
    """
    High-level convenience function for model optimization.
    
    This is the main entry point most users should use.
    """
    orchestrator = OptimizationOrchestrator()
    
    request = OptimizationRequest(
        objective_function=objective_function,
        initial_parameters=initial_parameters,
        bounds=bounds,
        preferred_strategy=preferred_strategy,
        **kwargs
    )
    
    return orchestrator.optimize(request, context)


def compare_optimization_strategies(
    objective_function: Callable,
    initial_parameters: np.ndarray,
    context: ModelContext,
    strategies: Optional[List[OptimizationStrategy]] = None,
    **kwargs
) -> Dict[str, OptimizationResponse]:
    """Compare multiple optimization strategies."""
    orchestrator = OptimizationOrchestrator()
    
    request = OptimizationRequest(
        objective_function=objective_function,
        initial_parameters=initial_parameters,
        **kwargs
    )
    
    return orchestrator.compare_strategies(request, context, strategies)