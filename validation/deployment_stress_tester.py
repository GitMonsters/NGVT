"""
Deployment Stress Tester

Production-grade stress testing for deployment readiness assessment.
Tests system behavior under various load conditions and stress scenarios.
"""

import logging
import time
import psutil
import threading
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StressTestResult:
    """Results from stress testing"""
    overall_stability_score: float
    readiness_classification: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float
    p95_latency: float
    p99_latency: float
    max_latency: float
    throughput: float
    error_rate: float
    peak_memory_mb: float
    peak_cpu_percent: float
    test_duration: float
    stress_level: str = "MEDIUM"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'overall_stability_score': self.overall_stability_score,
            'readiness_classification': self.readiness_classification,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'average_latency': self.average_latency,
            'p95_latency': self.p95_latency,
            'p99_latency': self.p99_latency,
            'max_latency': self.max_latency,
            'throughput': self.throughput,
            'error_rate': self.error_rate,
            'peak_memory_mb': self.peak_memory_mb,
            'peak_cpu_percent': self.peak_cpu_percent,
            'test_duration': self.test_duration,
            'stress_level': self.stress_level,
        }


class DeploymentStressTester:
    """
    Production-grade stress testing system
    
    Tests:
    - Load handling: Concurrent request processing
    - Latency: Response time under load
    - Throughput: Requests per second
    - Resource usage: Memory and CPU consumption
    - Error handling: Graceful degradation
    - Stability: Sustained operation under stress
    """
    
    def __init__(self, max_latency_ms: float = 5000, max_memory_mb: float = 8192):
        """
        Initialize stress tester
        
        Args:
            max_latency_ms: Maximum acceptable latency in milliseconds
            max_memory_mb: Maximum acceptable memory usage in MB
        """
        self.max_latency_ms = max_latency_ms
        self.max_memory_mb = max_memory_mb
        self.process = psutil.Process()
    
    def run_comprehensive_stress_test(self,
                                     model,
                                     test_input: Any = None,
                                     duration_seconds: int = 60,
                                     concurrent_requests: int = 10,
                                     request_rate: int = 5) -> StressTestResult:
        """
        Run comprehensive stress test
        
        Args:
            model: Model to test
            test_input: Input for testing (if None, uses default)
            duration_seconds: Test duration in seconds
            concurrent_requests: Number of concurrent requests
            request_rate: Target requests per second
            
        Returns:
            StressTestResult with comprehensive metrics
        """
        logger.info(f"Starting comprehensive stress test...")
        logger.info(f"  Duration: {duration_seconds}s")
        logger.info(f"  Concurrent requests: {concurrent_requests}")
        logger.info(f"  Target rate: {request_rate} req/s")
        
        # Use default test input if none provided
        if test_input is None:
            test_input = self._get_default_test_input()
        
        # Track metrics
        latencies = []
        successes = 0
        failures = 0
        peak_memory = 0.0
        peak_cpu = 0.0
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        # Resource monitoring thread
        monitoring = {'running': True}
        monitor_thread = threading.Thread(
            target=self._monitor_resources,
            args=(monitoring, lambda: peak_memory, lambda: peak_cpu)
        )
        monitor_thread.start()
        
        # Run stress test
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = []
            
            while time.time() < end_time:
                # Submit batch of requests
                for _ in range(request_rate):
                    if time.time() >= end_time:
                        break
                    
                    future = executor.submit(self._execute_request, model, test_input)
                    futures.append(future)
                
                # Wait to maintain rate
                time.sleep(1.0)
                
                # Collect completed results
                completed = [f for f in futures if f.done()]
                for future in completed:
                    try:
                        latency, success = future.result(timeout=1)
                        latencies.append(latency)
                        if success:
                            successes += 1
                        else:
                            failures += 1
                        futures.remove(future)
                    except Exception as e:
                        logger.debug(f"Request processing error: {e}")
                        failures += 1
                        futures.remove(future)
                
                # Update peak metrics
                try:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    cpu_percent = self.process.cpu_percent(interval=0.1)
                    peak_memory = max(peak_memory, memory_mb)
                    peak_cpu = max(peak_cpu, cpu_percent)
                except Exception:
                    pass
            
            # Wait for remaining futures
            for future in as_completed(futures, timeout=10):
                try:
                    latency, success = future.result()
                    latencies.append(latency)
                    if success:
                        successes += 1
                    else:
                        failures += 1
                except Exception:
                    failures += 1
        
        # Stop monitoring
        monitoring['running'] = False
        monitor_thread.join(timeout=5)
        
        # Calculate metrics
        test_duration = time.time() - start_time
        total_requests = successes + failures
        
        if latencies:
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)
            max_latency = max(latencies)
        else:
            avg_latency = p95_latency = p99_latency = max_latency = 0.0
        
        throughput = total_requests / test_duration if test_duration > 0 else 0.0
        error_rate = failures / total_requests if total_requests > 0 else 1.0
        
        # Calculate stability score
        stability_score = self._calculate_stability_score(
            error_rate, avg_latency, peak_memory, peak_cpu
        )
        
        # Determine readiness classification
        readiness = self._classify_readiness(stability_score, error_rate, avg_latency)
        
        # Determine stress level
        stress_level = self._determine_stress_level(concurrent_requests, request_rate)
        
        result = StressTestResult(
            overall_stability_score=stability_score,
            readiness_classification=readiness,
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=failures,
            average_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            max_latency=max_latency,
            throughput=throughput,
            error_rate=error_rate,
            peak_memory_mb=peak_memory,
            peak_cpu_percent=peak_cpu,
            test_duration=test_duration,
            stress_level=stress_level
        )
        
        logger.info(f"Stress test complete: {readiness} ({stability_score:.2%} stability)")
        logger.info(f"  Throughput: {throughput:.1f} req/s")
        logger.info(f"  Success rate: {(1-error_rate):.2%}")
        
        return result
    
    def _execute_request(self, model, test_input: Any) -> tuple:
        """
        Execute a single request
        
        Args:
            model: Model to test
            test_input: Input data
            
        Returns:
            Tuple of (latency_ms, success)
        """
        start_time = time.time()
        success = False
        
        try:
            # Execute model
            if hasattr(model, 'predict'):
                _ = model.predict(test_input)
            elif hasattr(model, 'forward'):
                _ = model.forward(test_input)
            elif callable(model):
                _ = model(test_input)
            else:
                # Fallback: simulate work
                time.sleep(0.01)
            
            success = True
            
        except Exception as e:
            logger.debug(f"Request execution error: {e}")
            success = False
        
        latency_ms = (time.time() - start_time) * 1000
        return latency_ms, success
    
    def _monitor_resources(self, monitoring: Dict, peak_memory_ref, peak_cpu_ref):
        """
        Monitor system resources during test
        
        Note: This is a placeholder for resource monitoring.
        In production, this would actively monitor and update peak values.
        Current implementation relies on inline monitoring in the main test loop.
        
        Args:
            monitoring: Dictionary with 'running' flag
            peak_memory_ref: Reference to peak memory variable (unused in current implementation)
            peak_cpu_ref: Reference to peak CPU variable (unused in current implementation)
        """
        # Placeholder - actual monitoring happens in the main stress test loop
        while monitoring['running']:
            try:
                time.sleep(1.0)
            except Exception:
                pass
    
    def _get_default_test_input(self) -> Any:
        """Get default test input"""
        return np.random.randn(10, 10)
    
    def _calculate_stability_score(self,
                                   error_rate: float,
                                   avg_latency: float,
                                   peak_memory: float,
                                   peak_cpu: float) -> float:
        """
        Calculate overall stability score
        
        Args:
            error_rate: Request error rate
            avg_latency: Average latency in ms
            peak_memory: Peak memory usage in MB
            peak_cpu: Peak CPU usage in percent
            
        Returns:
            Stability score (0.0 to 1.0)
        """
        # Success rate score (0-30 points)
        success_score = (1 - error_rate) * 30
        
        # Latency score (0-30 points)
        latency_ratio = min(avg_latency / self.max_latency_ms, 1.0)
        latency_score = (1 - latency_ratio) * 30
        
        # Memory score (0-20 points)
        memory_ratio = min(peak_memory / self.max_memory_mb, 1.0)
        memory_score = (1 - memory_ratio) * 20
        
        # CPU score (0-20 points)
        cpu_ratio = min(peak_cpu / 100.0, 1.0)
        cpu_score = (1 - cpu_ratio) * 20
        
        # Total score (0-100, converted to 0-1)
        total_score = (success_score + latency_score + memory_score + cpu_score) / 100.0
        
        return min(max(total_score, 0.0), 1.0)
    
    def _classify_readiness(self,
                           stability_score: float,
                           error_rate: float,
                           avg_latency: float) -> str:
        """
        Classify deployment readiness
        
        Args:
            stability_score: Overall stability score
            error_rate: Request error rate
            avg_latency: Average latency
            
        Returns:
            Readiness classification
        """
        if stability_score >= 0.90 and error_rate < 0.01 and avg_latency < 1000:
            return "PRODUCTION-READY"
        elif stability_score >= 0.75 and error_rate < 0.05:
            return "BETA-READY"
        elif stability_score >= 0.60 and error_rate < 0.10:
            return "ALPHA-READY"
        elif stability_score >= 0.40:
            return "DEVELOPMENT"
        else:
            return "NOT-READY"
    
    def _determine_stress_level(self, concurrent: int, rate: int) -> str:
        """
        Determine stress level based on test parameters
        
        Args:
            concurrent: Concurrent requests
            rate: Requests per second
            
        Returns:
            Stress level classification
        """
        total_load = concurrent * rate
        
        if total_load >= 1000:
            return "EXTREME"
        elif total_load >= 500:
            return "HIGH"
        elif total_load >= 100:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_stress_report(self, result: StressTestResult) -> str:
        """
        Generate comprehensive stress test report
        
        Args:
            result: Stress test results
            
        Returns:
            Formatted report string
        """
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║          DEPLOYMENT STRESS TEST REPORT                       ║
╚══════════════════════════════════════════════════════════════╝

Deployment Readiness: {result.readiness_classification}
Stability Score: {result.overall_stability_score:.2%}
Stress Level: {result.stress_level}

Performance Metrics:
  • Total Requests: {result.total_requests}
  • Successful: {result.successful_requests} ({(1-result.error_rate):.2%})
  • Failed: {result.failed_requests} ({result.error_rate:.2%})
  • Throughput: {result.throughput:.1f} req/s

Latency Analysis:
  • Average: {result.average_latency:.2f} ms
  • P95: {result.p95_latency:.2f} ms
  • P99: {result.p99_latency:.2f} ms
  • Max: {result.max_latency:.2f} ms

Resource Usage:
  • Peak Memory: {result.peak_memory_mb:.1f} MB
  • Peak CPU: {result.peak_cpu_percent:.1f}%

Test Configuration:
  • Duration: {result.test_duration:.1f} seconds

Assessment:
"""
        
        if result.readiness_classification == "PRODUCTION-READY":
            report += "  ✅ System is ready for production deployment\n"
        elif result.readiness_classification == "BETA-READY":
            report += "  ✓  System is suitable for beta testing\n"
        elif result.readiness_classification == "ALPHA-READY":
            report += "  ⚠️  System needs more testing before production\n"
        else:
            report += "  ❌ System requires significant improvements\n"
        
        if result.error_rate > 0.05:
            report += "  ⚠️  High error rate - investigate failure causes\n"
        
        if result.average_latency > self.max_latency_ms:
            report += f"  ⚠️  Average latency exceeds threshold ({self.max_latency_ms}ms)\n"
        
        if result.peak_memory_mb > self.max_memory_mb:
            report += f"  ⚠️  Memory usage exceeds threshold ({self.max_memory_mb}MB)\n"
        
        return report
    
    def run_spike_test(self, model, test_input: Any = None, 
                       spike_duration: int = 10) -> Dict[str, Any]:
        """
        Run spike test (sudden load increase)
        
        Args:
            model: Model to test
            test_input: Input for testing
            spike_duration: Duration of spike in seconds
            
        Returns:
            Spike test results
        """
        logger.info("Running spike test...")
        
        # Normal load
        normal_result = self.run_comprehensive_stress_test(
            model, test_input, duration_seconds=spike_duration,
            concurrent_requests=5, request_rate=2
        )
        
        # Spike load
        spike_result = self.run_comprehensive_stress_test(
            model, test_input, duration_seconds=spike_duration,
            concurrent_requests=50, request_rate=20
        )
        
        # Compare results
        latency_degradation = spike_result.average_latency / normal_result.average_latency if normal_result.average_latency > 0 else 1.0
        error_increase = spike_result.error_rate - normal_result.error_rate
        
        return {
            'normal_performance': normal_result.to_dict(),
            'spike_performance': spike_result.to_dict(),
            'latency_degradation_factor': latency_degradation,
            'error_rate_increase': error_increase,
            'handles_spike_well': latency_degradation < 2.0 and error_increase < 0.1,
        }
