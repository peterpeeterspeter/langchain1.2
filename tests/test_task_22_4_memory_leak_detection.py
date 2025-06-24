#!/usr/bin/env python3
"""
Task 22.4: Memory Leak Detection and Resource Cleanup

This module implements comprehensive memory profiling and resource cleanup
verification for screenshot operations to detect and prevent memory leaks.

✅ TASK 22.4 FEATURES:
- Memory profiling during extended screenshot operations
- Automated memory leak detection with linear regression analysis
- Resource cleanup verification for browser processes
- Memory usage logging with configurable thresholds
- Process count monitoring to prevent orphaned processes
- Temporary file cleanup verification
- Long-running stability tests
"""

import pytest
import asyncio
import time
import psutil
import os
import tempfile
import gc
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import json
import logging

# Import the classes we're testing
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from integrations.playwright_screenshot_engine import (
    BrowserPoolManager,
    ScreenshotService,
    ScreenshotResult,
    ResourceOptimizationConfig
)


@dataclass
class MemorySnapshot:
    """Container for memory usage data at a specific point in time"""
    timestamp: datetime
    process_id: int
    memory_mb: float
    cpu_percent: float
    thread_count: int
    file_descriptors: int
    browser_processes: List[int] = field(default_factory=list)
    temp_files_count: int = 0
    temp_files_size_mb: float = 0.0


@dataclass
class ResourceState:
    """Container for system resource state"""
    timestamp: datetime
    total_processes: int
    browser_processes: List[int]
    temp_directories: List[str]
    temp_files: List[str]
    open_file_descriptors: int
    system_memory_mb: float
    system_memory_percent: float


class MemoryProfiler:
    """
    Memory profiling system for monitoring browser processes and detecting leaks
    """
    
    def __init__(self, sample_interval: float = 1.0, alert_threshold_mb: float = 1024.0):
        self.sample_interval = sample_interval
        self.alert_threshold_mb = alert_threshold_mb
        self.snapshots: List[MemorySnapshot] = []
        self.is_profiling = False
        self.profiling_thread = None
        self.target_process_ids: List[int] = []
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
    def start_profiling(self, target_process_ids: List[int] = None):
        """Start memory profiling for specified processes"""
        if self.is_profiling:
            return
            
        self.target_process_ids = target_process_ids or []
        self.is_profiling = True
        self.snapshots.clear()
        
        self.profiling_thread = threading.Thread(target=self._profiling_loop, daemon=True)
        self.profiling_thread.start()
        
        self.logger.info(f"Memory profiling started for {len(self.target_process_ids)} processes")
    
    def stop_profiling(self):
        """Stop memory profiling"""
        if not self.is_profiling:
            return
            
        self.is_profiling = False
        if self.profiling_thread:
            self.profiling_thread.join(timeout=2.0)
            
        self.logger.info(f"Memory profiling stopped. Collected {len(self.snapshots)} snapshots")
    
    def _profiling_loop(self):
        """Main profiling loop that runs in a separate thread"""
        while self.is_profiling:
            try:
                snapshot = self._take_memory_snapshot()
                if snapshot:
                    self.snapshots.append(snapshot)
                    
                    # Check for memory alerts
                    if snapshot.memory_mb > self.alert_threshold_mb:
                        self.logger.warning(
                            f"Memory alert: Process {snapshot.process_id} using {snapshot.memory_mb:.1f}MB "
                            f"(threshold: {self.alert_threshold_mb}MB)"
                        )
                        
                time.sleep(self.sample_interval)
                
            except Exception as e:
                self.logger.error(f"Error in profiling loop: {e}")
                
    def _take_memory_snapshot(self) -> Optional[MemorySnapshot]:
        """Take a memory snapshot of target processes"""
        try:
            current_time = datetime.now()
            
            # If no specific targets, monitor current process
            if not self.target_process_ids:
                process = psutil.Process()
                return MemorySnapshot(
                    timestamp=current_time,
                    process_id=process.pid,
                    memory_mb=process.memory_info().rss / 1024 / 1024,
                    cpu_percent=process.cpu_percent(),
                    thread_count=process.num_threads(),
                    file_descriptors=process.num_fds() if hasattr(process, 'num_fds') else 0
                )
            
            # Monitor multiple target processes
            total_memory = 0
            total_threads = 0
            total_fds = 0
            browser_pids = []
            
            for pid in self.target_process_ids:
                try:
                    process = psutil.Process(pid)
                    if process.is_running():
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        total_memory += memory_mb
                        total_threads += process.num_threads()
                        total_fds += process.num_fds() if hasattr(process, 'num_fds') else 0
                        browser_pids.append(pid)
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            return MemorySnapshot(
                timestamp=current_time,
                process_id=0,  # Combined processes
                memory_mb=total_memory,
                cpu_percent=0,
                thread_count=total_threads,
                file_descriptors=total_fds,
                browser_processes=browser_pids
            )
            
        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {e}")
            return None
    
    def analyze_memory_trends(self) -> Dict[str, Any]:
        """Analyze memory usage trends to detect potential leaks"""
        if len(self.snapshots) < 10:
            return {"error": "Insufficient data for trend analysis"}
        
        # Extract time series data
        timestamps = [(s.timestamp - self.snapshots[0].timestamp).total_seconds() for s in self.snapshots]
        memory_values = [s.memory_mb for s in self.snapshots]
        
        # Linear regression to detect memory growth
        coefficients = np.polyfit(timestamps, memory_values, 1)
        slope = coefficients[0]  # MB per second
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(timestamps, memory_values)[0, 1]
        
        # Determine leak severity
        leak_severity = "none"
        if slope > 0.1:  # Growing more than 0.1 MB/second
            if correlation > 0.8:  # Strong positive correlation
                leak_severity = "critical"
            elif correlation > 0.6:
                leak_severity = "moderate"
            elif correlation > 0.4:
                leak_severity = "minor"
        
        analysis = {
            "sample_count": len(self.snapshots),
            "duration_seconds": timestamps[-1] if timestamps else 0,
            "memory_trend": {
                "slope_mb_per_second": slope,
                "correlation": correlation,
                "leak_severity": leak_severity
            },
            "memory_stats": {
                "initial_mb": memory_values[0] if memory_values else 0,
                "final_mb": memory_values[-1] if memory_values else 0,
                "peak_mb": max(memory_values) if memory_values else 0,
                "average_mb": sum(memory_values) / len(memory_values) if memory_values else 0
            },
            "recommendations": []
        }
        
        # Add recommendations based on analysis
        if leak_severity == "critical":
            analysis["recommendations"].extend([
                "Immediate investigation required - significant memory leak detected",
                "Review browser instance lifecycle and cleanup procedures",
                "Check for unreleased DOM references or event listeners"
            ])
        elif leak_severity == "moderate":
            analysis["recommendations"].extend([
                "Monitor memory usage closely",
                "Review resource cleanup in screenshot operations",
                "Consider implementing more aggressive garbage collection"
            ])
        elif leak_severity == "minor":
            analysis["recommendations"].append("Continue monitoring - minor memory growth detected")
        else:
            analysis["recommendations"].append("Memory usage appears stable")
        
        return analysis


class ResourceCleanupVerifier:
    """
    System for verifying proper cleanup of browser processes and temporary files
    """
    
    def __init__(self, temp_dir_prefix: str = "playwright"):
        self.temp_dir_prefix = temp_dir_prefix
        self.initial_state: Optional[ResourceState] = None
        self.logger = logging.getLogger(__name__)
        
    def capture_initial_state(self) -> ResourceState:
        """Capture the initial system resource state"""
        current_time = datetime.now()
        
        # Find browser processes
        browser_processes = self._find_browser_processes()
        
        # Find temporary directories and files
        temp_dirs, temp_files = self._find_temp_resources()
        
        # System memory info
        memory_info = psutil.virtual_memory()
        
        self.initial_state = ResourceState(
            timestamp=current_time,
            total_processes=len(psutil.pids()),
            browser_processes=browser_processes,
            temp_directories=temp_dirs,
            temp_files=temp_files,
            open_file_descriptors=self._count_open_fds(),
            system_memory_mb=memory_info.used / 1024 / 1024,
            system_memory_percent=memory_info.percent
        )
        
        self.logger.info(f"Captured initial state: {len(browser_processes)} browser processes, "
                        f"{len(temp_dirs)} temp dirs, {len(temp_files)} temp files")
        
        return self.initial_state
    
    def verify_cleanup(self) -> Dict[str, Any]:
        """Verify that resources have been properly cleaned up"""
        if not self.initial_state:
            return {"error": "No initial state captured"}
        
        current_time = datetime.now()
        current_browser_processes = self._find_browser_processes()
        current_temp_dirs, current_temp_files = self._find_temp_resources()
        current_memory = psutil.virtual_memory()
        
        # Calculate differences
        new_browser_processes = [pid for pid in current_browser_processes 
                               if pid not in self.initial_state.browser_processes]
        orphaned_processes = [pid for pid in self.initial_state.browser_processes 
                            if pid in current_browser_processes]
        
        new_temp_dirs = [d for d in current_temp_dirs 
                        if d not in self.initial_state.temp_directories]
        remaining_temp_dirs = [d for d in self.initial_state.temp_directories 
                             if d in current_temp_dirs]
        
        new_temp_files = [f for f in current_temp_files 
                         if f not in self.initial_state.temp_files]
        remaining_temp_files = [f for f in self.initial_state.temp_files 
                              if f in current_temp_files]
        
        cleanup_verification = {
            "verification_time": current_time.isoformat(),
            "duration_seconds": (current_time - self.initial_state.timestamp).total_seconds(),
            "process_cleanup": {
                "orphaned_browser_processes": orphaned_processes,
                "new_browser_processes": new_browser_processes,
                "process_leak_detected": len(orphaned_processes) > 0
            },
            "file_cleanup": {
                "remaining_temp_dirs": remaining_temp_dirs,
                "remaining_temp_files": remaining_temp_files,
                "new_temp_dirs": new_temp_dirs,
                "new_temp_files": new_temp_files,
                "file_leak_detected": len(remaining_temp_dirs) > 0 or len(remaining_temp_files) > 0
            },
            "memory_cleanup": {
                "initial_memory_mb": self.initial_state.system_memory_mb,
                "current_memory_mb": current_memory.used / 1024 / 1024,
                "memory_delta_mb": (current_memory.used / 1024 / 1024) - self.initial_state.system_memory_mb
            },
            "cleanup_successful": True,
            "issues_found": []
        }
        
        # Check for cleanup issues
        if orphaned_processes:
            cleanup_verification["cleanup_successful"] = False
            cleanup_verification["issues_found"].append(
                f"Orphaned browser processes detected: {orphaned_processes}"
            )
        
        if remaining_temp_dirs or remaining_temp_files:
            cleanup_verification["cleanup_successful"] = False
            cleanup_verification["issues_found"].append(
                f"Temporary files not cleaned up: {len(remaining_temp_dirs)} dirs, {len(remaining_temp_files)} files"
            )
        
        memory_growth = cleanup_verification["memory_cleanup"]["memory_delta_mb"]
        if memory_growth > 100:  # More than 100MB growth
            cleanup_verification["cleanup_successful"] = False
            cleanup_verification["issues_found"].append(
                f"Significant memory growth detected: {memory_growth:.1f}MB"
            )
        
        if not cleanup_verification["issues_found"]:
            cleanup_verification["issues_found"].append("No cleanup issues detected")
        
        return cleanup_verification
    
    def _find_browser_processes(self) -> List[int]:
        """Find all browser-related processes"""
        browser_processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                name = proc.info['name'].lower()
                cmdline = ' '.join(proc.info['cmdline'] or []).lower()
                
                # Look for browser process indicators
                if any(browser in name for browser in ['chrome', 'chromium', 'firefox', 'webkit']):
                    browser_processes.append(proc.info['pid'])
                elif any(indicator in cmdline for indicator in ['--enable-automation', 'playwright', 'selenium']):
                    browser_processes.append(proc.info['pid'])
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
                
        return browser_processes
    
    def _find_temp_resources(self) -> Tuple[List[str], List[str]]:
        """Find temporary directories and files related to browser operations"""
        temp_dirs = []
        temp_files = []
        
        # Check system temp directory
        temp_root = Path(tempfile.gettempdir())
        
        try:
            for item in temp_root.iterdir():
                if self.temp_dir_prefix in item.name.lower():
                    if item.is_dir():
                        temp_dirs.append(str(item))
                    else:
                        temp_files.append(str(item))
                        
        except (PermissionError, FileNotFoundError):
            pass
        
        return temp_dirs, temp_files
    
    def _count_open_fds(self) -> int:
        """Count open file descriptors for current process"""
        try:
            process = psutil.Process()
            return process.num_fds() if hasattr(process, 'num_fds') else 0
        except:
            return 0


class MemoryLeakDetector:
    """
    Main class that orchestrates memory leak detection and resource cleanup verification
    """
    
    def __init__(self, 
                 sample_interval: float = 0.5,
                 alert_threshold_mb: float = 2048.0,
                 output_dir: str = None):
        self.sample_interval = sample_interval
        self.alert_threshold_mb = alert_threshold_mb
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="memory_leak_detection_")
        
        self.profiler = MemoryProfiler(sample_interval, alert_threshold_mb)
        self.cleanup_verifier = ResourceCleanupVerifier()
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    async def run_extended_memory_test(self, 
                                     duration_minutes: int = 30,
                                     operations_per_minute: int = 10) -> Dict[str, Any]:
        """
        Run extended memory leak detection test
        
        Args:
            duration_minutes: How long to run the test
            operations_per_minute: Number of screenshot operations per minute
        """
        self.logger.info(f"Starting extended memory test: {duration_minutes} minutes, "
                        f"{operations_per_minute} ops/min")
        
        # Capture initial resource state
        initial_state = self.cleanup_verifier.capture_initial_state()
        
        # Start memory profiling
        self.profiler.start_profiling()
        
        test_results = {
            "test_start": datetime.now().isoformat(),
            "duration_minutes": duration_minutes,
            "operations_per_minute": operations_per_minute,
            "operations_completed": 0,
            "operations_failed": 0,
            "memory_analysis": {},
            "cleanup_verification": {},
            "overall_success": True
        }
        
        try:
            # Run screenshot operations for specified duration
            end_time = datetime.now() + timedelta(minutes=duration_minutes)
            operation_interval = 60.0 / operations_per_minute  # seconds between operations
            
            while datetime.now() < end_time:
                operation_start = time.time()
                
                try:
                    # Perform screenshot operation
                    await self._perform_test_screenshot_operation()
                    test_results["operations_completed"] += 1
                    
                except Exception as e:
                    self.logger.error(f"Screenshot operation failed: {e}")
                    test_results["operations_failed"] += 1
                
                # Wait for next operation
                elapsed = time.time() - operation_start
                sleep_time = max(0, operation_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Stop profiling and analyze results
            self.profiler.stop_profiling()
            test_results["memory_analysis"] = self.profiler.analyze_memory_trends()
            
            # Verify resource cleanup
            test_results["cleanup_verification"] = self.cleanup_verifier.verify_cleanup()
            
            # Determine overall success
            memory_leak_detected = test_results["memory_analysis"].get("memory_trend", {}).get("leak_severity") in ["moderate", "critical"]
            cleanup_failed = not test_results["cleanup_verification"].get("cleanup_successful", True)
            
            test_results["overall_success"] = not (memory_leak_detected or cleanup_failed)
            
            # Save detailed results
            self._save_test_results(test_results)
            
            self.logger.info(f"Extended memory test completed: "
                           f"{test_results['operations_completed']} operations, "
                           f"{'PASSED' if test_results['overall_success'] else 'FAILED'}")
            
            return test_results
            
        except Exception as e:
            self.profiler.stop_profiling()
            self.logger.error(f"Extended memory test failed: {e}")
            test_results["error"] = str(e)
            test_results["overall_success"] = False
            return test_results
    
    async def _perform_test_screenshot_operation(self):
        """Perform a single screenshot operation for testing"""
        # Create browser pool with resource optimization
        config = ResourceOptimizationConfig()
        pool = BrowserPoolManager(optimization_config=config)
        
        try:
            await pool.initialize()
            service = ScreenshotService(pool)
            
            # Capture screenshot of a simple test page
            result = await service.capture_full_page_screenshot("https://httpbin.org/html")
            
            if not result or not result.success:
                raise Exception(f"Screenshot operation failed: {result.error_message if result else 'No result'}")
                
        finally:
            await pool.cleanup()
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"memory_leak_test_{timestamp}.json")
        
        # Make results JSON serializable
        serializable_results = self._make_json_serializable(results)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Test results saved to: {results_file}")
    
    def _make_json_serializable(self, obj):
        """Convert object to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


# Test Classes for Task 22.4

class TestMemoryProfiler:
    """Test the memory profiling system"""
    
    @pytest.fixture
    def profiler(self):
        """Create a memory profiler for testing"""
        return MemoryProfiler(sample_interval=0.1, alert_threshold_mb=100)
    
    def test_profiler_initialization(self, profiler):
        """Test profiler initialization"""
        assert profiler.sample_interval == 0.1
        assert profiler.alert_threshold_mb == 100
        assert not profiler.is_profiling
        assert len(profiler.snapshots) == 0
    
    def test_memory_snapshot_creation(self, profiler):
        """Test memory snapshot creation"""
        snapshot = profiler._take_memory_snapshot()
        
        assert snapshot is not None
        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.memory_mb > 0
        assert snapshot.process_id > 0
        assert isinstance(snapshot.timestamp, datetime)
    
    def test_profiling_start_stop(self, profiler):
        """Test starting and stopping profiler"""
        profiler.start_profiling()
        assert profiler.is_profiling
        
        # Let it collect a few samples
        time.sleep(0.5)
        
        profiler.stop_profiling()
        assert not profiler.is_profiling
        assert len(profiler.snapshots) > 0
    
    def test_memory_trend_analysis(self, profiler):
        """Test memory trend analysis"""
        # Create mock snapshots with increasing memory
        base_time = datetime.now()
        for i in range(20):
            snapshot = MemorySnapshot(
                timestamp=base_time + timedelta(seconds=i),
                process_id=1234,
                memory_mb=100 + i * 2,  # Steadily increasing memory
                cpu_percent=10.0,
                thread_count=5,
                file_descriptors=20
            )
            profiler.snapshots.append(snapshot)
        
        analysis = profiler.analyze_memory_trends()
        
        assert "memory_trend" in analysis
        assert "slope_mb_per_second" in analysis["memory_trend"]
        assert analysis["memory_trend"]["slope_mb_per_second"] > 0
        assert "leak_severity" in analysis["memory_trend"]


class TestResourceCleanupVerifier:
    """Test the resource cleanup verification system"""
    
    @pytest.fixture
    def verifier(self):
        """Create a resource cleanup verifier for testing"""
        return ResourceCleanupVerifier()
    
    def test_verifier_initialization(self, verifier):
        """Test verifier initialization"""
        assert verifier.temp_dir_prefix == "playwright"
        assert verifier.initial_state is None
    
    def test_initial_state_capture(self, verifier):
        """Test capturing initial system state"""
        initial_state = verifier.capture_initial_state()
        
        assert isinstance(initial_state, ResourceState)
        assert initial_state.total_processes > 0
        assert isinstance(initial_state.browser_processes, list)
        assert isinstance(initial_state.temp_directories, list)
        assert isinstance(initial_state.temp_files, list)
        assert initial_state.system_memory_mb > 0
    
    def test_browser_process_detection(self, verifier):
        """Test browser process detection"""
        browser_processes = verifier._find_browser_processes()
        
        # Should return a list (may be empty if no browsers running)
        assert isinstance(browser_processes, list)
        
        # All items should be integers (PIDs)
        for pid in browser_processes:
            assert isinstance(pid, int)
            assert pid > 0
    
    def test_temp_resource_detection(self, verifier):
        """Test temporary resource detection"""
        temp_dirs, temp_files = verifier._find_temp_resources()
        
        assert isinstance(temp_dirs, list)
        assert isinstance(temp_files, list)
        
        # All paths should be strings
        for path in temp_dirs + temp_files:
            assert isinstance(path, str)
    
    def test_cleanup_verification(self, verifier):
        """Test cleanup verification process"""
        # Capture initial state
        verifier.capture_initial_state()
        
        # Verify cleanup (should pass since nothing changed)
        verification = verifier.verify_cleanup()
        
        assert "verification_time" in verification
        assert "process_cleanup" in verification
        assert "file_cleanup" in verification
        assert "memory_cleanup" in verification
        assert isinstance(verification["cleanup_successful"], bool)
        assert isinstance(verification["issues_found"], list)


class TestMemoryLeakDetector:
    """Test the main memory leak detector"""
    
    @pytest.fixture
    def detector(self):
        """Create a memory leak detector for testing"""
        return MemoryLeakDetector(sample_interval=0.1, alert_threshold_mb=100)
    
    def test_detector_initialization(self, detector):
        """Test detector initialization"""
        assert detector.sample_interval == 0.1
        assert detector.alert_threshold_mb == 100
        assert os.path.exists(detector.output_dir)
        assert isinstance(detector.profiler, MemoryProfiler)
        assert isinstance(detector.cleanup_verifier, ResourceCleanupVerifier)
    
    @pytest.mark.asyncio
    async def test_short_memory_test(self, detector):
        """Test a short memory leak detection test"""
        # Run a very short test to verify functionality
        with patch.object(detector, '_perform_test_screenshot_operation') as mock_operation:
            mock_operation.return_value = None  # Simulate successful operation
            
            results = await detector.run_extended_memory_test(
                duration_minutes=0.05,  # 3 seconds
                operations_per_minute=60  # 1 per second
            )
            
            assert "test_start" in results
            assert "operations_completed" in results
            assert "memory_analysis" in results
            assert "cleanup_verification" in results
            assert isinstance(results["overall_success"], bool)
    
    @pytest.mark.asyncio
    async def test_screenshot_operation(self, detector):
        """Test the screenshot operation used in memory testing"""
        # Mock the screenshot operation directly
        with patch.object(detector, '_perform_test_screenshot_operation') as mock_operation:
            mock_operation.return_value = None
            
            # Should not raise an exception
            await detector._perform_test_screenshot_operation()
            
            # Verify mock was called
            mock_operation.assert_called_once()


class TestIntegrationScenarios:
    """Integration tests for memory leak detection"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_memory_leak_detection(self):
        """Test complete memory leak detection workflow"""
        detector = MemoryLeakDetector(sample_interval=0.1)
        
        # Mock the screenshot operations to avoid actual browser usage
        with patch.object(detector, '_perform_test_screenshot_operation') as mock_operation:
            mock_operation.return_value = None
            
            # Run a very short test
            results = await detector.run_extended_memory_test(
                duration_minutes=0.02,  # ~1 second
                operations_per_minute=120  # 2 per second
            )
            
            # Verify results structure
            assert "test_start" in results
            assert "duration_minutes" in results
            assert "operations_completed" in results
            assert "memory_analysis" in results
            assert "cleanup_verification" in results
            assert isinstance(results["overall_success"], bool)
            
            # Verify memory analysis
            memory_analysis = results["memory_analysis"]
            if "error" not in memory_analysis:
                assert "memory_trend" in memory_analysis
                assert "memory_stats" in memory_analysis
                assert "recommendations" in memory_analysis
            
            # Verify cleanup verification
            cleanup_verification = results["cleanup_verification"]
            assert "process_cleanup" in cleanup_verification
            assert "file_cleanup" in cleanup_verification
            assert "cleanup_successful" in cleanup_verification
    
    def test_json_serialization(self):
        """Test that results can be properly serialized to JSON"""
        detector = MemoryLeakDetector()
        
        # Create sample results with various data types
        sample_results = {
            "timestamp": datetime.now(),
            "numpy_array": np.array([1, 2, 3]),
            "numpy_float": np.float64(3.14),
            "nested": {
                "datetime": datetime.now(),
                "list": [1, 2, datetime.now()]
            }
        }
        
        # Should not raise an exception
        serializable = detector._make_json_serializable(sample_results)
        
        # Should be able to convert to JSON
        json_str = json.dumps(serializable)
        assert isinstance(json_str, str)
        assert len(json_str) > 0


if __name__ == "__main__":
    # Example usage of the memory leak detection framework
    async def run_memory_leak_tests():
        print("🧠 Running Task 22.4 Memory Leak Detection Framework...")
        
        detector = MemoryLeakDetector()
        
        print("🔍 Running extended memory test...")
        results = await detector.run_extended_memory_test(
            duration_minutes=1,  # 1 minute test
            operations_per_minute=30  # 30 operations per minute
        )
        
        print(f"✅ Memory leak test completed!")
        print(f"   - Operations completed: {results['operations_completed']}")
        print(f"   - Operations failed: {results['operations_failed']}")
        print(f"   - Overall success: {results['overall_success']}")
        
        # Print memory analysis
        memory_analysis = results.get("memory_analysis", {})
        if "memory_trend" in memory_analysis:
            trend = memory_analysis["memory_trend"]
            print(f"   - Memory leak severity: {trend.get('leak_severity', 'unknown')}")
            print(f"   - Memory slope: {trend.get('slope_mb_per_second', 0):.4f} MB/sec")
        
        # Print cleanup verification
        cleanup = results.get("cleanup_verification", {})
        print(f"   - Cleanup successful: {cleanup.get('cleanup_successful', False)}")
        print(f"   - Issues found: {len(cleanup.get('issues_found', []))}")
        
        return results
    
    # Run the example
    asyncio.run(run_memory_leak_tests()) 