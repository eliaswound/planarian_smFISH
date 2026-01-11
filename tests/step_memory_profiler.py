#!/usr/bin/env python3
"""
Step-by-step memory profiler to identify exact OOM location.
Tracks memory usage at each step and logs detailed information.
"""

import psutil
import os
import sys
import gc
import tracemalloc
import traceback
import functools
from pathlib import Path
from typing import Callable, Optional, Dict, List
from contextlib import contextmanager
import time

def format_size(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

class StepMemoryProfiler:
    """Tracks memory usage at each step of processing"""
    
    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.log_file = log_file
        self.verbose = verbose
        self.steps = []
        self.start_memory = None
        self.snapshots = []
        self.tracemalloc_active = False
        
    def _get_memory(self) -> Dict:
        """Get current memory usage"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        system_mem = psutil.virtual_memory()
        
        return {
            'rss': mem_info.rss,  # Resident Set Size
            'vms': mem_info.vms,  # Virtual Memory Size
            'percent': process.memory_percent(),
            'system_total': system_mem.total,
            'system_available': system_mem.available,
            'system_percent': system_mem.percent,
            'system_used': system_mem.used
        }
    
    def _log(self, message: str):
        """Log message to file and/or stdout"""
        if self.verbose:
            print(message)
        if self.log_file:
            try:
                # Ensure directory exists
                log_path = Path(self.log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                # Append to file (creates if doesn't exist)
                with open(self.log_file, 'a') as f:
                    f.write(message + '\n')
            except Exception as e:
                # If logging fails, at least print to stdout
                print(f"Warning: Could not write to log file {self.log_file}: {e}")
    
    def start(self):
        """Start memory profiling"""
        try:
            self.start_memory = self._get_memory()
            tracemalloc.start()
            self.tracemalloc_active = True
            
            self._log("\n" + "="*70)
            self._log("STEP-BY-STEP MEMORY PROFILING STARTED")
            self._log("="*70)
            self._log(f"Initial memory: RSS={format_size(self.start_memory['rss'])}, "
                     f"Available={format_size(self.start_memory['system_available'])}")
            self._log("="*70 + "\n")
        except Exception as e:
            # If start fails, log the error
            error_msg = f"ERROR: Profiler start failed: {e}"
            print(error_msg)
            if self.log_file:
                try:
                    Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
                    with open(self.log_file, 'w') as f:
                        f.write(error_msg + '\n')
                except:
                    pass
            raise
    
    def stop(self):
        """Stop memory profiling"""
        if self.tracemalloc_active:
            tracemalloc.stop()
            self.tracemalloc_active = False
        
        final_memory = self._get_memory()
        self._log("\n" + "="*70)
        self._log("MEMORY PROFILING SUMMARY")
        self._log("="*70)
        self._log(f"Initial RSS: {format_size(self.start_memory['rss'])}")
        self._log(f"Final RSS: {format_size(final_memory['rss'])}")
        self._log(f"Total increase: {format_size(final_memory['rss'] - self.start_memory['rss'])}")
        self._log("\nStep-by-step breakdown:")
        
        for i, step in enumerate(self.steps):
            self._log(f"\n  Step {i+1}: {step['name']}")
            self._log(f"    Location: {step['location']}")
            self._log(f"    Memory: {format_size(step['memory']['rss'])} "
                     f"(+{format_size(step['memory']['rss'] - (self.steps[i-1]['memory']['rss'] if i > 0 else self.start_memory['rss']))})")
            if step.get('peak_memory'):
                self._log(f"    Peak during step: {format_size(step['peak_memory'])}")
            if step.get('error'):
                self._log(f"    ERROR: {step['error']}")
        
        self._log("="*70 + "\n")
    
    def checkpoint(self, step_name: str, location: str = ""):
        """Record memory at a checkpoint"""
        mem = self._get_memory()
        
        # Get caller location if not provided
        if not location:
            try:
                frame = sys._getframe(1)
                location = f"{frame.f_code.co_filename}:{frame.f_lineno} in {frame.f_code.co_name}"
            except:
                location = "unknown"
        
        step_info = {
            'name': step_name,
            'location': location,
            'memory': mem,
            'timestamp': time.time()
        }
        
        self.steps.append(step_info)
        
        prev_rss = self.steps[-2]['memory']['rss'] if len(self.steps) > 1 else self.start_memory['rss']
        delta = mem['rss'] - prev_rss
        
        self._log(f"✓ Checkpoint: {step_name}")
        self._log(f"  Location: {location}")
        self._log(f"  Memory: RSS={format_size(mem['rss'])} "
                 f"(+{format_size(delta)}, {format_size(mem['system_available'])} available)")
        
        # Warning if large increase
        if delta > 1024 * 1024 * 1024:  # > 1GB
            self._log(f"  ⚠️  WARNING: Large memory increase detected!")
        
        # Take tracemalloc snapshot
        if self.tracemalloc_active:
            snapshot = tracemalloc.take_snapshot()
            self.snapshots.append((step_name, snapshot))
    
    @contextmanager
    def step(self, step_name: str, location: str = ""):
        """Context manager for tracking a step"""
        # Get caller location if not provided
        if not location:
            try:
                frame = sys._getframe(1)
                location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
            except:
                location = "unknown"
        
        # Record start
        mem_before = self._get_memory()
        self._log(f"\n{'='*70}")
        self._log(f"STEP: {step_name}")
        self._log(f"Location: {location}")
        self._log(f"Memory before: RSS={format_size(mem_before['rss'])}")
        self._log(f"{'='*70}")
        
        peak_memory = mem_before['rss']
        error = None
        
        try:
            yield
            mem_after = self._get_memory()
            peak_memory = max(peak_memory, mem_after['rss'])
            delta = mem_after['rss'] - mem_before['rss']
            
            self._log(f"\nSTEP COMPLETE: {step_name}")
            self._log(f"Memory after: RSS={format_size(mem_after['rss'])} "
                     f"(+{format_size(delta)})")
            
            if delta > 1024 * 1024 * 1024:  # > 1GB
                self._log(f"⚠️  WARNING: Step caused {format_size(delta)} memory increase!")
            
            # Record checkpoint
            self.checkpoint(step_name, location)
            self.steps[-1]['peak_memory'] = peak_memory
            
        except MemoryError as e:
            error = str(e)
            mem_after = self._get_memory()
            self._log(f"\n❌ OUT OF MEMORY in step: {step_name}")
            self._log(f"Error: {error}")
            self._log(f"Memory at failure: RSS={format_size(mem_after['rss'])}")
            self._log(f"Memory increase in this step: {format_size(mem_after['rss'] - mem_before['rss'])}")
            
            # Show top allocations
            if self.tracemalloc_active:
                try:
                    current_snapshot = tracemalloc.take_snapshot()
                    if len(self.snapshots) > 0:
                        prev_snapshot = self.snapshots[-1][1]
                        top_stats = current_snapshot.compare_to(prev_snapshot, 'lineno')
                        self._log("\nTop memory allocations in this step:")
                        for stat in top_stats[:10]:
                            frame = stat.traceback[0]
                            self._log(f"  {format_size(stat.size)} - {frame.filename}:{frame.lineno}")
                except:
                    pass
            
            self.checkpoint(f"{step_name} (FAILED)", location)
            self.steps[-1]['error'] = error
            raise
            
        except Exception as e:
            error = str(e)
            mem_after = self._get_memory()
            self._log(f"\n⚠️  ERROR in step: {step_name}")
            self._log(f"Error: {error}")
            self._log(f"Memory at error: RSS={format_size(mem_after['rss'])}")
            self.checkpoint(f"{step_name} (ERROR)", location)
            self.steps[-1]['error'] = error
            raise
        
        finally:
            # Force garbage collection after each step
            collected = gc.collect()
            if collected > 0:
                self._log(f"  Garbage collected {collected} objects")
    
    def get_top_allocations(self, step_index: int = -1) -> List:
        """Get top memory allocations for a step"""
        if not self.tracemalloc_active or not self.snapshots:
            return []
        
        if step_index < 0:
            step_index = len(self.snapshots) + step_index
        
        if step_index >= len(self.snapshots):
            return []
        
        if step_index == 0:
            # Compare to start
            snapshot = self.snapshots[0][1]
            top_stats = snapshot.statistics('lineno')
        else:
            # Compare to previous
            _, snapshot = self.snapshots[step_index]
            _, prev_snapshot = self.snapshots[step_index - 1]
            top_stats = snapshot.compare_to(prev_snapshot, 'lineno')
        
        results = []
        for stat in top_stats[:20]:
            frame = stat.traceback[0]
            results.append({
                'size': stat.size,
                'filename': frame.filename,
                'lineno': frame.lineno,
                'traceback': stat.traceback
            })
        
        return results


# Global profiler instance
_profiler: Optional[StepMemoryProfiler] = None

def initialize_profiler(log_file: Optional[str] = None, verbose: bool = True) -> StepMemoryProfiler:
    """Initialize the global profiler"""
    global _profiler
    _profiler = StepMemoryProfiler(log_file=log_file, verbose=verbose)
    _profiler.start()
    return _profiler

def get_profiler() -> Optional[StepMemoryProfiler]:
    """Get the global profiler instance"""
    return _profiler

def step(step_name: str):
    """Decorator to profile a function step"""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            if profiler:
                location = f"{func.__module__}.{func.__name__}"
                with profiler.step(step_name, location):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

if __name__ == '__main__':
    # Test the profiler
    profiler = StepMemoryProfiler(verbose=True)
    profiler.start()
    
    try:
        with profiler.step("Test step 1"):
            # Simulate some work
            data = [0] * 1000000
        
        with profiler.step("Test step 2"):
            # Simulate more work
            data2 = [0] * 5000000
        
        with profiler.step("Cleanup"):
            del data, data2
            gc.collect()
    
    finally:
        profiler.stop()
