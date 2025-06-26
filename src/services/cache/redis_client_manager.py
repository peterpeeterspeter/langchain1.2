"""
Redis Client Manager Module.

This module provides a shared Redis client architecture for efficient connection
management across multiple cache instances. It implements connection pooling,
lifecycle management, and error handling for both synchronous and asynchronous
Redis operations.

Features include:
- Shared Redis client pool for multiple cache instances
- Connection lifecycle management (creation, reuse, cleanup)
- Support for both sync and async Redis clients
- Connection retry and error handling mechanisms
- Thread-safe implementation for multi-cache scenarios
- Optimized Redis client parameters for semantic caching workloads

The manager ensures efficient resource utilization while maintaining high
performance for cache operations across different content types.
""" 