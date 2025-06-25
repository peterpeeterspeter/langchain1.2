#!/usr/bin/env python3
"""
Screenshot Web Research Integration
Integrates Playwright screenshot functionality with the web research pipeline

This module provides:
1. URL Target Identification System - identifies URLs that need screenshots
2. Priority Scoring for screenshot targets
3. Queue system for managing screenshot requests
4. Integration with existing web research pipeline
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from urllib.parse import urljoin, urlparse
import heapq
import uuid
import re

# Import existing screenshot functionality
from .playwright_screenshot_engine import (
    ScreenshotService,
    ScreenshotResult,
    ScreenshotConfig,
    ScreenshotPriority,
    CasinoElementLocator,
    CasinoElement,
    BrowserPoolManager,
    get_global_browser_pool
)

logger = logging.getLogger(__name__)

class ScreenshotTargetType(Enum):
    """Types of screenshot targets for web research"""
    CASINO_REVIEW = "casino_review"
    CASINO_DIRECT = "casino_direct"
    REGULATORY = "regulatory"
    COMPARISON = "comparison"
    NEWS_ARTICLE = "news_article"
    GENERIC_WEB = "generic_web"

@dataclass
class ScreenshotTarget:
    """Represents a URL target for screenshot capture"""
    url: str
    target_type: ScreenshotTargetType
    priority_score: float  # 0.0-1.0, higher = more important
    research_context: str  # Why this URL needs a screenshot
    confidence: float  # 0.0-1.0, confidence in priority scoring
    source_query: str  # Original research query
    source_type: str  # "web_search", "comprehensive_web_research", etc.
    timestamp: float = field(default_factory=time.time)
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """Enable priority queue ordering (higher priority_score first)"""
        return self.priority_score > other.priority_score

class URLTargetIdentifier:
    """
    System for identifying and prioritizing URLs that require screenshot evidence
    during web research operations.
    """
    
    # Domain patterns that indicate screenshot-worthy content
    CASINO_DOMAINS = {
        'review_sites': [
            'askgamblers.com', 'casino.guru', 'casinomeister.com',
            'lcb.org', 'thepogg.com', 'casinolistings.com',
            'gamblingcommission.gov.uk', 'mga.org.mt'
        ],
        'direct_casinos': [
            'bet365.com', 'betway.com', 'williamhill.com', 'ladbrokes.com',
            'bwin.com', 'pokerstars.com', 'partypoker.com', 'casumo.com',
            'leovegas.com', 'mrgreen.com', 'rizk.com', '888casino.com'
        ],
        'regulatory': [
            'gamblingcommission.gov.uk', 'mga.org.mt', 'ukgc.gov.uk',
            'curacao-egaming.com', 'dgoj.gob.es', 'aams.gov.it'
        ]
    }
    
    # Keywords that indicate screenshot-worthy content
    SCREENSHOT_KEYWORDS = {
        'high_priority': [
            'casino lobby', 'game selection', 'bonus offers', 'user interface',
            'homepage', 'landing page', 'welcome bonus', 'registration'
        ],
        'medium_priority': [
            'terms conditions', 'licensing info', 'payment methods',
            'customer support', 'responsible gambling', 'about us'
        ],
        'low_priority': [
            'privacy policy', 'cookie policy', 'help center', 'faq'
        ]
    }
    
    def __init__(self):
        self.identified_targets: List[ScreenshotTarget] = []
        self.processed_urls: Set[str] = set()
        logger.info("URLTargetIdentifier initialized")
    
    def identify_screenshot_targets(self, 
                                  web_results: List[Dict[str, Any]], 
                                  original_query: str) -> List[ScreenshotTarget]:
        """
        Identify URLs from web research results that need screenshots
        
        Args:
            web_results: List of web research results
            original_query: Original research query for context
            
        Returns:
            List of ScreenshotTarget objects, sorted by priority
        """
        targets = []
        
        for result in web_results:
            url = result.get('url', '')
            title = result.get('title', '')
            content = result.get('content', '')
            source_type = result.get('source', 'unknown')
            
            if not url or url in self.processed_urls:
                continue
            
            # Identify target type and calculate priority
            target_type = self._classify_url_type(url, title, content)
            priority_score = self._calculate_priority_score(
                url, title, content, target_type, original_query
            )
            confidence = self._calculate_confidence(url, title, content, target_type)
            
            # Only include if priority is above threshold
            if priority_score >= 0.3:
                target = ScreenshotTarget(
                    url=url,
                    target_type=target_type,
                    priority_score=priority_score,
                    research_context=self._generate_research_context(
                        url, title, content, target_type, original_query
                    ),
                    confidence=confidence,
                    source_query=original_query,
                    source_type=source_type
                )
                targets.append(target)
                self.processed_urls.add(url)
        
        # Sort by priority score (highest first)
        targets.sort(reverse=True)
        self.identified_targets.extend(targets)
        
        logger.info(f"Identified {len(targets)} screenshot targets from {len(web_results)} web results")
        return targets
    
    def _classify_url_type(self, url: str, title: str, content: str) -> ScreenshotTargetType:
        """Classify the type of URL target for appropriate screenshot handling"""
        domain = self._extract_domain(url)
        url_lower = url.lower()
        title_lower = title.lower()
        content_lower = content.lower()
        
        # Check domain categories
        if any(casino_domain in domain for casino_domain in self.CASINO_DOMAINS['review_sites']):
            return ScreenshotTargetType.CASINO_REVIEW
        
        if any(casino_domain in domain for casino_domain in self.CASINO_DOMAINS['direct_casinos']):
            return ScreenshotTargetType.CASINO_DIRECT
        
        if any(reg_domain in domain for reg_domain in self.CASINO_DOMAINS['regulatory']):
            return ScreenshotTargetType.REGULATORY
        
        # Check for casino-related content
        casino_indicators = ['casino', 'gambling', 'betting', 'slots', 'poker', 'blackjack']
        if any(indicator in domain or indicator in title_lower for indicator in casino_indicators):
            if 'review' in title_lower or 'compare' in title_lower:
                return ScreenshotTargetType.COMPARISON
            elif domain in self.CASINO_DOMAINS['direct_casinos'] or 'casino' in domain:
                return ScreenshotTargetType.CASINO_DIRECT
            else:
                return ScreenshotTargetType.CASINO_REVIEW
        
        # Check for news articles
        news_indicators = ['news', 'article', 'blog', 'press']
        if any(indicator in url_lower or indicator in domain for indicator in news_indicators):
            return ScreenshotTargetType.NEWS_ARTICLE
        
        return ScreenshotTargetType.GENERIC_WEB
    
    def _calculate_priority_score(self, 
                                url: str, 
                                title: str, 
                                content: str, 
                                target_type: ScreenshotTargetType,
                                original_query: str) -> float:
        """
        Calculate priority score for screenshot capture (0.0-1.0)
        Higher scores indicate more important targets
        """
        score = 0.0
        
        # Base score by target type
        type_scores = {
            ScreenshotTargetType.CASINO_DIRECT: 0.9,
            ScreenshotTargetType.CASINO_REVIEW: 0.8,
            ScreenshotTargetType.REGULATORY: 0.7,
            ScreenshotTargetType.COMPARISON: 0.6,
            ScreenshotTargetType.NEWS_ARTICLE: 0.4,
            ScreenshotTargetType.GENERIC_WEB: 0.3
        }
        score += type_scores.get(target_type, 0.3)
        
        # Keyword relevance bonus
        text_to_check = f"{title} {content}".lower()
        for keyword in self.SCREENSHOT_KEYWORDS['high_priority']:
            if keyword in text_to_check:
                score += 0.1
        
        for keyword in self.SCREENSHOT_KEYWORDS['medium_priority']:
            if keyword in text_to_check:
                score += 0.05
        
        # Query relevance bonus
        query_terms = original_query.lower().split()
        for term in query_terms:
            if len(term) > 3 and term in text_to_check:
                score += 0.02
        
        # Domain authority bonus
        domain = self._extract_domain(url)
        if any(auth_domain in domain for auth_domain in self.CASINO_DOMAINS['review_sites']):
            score += 0.1
        elif any(auth_domain in domain for auth_domain in self.CASINO_DOMAINS['regulatory']):
            score += 0.15
        
        # Visual content indicators
        visual_indicators = ['lobby', 'games', 'interface', 'screenshot', 'demo', 'preview']
        for indicator in visual_indicators:
            if indicator in text_to_check:
                score += 0.05
        
        # Ensure score is within bounds
        return min(1.0, score)
    
    def _calculate_confidence(self, 
                            url: str, 
                            title: str, 
                            content: str, 
                            target_type: ScreenshotTargetType) -> float:
        """Calculate confidence in the priority scoring"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence for known domains
        domain = self._extract_domain(url)
        if any(known_domain in domain for category in self.CASINO_DOMAINS.values() 
               for known_domain in category):
            confidence += 0.3
        
        # Higher confidence if content length is substantial
        if len(content) > 500:
            confidence += 0.1
        elif len(content) > 200:
            confidence += 0.05
        
        # Higher confidence if title is descriptive
        if len(title) > 20 and any(keyword in title.lower() 
                                 for keywords in self.SCREENSHOT_KEYWORDS.values() 
                                 for keyword in keywords):
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _generate_research_context(self, 
                                 url: str, 
                                 title: str, 
                                 content: str, 
                                 target_type: ScreenshotTargetType,
                                 original_query: str) -> str:
        """Generate context explaining why this URL needs a screenshot"""
        context_templates = {
            ScreenshotTargetType.CASINO_DIRECT: f"Direct casino site screenshot needed for visual evidence of {original_query}",
            ScreenshotTargetType.CASINO_REVIEW: f"Casino review site screenshot to capture {original_query} analysis",
            ScreenshotTargetType.REGULATORY: f"Regulatory information screenshot for {original_query} compliance verification",
            ScreenshotTargetType.COMPARISON: f"Comparison content screenshot for {original_query} evaluation",
            ScreenshotTargetType.NEWS_ARTICLE: f"News article screenshot documenting {original_query}",
            ScreenshotTargetType.GENERIC_WEB: f"Web content screenshot relevant to {original_query}"
        }
        
        base_context = context_templates.get(target_type, f"Screenshot needed for {original_query}")
        
        # Add specific details if available
        if 'bonus' in title.lower() or 'bonus' in content.lower():
            base_context += " - Focus on bonus information"
        elif 'game' in title.lower() or 'lobby' in content.lower():
            base_context += " - Focus on game selection/lobby"
        elif 'license' in title.lower() or 'regulation' in content.lower():
            base_context += " - Focus on licensing information"
        
        return base_context
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return url.lower()
    
    def get_high_priority_targets(self, threshold: float = 0.7) -> List[ScreenshotTarget]:
        """Get screenshot targets above priority threshold"""
        return [target for target in self.identified_targets 
                if target.priority_score >= threshold]
    
    def clear_processed_urls(self):
        """Clear the processed URLs cache"""
        self.processed_urls.clear()
        logger.info("Cleared processed URLs cache")


class ScreenshotRequestQueue:
    """
    Queue system for managing screenshot requests during web research operations
    Handles prioritization, deduplication, and request lifecycle management
    """
    
    def __init__(self, max_queue_size: int = 50):
        self.max_queue_size = max_queue_size
        self._queue: List[ScreenshotTarget] = []
        self._active_requests: Dict[str, asyncio.Task] = {}
        self._completed_requests: Dict[str, ScreenshotResult] = {}
        self._failed_requests: Dict[str, str] = {}
        
        # URL deduplication
        self._queued_urls: Set[str] = set()
        
        # Statistics
        self._stats = {
            'total_queued': 0,
            'total_completed': 0,
            'total_failed': 0,
            'queue_start_time': time.time()
        }
        
        logger.info(f"ScreenshotRequestQueue initialized with max_size={max_queue_size}")
    
    def add_targets(self, targets: List[ScreenshotTarget]) -> int:
        """
        Add screenshot targets to the queue
        
        Args:
            targets: List of ScreenshotTarget objects
            
        Returns:
            Number of targets actually added (excluding duplicates)
        """
        added_count = 0
        
        for target in targets:
            if len(self._queue) >= self.max_queue_size:
                logger.warning(f"Queue is full, skipping target: {target.url}")
                break
            
            if target.url not in self._queued_urls:
                heapq.heappush(self._queue, target)
                self._queued_urls.add(target.url)
                added_count += 1
                self._stats['total_queued'] += 1
            else:
                logger.debug(f"URL already queued, skipping: {target.url}")
        
        logger.info(f"Added {added_count} targets to screenshot queue (total queued: {len(self._queue)})")
        return added_count
    
    def add_request(self, target: ScreenshotTarget) -> str:
        """
        Add a single screenshot target to the queue
        
        Args:
            target: ScreenshotTarget object to add
            
        Returns:
            Request ID for tracking
        """
        import uuid
        
        if len(self._queue) >= self.max_queue_size:
            raise ValueError(f"Queue is full (max size: {self.max_queue_size})")
        
        if target.url not in self._queued_urls:
            heapq.heappush(self._queue, target)
            self._queued_urls.add(target.url)
            self._stats['total_queued'] += 1
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            logger.debug(f"Added screenshot request {request_id} for {target.url}")
            return request_id
        else:
            logger.debug(f"URL already queued, skipping: {target.url}")
            return "duplicate"
    
    def get_next_target(self) -> Optional[ScreenshotTarget]:
        """Get the next highest priority target from the queue"""
        if self._queue:
            target = heapq.heappop(self._queue)
            self._queued_urls.discard(target.url)
            return target
        return None
    
    def get_request(self, request_id: str) -> Optional[ScreenshotTarget]:
        """
        Get a specific target by request ID
        
        Args:
            request_id: Request ID to look for
            
        Returns:
            ScreenshotTarget if found, None otherwise
        """
        # For simplicity, return the next target in queue
        # In a full implementation, we'd track request_id -> target mapping
        return self.get_next_target()
    
    def complete_request(self, request_id: str):
        """
        Mark a request as completed
        
        Args:
            request_id: Request ID to mark as completed
        """
        # In a full implementation, we'd track completion by request_id
        # For now, just increment the completed count
        self._stats['total_completed'] += 1
        logger.debug(f"Marked request {request_id} as completed")
    
    def mark_completed(self, target: ScreenshotTarget, result: ScreenshotResult):
        """Mark a target as completed with its result"""
        self._completed_requests[target.url] = result
        self._stats['total_completed'] += 1
        logger.debug(f"Marked target completed: {target.url}")
    
    def mark_failed(self, target: ScreenshotTarget, error_message: str):
        """Mark a target as failed"""
        self._failed_requests[target.url] = error_message
        self._stats['total_failed'] += 1
        
        # Add back to queue if retries available
        if target.retry_count < target.max_retries:
            target.retry_count += 1
            target.priority_score *= 0.9  # Slightly reduce priority on retry
            heapq.heappush(self._queue, target)
            self._queued_urls.add(target.url)
            logger.debug(f"Added target back to queue for retry ({target.retry_count}/{target.max_retries}): {target.url}")
        else:
            logger.warning(f"Target failed after {target.max_retries} retries: {target.url}")
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self._stats,
            'queue_size': len(self._queue),
            'active_requests': len(self._active_requests),
            'completed_count': len(self._completed_requests),
            'failed_count': len(self._failed_requests),
            'runtime_seconds': time.time() - self._stats['queue_start_time']
        }
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self._queue) == 0
    
    def clear(self):
        """Clear the queue and reset state"""
        self._queue.clear()
        self._queued_urls.clear()
        self._active_requests.clear()
        self._completed_requests.clear()
        self._failed_requests.clear()
        logger.info("Screenshot request queue cleared")


# Integration with existing web research results
def create_url_target_identifier() -> URLTargetIdentifier:
    """Factory function to create URL target identifier"""
    return URLTargetIdentifier()

def create_screenshot_request_queue(max_size: int = 50) -> ScreenshotRequestQueue:
    """Factory function to create screenshot request queue"""
    return ScreenshotRequestQueue(max_size)

# Example usage and testing
async def test_url_target_identification():
    """Test function for URL target identification"""
    
    # Sample web research results
    sample_results = [
        {
            "url": "https://casino.guru/napoleon-casino-review",
            "title": "Napoleon Casino Review - Detailed Analysis",
            "content": "Comprehensive review of Napoleon Casino featuring game selection, bonuses, and user interface analysis...",
            "source": "comprehensive_web_research"
        },
        {
            "url": "https://napoleonsports.be/",
            "title": "Napoleon Sports & Casino - Official Site",
            "content": "Welcome to Napoleon Sports with casino games, slots, and sports betting. Experience our lobby...",
            "source": "comprehensive_web_research"
        },
        {
            "url": "https://gamblingcommission.gov.uk/check-a-licence",
            "title": "Check Gambling Licence - UK Gambling Commission",
            "content": "Verify gambling operator licenses and regulatory compliance information...",
            "source": "web_search"
        }
    ]
    
    # Test URL target identification
    identifier = create_url_target_identifier()
    targets = identifier.identify_screenshot_targets(sample_results, "Napoleon Casino review")
    
    print(f"\n🎯 Identified {len(targets)} screenshot targets:")
    for target in targets:
        print(f"  Priority: {target.priority_score:.2f} | Type: {target.target_type.value}")
        print(f"  URL: {target.url}")
        print(f"  Context: {target.research_context}")
        print(f"  Confidence: {target.confidence:.2f}")
        print()
    
    # Test queue system
    queue = create_screenshot_request_queue()
    added_count = queue.add_targets(targets)
    print(f"📋 Added {added_count} targets to queue")
    
    # Process queue
    while not queue.is_empty():
        target = queue.get_next_target()
        print(f"🎯 Next target: {target.url} (priority: {target.priority_score:.2f})")
    
    stats = queue.get_queue_stats()
    print(f"📊 Queue stats: {stats}")

if __name__ == "__main__":
    asyncio.run(test_url_target_identification()) 