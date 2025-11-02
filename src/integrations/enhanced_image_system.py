"""
Enhanced Image System for Agent-Based CMS
Multi-strategy image acquisition and contextual placement

Content Types:
- Casino Reviews: Playwright screenshots (logos, lobbies) - anti-scraping handling
- Game Reviews: DataForSEO/google search (aviator, crash, slots, live roulette)
- Bonus/General Articles: Gemini 2.5 Flash image generation or prompt-based search
"""

import logging
import re
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    from google import genai
    from google.genai import types
    from PIL import Image
    from io import BytesIO
    import base64
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None
    Image = None
    BytesIO = None

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """Content type detection for image strategy selection"""
    CASINO_REVIEW = "casino_review"
    GAME_REVIEW = "game_review"
    BONUS_ARTICLE = "bonus_article"
    GENERAL_ARTICLE = "general_article"
    UNKNOWN = "unknown"


@dataclass
class ImageStrategy:
    """Image acquisition strategy configuration"""
    content_type: ContentType
    use_playwright: bool = False
    use_dataforseo: bool = False
    use_gemini_generation: bool = False
    priority_order: List[str] = None  # Order of strategies to try
    
    def __post_init__(self):
        if self.priority_order is None:
            if self.content_type == ContentType.CASINO_REVIEW:
                self.priority_order = ["playwright", "dataforseo"]
                self.use_playwright = True
                self.use_dataforseo = True
            elif self.content_type == ContentType.GAME_REVIEW:
                self.priority_order = ["dataforseo"]
                self.use_dataforseo = True
            elif self.content_type == ContentType.BONUS_ARTICLE:
                self.priority_order = ["gemini", "dataforseo"]
                self.use_gemini_generation = True
                self.use_dataforseo = True
            else:  # GENERAL_ARTICLE
                self.priority_order = ["dataforseo", "gemini"]
                self.use_dataforseo = True
                self.use_gemini_generation = True


class ContentTypeDetector:
    """Detect content type from query and content"""
    
    CASINO_REVIEW_KEYWORDS = [
        "casino review", "casino rating", "best casino", "top casino",
        "trustdice", "betway", "stake", "roobet", "bc.game", "duelbits"
    ]
    
    GAME_REVIEW_KEYWORDS = [
        "aviator", "crash game", "crash gambling", "aviator game",
        "live roulette", "roulette review", "slot review", "slots",
        "pragmatic", "evolution gaming", "netent", "game review"
    ]
    
    BONUS_KEYWORDS = [
        "bonus", "promotion", "welcome bonus", "free spins",
        "deposit bonus", "no deposit", "cashback"
    ]
    
    @classmethod
    def detect(cls, query: str, content: str = "") -> ContentType:
        """Detect content type from query and content"""
        query_lower = query.lower()
        content_lower = content.lower() if content else ""
        combined = f"{query_lower} {content_lower}"
        
        # Casino review detection
        if any(keyword in combined for keyword in cls.CASINO_REVIEW_KEYWORDS):
            # Check if it's specifically about a casino site
            casino_patterns = [
                r'\b(casino|site|platform)\s+(review|rating|analysis)',
                r'(best|top|review)\s+\w+\s+casino',
                r'\w+\s+casino\s+(review|rating)'
            ]
            if any(re.search(pattern, combined) for pattern in casino_patterns):
                return ContentType.CASINO_REVIEW
        
        # Game review detection
        if any(keyword in combined for keyword in cls.GAME_REVIEW_KEYWORDS):
            game_patterns = [
                r'\b(game|slot|roulette|aviator|crash)\s+(review|guide|how to)',
                r'(play|review|best)\s+\w+\s+(game|slot)'
            ]
            if any(re.search(pattern, combined) for pattern in game_patterns):
                return ContentType.GAME_REVIEW
        
        # Bonus article detection
        if any(keyword in combined for keyword in cls.BONUS_KEYWORDS):
            return ContentType.BONUS_ARTICLE
        
        # Default to general article
        return ContentType.GENERAL_ARTICLE


class EnhancedImageSystem:
    """
    Enhanced image system with multi-strategy acquisition and contextual placement
    
    Features:
    - Content type detection
    - Multi-strategy image acquisition (Playwright, DataForSEO, Gemini)
    - Contextual HTML placement (hero, inline, gallery)
    - Bulletproof WordPress upload integration
    - Authoritative link references
    """
    
    def __init__(
        self,
        playwright_engine=None,
        dataforseo_client=None,
        gemini_api_key=None,
        bulletproof_uploader=None
    ):
        """
        Initialize enhanced image system
        
        Args:
            playwright_engine: Playwright screenshot engine for casino sites
            dataforseo_client: DataForSEO client for image search
            gemini_api_key: Gemini API key for image generation (or will use GOOGLE_API_KEY/GEMINI_API_KEY env var)
            bulletproof_uploader: Bulletproof image uploader for WordPress
        """
        self.playwright_engine = playwright_engine
        self.dataforseo_client = dataforseo_client
        self.gemini_api_key = gemini_api_key
        self.bulletproof_uploader = bulletproof_uploader
        
        self.content_detector = ContentTypeDetector()
    
    async def acquire_images(
        self,
        query: str,
        content: str,
        max_images: int = 5
    ) -> Tuple[List[Dict[str, Any]], ImageStrategy]:
        """
        Acquire images using appropriate strategy based on content type
        
        Returns:
            Tuple of (images list, strategy used)
        """
        # Detect content type
        content_type = self.content_detector.detect(query, content)
        strategy = ImageStrategy(content_type=content_type)
        
        logger.info(f"📸 Detected content type: {content_type.value}, using strategy: {strategy.priority_order}")
        
        images = []
        
        # Try strategies in priority order
        for strategy_name in strategy.priority_order:
            if len(images) >= max_images:
                break
            
            try:
                if strategy_name == "playwright" and strategy.use_playwright:
                    playwright_images = await self._acquire_playwright_images(
                        query, content, max_images - len(images)
                    )
                    images.extend(playwright_images)
                    
                elif strategy_name == "dataforseo" and strategy.use_dataforseo:
                    dataforseo_images = await self._acquire_dataforseo_images(
                        query, content, content_type, max_images - len(images)
                    )
                    images.extend(dataforseo_images)
                    
                elif strategy_name == "gemini" and strategy.use_gemini_generation:
                    gemini_images = await self._acquire_gemini_images(
                        query, content, max_images - len(images)
                    )
                    images.extend(gemini_images)
                    
            except Exception as e:
                logger.warning(f"⚠️ Strategy {strategy_name} failed: {e}, trying next strategy")
                continue
        
        # Limit to max_images
        images = images[:max_images]
        
        logger.info(f"✅ Acquired {len(images)} images using {content_type.value} strategy")
        return images, strategy
    
    async def _acquire_playwright_images(
        self,
        query: str,
        content: str,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """Acquire images using Playwright screenshot engine (for casino sites)"""
        if not self.playwright_engine:
            logger.warning("Playwright engine not available")
            return []
        
        try:
            # Extract casino domain from query/content
            casino_domain = self._extract_casino_domain(query, content)
            if not casino_domain:
                return []
            
            # Capture screenshots: logo, lobby, games section
            screenshots = []
            
            # Check if playwright engine has the right interface
            # Try different method names based on service type
            try:
                # Try stealth screenshot capture (StealthScreenshotService)
                if hasattr(self.playwright_engine, 'capture_stealth_screenshot'):
                    logo_result = await self.playwright_engine.capture_stealth_screenshot(
                        url=f"https://{casino_domain}",
                        capture_type="full_page"
                    )
                    if logo_result and logo_result.get("success"):
                        screenshot_data = logo_result.get("stealth_result", {}).get("screenshot_data")
                        if screenshot_data:
                            screenshots.append({
                                "url": f"data:image/png;base64,{screenshot_data}" if isinstance(screenshot_data, bytes) else screenshot_data,
                                "type": "lobby",
                                "alt_text": f"{casino_domain} casino lobby",
                                "source": "playwright"
                            })
                # Try screenshot service (ScreenshotService)
                elif hasattr(self.playwright_engine, 'capture_full_page_screenshot'):
                    lobby_result = await self.playwright_engine.capture_full_page_screenshot(
                        url=f"https://{casino_domain}"
                    )
                    if lobby_result:
                        # ScreenshotResult object
                        if hasattr(lobby_result, 'success') and lobby_result.success:
                            screenshot_data = getattr(lobby_result, 'screenshot_data', None)
                            if screenshot_data:
                                import base64
                                if isinstance(screenshot_data, bytes):
                                    screenshot_b64 = base64.b64encode(screenshot_data).decode()
                                    screenshots.append({
                                        "url": f"data:image/png;base64,{screenshot_b64}",
                                        "type": "lobby",
                                        "alt_text": f"{casino_domain} casino lobby",
                                        "source": "playwright",
                                        "width": getattr(lobby_result, 'width', 1920),
                                        "height": getattr(lobby_result, 'height', 1080)
                                    })
                        # Dictionary result
                        elif isinstance(lobby_result, dict) and lobby_result.get("success"):
                            screenshot_data = lobby_result.get("screenshot_data")
                            if screenshot_data:
                                import base64
                                if isinstance(screenshot_data, bytes):
                                    screenshot_b64 = base64.b64encode(screenshot_data).decode()
                                    screenshots.append({
                                        "url": f"data:image/png;base64,{screenshot_b64}",
                                        "type": "lobby",
                                        "alt_text": f"{casino_domain} casino lobby",
                                        "source": "playwright"
                                    })
                else:
                    logger.warning(f"Playwright engine doesn't have expected methods: {dir(self.playwright_engine)}")
                    
            except Exception as e:
                logger.warning(f"Playwright screenshot capture failed: {e}")
            
            return screenshots[:max_images]
            
        except Exception as e:
            logger.error(f"Playwright image acquisition failed: {e}")
            return []
    
    async def _acquire_dataforseo_images(
        self,
        query: str,
        content: str,
        content_type: ContentType,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """Acquire images using DataForSEO (for games, general content)"""
        if not self.dataforseo_client:
            logger.warning("DataForSEO client not available")
            return []
        
        try:
            from src.integrations.dataforseo_image_search import ImageSearchRequest, ImageType, ImageSize
            
            # Generate content-specific search queries
            search_queries = self._generate_content_specific_queries(query, content, content_type)
            
            all_images = []
            for search_query in search_queries[:3]:  # Max 3 searches
                try:
                    search_request = ImageSearchRequest(
                        keyword=search_query,
                        max_results=min(max_images, 10),
                        image_type=ImageType.PHOTO,
                        image_size=ImageSize.MEDIUM,
                        safe_search=True
                    )
                    
                    results = await self.dataforseo_client.search_images_async(search_request)
                    
                    for result in results[:max_images]:
                        all_images.append({
                            "url": result.url,
                            "title": result.title or search_query,
                            "alt_text": result.alt_text or result.title or search_query,
                            "width": result.width,
                            "height": result.height,
                            "file_size": result.file_size,
                            "quality_score": getattr(result, 'quality_score', 0.7),
                            "source": "dataforseo",
                            "type": "game" if content_type == ContentType.GAME_REVIEW else "general"
                        })
                    
                    if len(all_images) >= max_images:
                        break
                        
                except Exception as e:
                    logger.warning(f"DataForSEO search failed for '{search_query}': {e}")
                    continue
            
            return all_images[:max_images]
            
        except Exception as e:
            logger.error(f"DataForSEO image acquisition failed: {e}")
            return []
    
    async def _acquire_gemini_images(
        self,
        query: str,
        content: str,
        max_images: int
    ) -> List[Dict[str, Any]]:
        """
        Acquire images using Gemini 2.5 Flash native image generation
        
        Uses gemini-2.5-flash-image model for direct image generation.
        Reference: https://ai.google.dev/gemini-api/docs/image-generation
        """
        if not GEMINI_AVAILABLE:
            logger.warning("Gemini image generation not available (google.genai not installed)")
            return []
        
        try:
            import os
            
            # Initialize Gemini client
            api_key = self.gemini_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not api_key:
                logger.warning("Gemini API key not set (use gemini_api_key parameter or GOOGLE_API_KEY/GEMINI_API_KEY env var)")
                return []
            
            client = genai.Client(api_key=api_key)
            
            # Generate images based on content context
            images = []
            
            # Generate prompts from content
            image_prompts = self._generate_gemini_image_prompts(query, content, max_images)
            
            for prompt in image_prompts[:max_images]:
                try:
                    logger.info(f"🖼️ Generating image with Gemini: {prompt[:50]}...")
                    
                    # Generate image using gemini-2.5-flash-image
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-image",
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            image_config=types.ImageConfig(
                                aspect_ratio="16:9",  # Good for articles
                            )
                        )
                    )
                    
                    # Extract image from response
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            # Convert to PIL Image and save temporarily
                            image_data = part.inline_data.data
                            pil_image = Image.open(BytesIO(image_data))
                            
                            # Convert to base64 for storage/upload
                            img_buffer = BytesIO()
                            pil_image.save(img_buffer, format='PNG')
                            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                            
                            images.append({
                                "url": f"data:image/png;base64,{img_base64}",  # Base64 data URL
                                "title": prompt[:100],  # Use prompt as title
                                "alt_text": self._generate_alt_from_prompt(prompt),
                                "width": pil_image.width,
                                "height": pil_image.height,
                                "source": "gemini",
                                "type": "generated",
                                "quality_score": 0.9,  # Generated images are high quality
                                "raw_data": image_data  # Keep raw bytes for upload
                            })
                            
                            logger.info(f"✅ Generated image: {pil_image.width}x{pil_image.height}")
                            break
                            
                except Exception as e:
                    logger.warning(f"⚠️ Gemini image generation failed for prompt '{prompt[:50]}...': {e}")
                    continue
            
            logger.info(f"✅ Generated {len(images)} images with Gemini")
            return images
            
        except Exception as e:
            logger.error(f"Gemini image generation failed: {e}", exc_info=True)
            return []
    
    def _generate_gemini_image_prompts(
        self,
        query: str,
        content: str,
        max_images: int
    ) -> List[str]:
        """Generate optimized prompts for Gemini image generation"""
        prompts = []
        
        # Extract key themes from content
        content_preview = content[:1000].lower()
        
        # For bonus articles
        if "bonus" in query.lower() or "bonus" in content_preview:
            prompts.extend([
                "Professional illustration of casino welcome bonus offer, modern design, vibrant colors",
                "Casino bonus graphic with gift box and coins, promotional style",
                "Free spins promotion illustration, slot machine theme, premium quality"
            ])
        
        # For general casino content
        elif "casino" in query.lower() or "casino" in content_preview:
            prompts.extend([
                "Professional casino illustration, modern gaming theme, high quality",
                "Casino games collection illustration, cards and chips, premium style"
            ])
        
        # For game reviews
        elif any(game in content_preview for game in ["aviator", "crash", "roulette", "slot"]):
            game_name = next((game for game in ["aviator", "crash", "roulette", "slot"] if game in content_preview), "casino game")
            prompts.append(f"Professional {game_name} game illustration, modern design, high quality")
        
        # Generic fallback
        if not prompts:
            # Extract key terms from query
            key_terms = [w for w in query.lower().split() if len(w) > 4 and w not in ["review", "guide", "article"]][:3]
            if key_terms:
                prompts.append(f"Professional illustration: {', '.join(key_terms)}, modern design, high quality")
            else:
                prompts.append("Professional casino and gambling related illustration, modern design, high quality")
        
        return prompts[:max_images]
    
    def _generate_alt_from_prompt(self, prompt: str) -> str:
        """Generate alt text from image generation prompt"""
        # Clean up prompt for alt text
        alt = prompt.lower()
        # Remove common image generation phrases
        alt = alt.replace("professional illustration", "").replace("high quality", "").replace("modern design", "")
        alt = alt.replace("premium", "").replace("style", "").strip()
        # Capitalize first letter
        if alt:
            alt = alt[0].upper() + alt[1:]
        return alt[:125]  # Limit length
    
    def _generate_content_specific_queries(
        self,
        query: str,
        content: str,
        content_type: ContentType
    ) -> List[str]:
        """Generate content-specific image search queries"""
        queries = []
        
        if content_type == ContentType.GAME_REVIEW:
            # Extract game name
            game_match = re.search(r'\b(aviator|crash|roulette|slots?|blackjack|poker)\b', query.lower())
            if game_match:
                game_name = game_match.group(1)
                queries.extend([
                    f"{game_name} game screenshot",
                    f"{game_name} casino game",
                    f"{game_name} online gambling"
                ])
            
            # Provider-specific queries
            provider_match = re.search(r'\b(pragmatic|evolution|netent|microgaming|playtech)\b', query.lower())
            if provider_match:
                provider = provider_match.group(1)
                queries.append(f"{provider} gaming screenshot")
        
        elif content_type == ContentType.CASINO_REVIEW:
            # Casino-specific queries
            casino_name = query.split()[0] if query.split() else "casino"
            queries.extend([
                f"{casino_name} casino logo",
                f"{casino_name} casino lobby",
                f"{casino_name} gaming platform"
            ])
        
        elif content_type == ContentType.BONUS_ARTICLE:
            queries.extend([
                "casino bonus illustration",
                "welcome bonus graphic",
                "free spins promotion"
            ])
        
        else:  # GENERAL_ARTICLE
            # Extract key terms
            key_terms = [w for w in query.lower().split() if len(w) > 4][:3]
            if key_terms:
                queries.append(" ".join(key_terms))
        
        # Fallback to original query
        if not queries:
            queries.append(query)
        
        return queries
    
    def _extract_casino_domain(self, query: str, content: str) -> Optional[str]:
        """Extract casino domain from query/content"""
        # Look for domain patterns
        domain_pattern = r'\b(\w+\.(com|io|net|org|co\.uk))\b'
        
        matches = re.findall(domain_pattern, f"{query} {content}".lower())
        if matches:
            return matches[0][0]
        
        # Look for casino name patterns
        casino_patterns = [
            r'\b(\w+)\s+casino',
            r'casino\s+(\w+)',
            r'review\s+(\w+)'
        ]
        
        for pattern in casino_patterns:
            match = re.search(pattern, query.lower())
            if match:
                casino_name = match.group(1)
                # Common TLDs
                for tld in ['com', 'io', 'net']:
                    return f"{casino_name}.{tld}"
        
        return None
    
    def embed_images_contextually(
        self,
        content: str,
        images: List[Dict[str, Any]],
        content_type: ContentType
    ) -> str:
        """
        Embed images into HTML content with contextual placement
        
        Placement strategy:
        - Hero image: After title/first paragraph
        - Inline images: After relevant section headers
        - Gallery: Remaining images at end
        """
        if not images or not BS4_AVAILABLE:
            return content
        
        try:
            # Ensure content is HTML
            if not content.strip().startswith('<'):
                # Convert markdown/plain text to HTML first
                content = self._preprocess_content_to_html(content)
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Strategy 1: Hero image
            if images:
                hero_image = self._select_hero_image(images, content_type)
                if hero_image:
                    hero_html = self._create_hero_image_html(hero_image)
                    hero_soup = BeautifulSoup(hero_html, 'html.parser')
                    
                    # Insert after first h1 or first paragraph
                    first_h1 = soup.find('h1')
                    if first_h1:
                        first_h1.insert_after(hero_soup)
                    else:
                        first_p = soup.find('p')
                        if first_p:
                            first_p.insert_after(hero_soup)
                        else:
                            soup.insert(0, hero_soup)
            
            # Strategy 2: Inline images after relevant sections
            remaining_images = images[1:] if images else []
            if remaining_images:
                self._embed_inline_images(soup, remaining_images, content_type)
            
            # Strategy 3: Gallery for remaining images
            gallery_images = remaining_images[2:] if len(remaining_images) > 2 else []
            if gallery_images:
                gallery_html = self._create_gallery_html(gallery_images)
                gallery_soup = BeautifulSoup(gallery_html, 'html.parser')
                soup.append(gallery_soup)
            
            return str(soup)
            
        except Exception as e:
            logger.error(f"Failed to embed images contextually: {e}")
            return content
    
    def _preprocess_content_to_html(self, content: str) -> str:
        """Convert markdown/plain text to basic HTML"""
        if not BS4_AVAILABLE:
            return content
        
        # Simple markdown to HTML conversion
        lines = content.split('\n')
        html_parts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                html_parts.append('<p></p>')
                continue
            
            # Headers
            if line.startswith('# '):
                html_parts.append(f'<h1>{line[2:]}</h1>')
            elif line.startswith('## '):
                html_parts.append(f'<h2>{line[3:]}</h2>')
            elif line.startswith('### '):
                html_parts.append(f'<h3>{line[4:]}</h3>')
            # Lists
            elif line.startswith('- ') or line.startswith('* '):
                html_parts.append(f'<li>{line[2:]}</li>')
            # Paragraphs
            else:
                html_parts.append(f'<p>{line}</p>')
        
        return '\n'.join(html_parts)
    
    def _select_hero_image(
        self,
        images: List[Dict[str, Any]],
        content_type: ContentType
    ) -> Optional[Dict[str, Any]]:
        """Select best hero image based on content type"""
        if not images:
            return None
        
        # Score images for hero placement
        scored = []
        for img in images:
            score = 0
            
            # Content type preferences
            if content_type == ContentType.CASINO_REVIEW:
                if img.get("type") == "logo":
                    score += 5
                elif img.get("type") == "lobby":
                    score += 3
            elif content_type == ContentType.GAME_REVIEW:
                if img.get("type") == "game":
                    score += 5
            
            # Quality score
            score += img.get("quality_score", 0) * 2
            
            # Landscape orientation preference
            width = img.get("width", 0)
            height = img.get("height", 0)
            if width > height and width >= 800:
                score += 2
            
            scored.append((score, img))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored else images[0]
    
    def _create_hero_image_html(self, image: Dict[str, Any]) -> str:
        """Create hero image HTML"""
        url = image.get("url", "")
        alt = image.get("alt_text", image.get("title", "Featured image"))
        title = image.get("title", "")
        
        return f"""
<div class="hero-image-container" style="margin: 2rem 0; text-align: center;">
    <img src="{url}" 
         alt="{alt}" 
         title="{title}"
         class="hero-image wp-image"
         style="width: 100%; max-width: 1200px; height: auto;"
         loading="eager" />
    {f'<p class="hero-caption" style="font-style: italic; margin-top: 0.5rem; color: #666;">{title}</p>' if title else ''}
</div>
"""
    
    def _embed_inline_images(
        self,
        soup: BeautifulSoup,
        images: List[Dict[str, Any]],
        content_type: ContentType
    ):
        """Embed images inline after relevant sections"""
        headers = soup.find_all(['h2', 'h3'])
        images_used = 0
        max_inline = min(len(images), len(headers), 3)
        
        for i, header in enumerate(headers):
            if images_used >= max_inline:
                break
            
            img = images[images_used]
            img_html = self._create_inline_image_html(img)
            img_soup = BeautifulSoup(img_html, 'html.parser')
            
            # Insert after header's next paragraph
            next_p = header.find_next_sibling('p')
            if next_p:
                next_p.insert_after(img_soup)
            else:
                header.insert_after(img_soup)
            
            images_used += 1
    
    def _create_inline_image_html(self, image: Dict[str, Any]) -> str:
        """Create inline image HTML"""
        url = image.get("url", "")
        alt = image.get("alt_text", image.get("title", "Content image"))
        title = image.get("title", "")
        
        return f"""
<div class="content-image-container" style="margin: 1.5rem 0; text-align: center;">
    <img src="{url}" 
         alt="{alt}" 
         title="{title}"
         class="content-image wp-image"
         style="width: 100%; max-width: 800px; height: auto;"
         loading="lazy" />
    {f'<p class="image-caption" style="font-style: italic; margin-top: 0.5rem; color: #666; font-size: 0.9em;">{title}</p>' if title else ''}
</div>
"""
    
    def _create_gallery_html(self, images: List[Dict[str, Any]]) -> str:
        """Create gallery section HTML"""
        gallery_items = []
        for img in images:
            url = img.get("url", "")
            alt = img.get("alt_text", img.get("title", "Gallery image"))
            title = img.get("title", "")
            
            gallery_items.append(f"""
    <div class="gallery-item" style="margin: 1rem; text-align: center;">
        <img src="{url}" 
             alt="{alt}" 
             title="{title}"
             class="gallery-image wp-image"
             style="width: 100%; max-width: 400px; height: auto;"
             loading="lazy" />
        {f'<p class="gallery-caption" style="font-size: 0.85em; margin-top: 0.5rem;">{title}</p>' if title else ''}
    </div>
""")
        
        return f"""
<div class="image-gallery" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; margin: 2rem 0;">
    {''.join(gallery_items)}
</div>
"""

