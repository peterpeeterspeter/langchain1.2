"""
Agent Tools for CMS Agents
Tools exposed as LangChain tools for agent use
"""

from .research_tools import (
    web_search_tool,
    comprehensive_research_tool,
    screenshot_tool,
    casino_intelligence_tool,
)

from .writing_tools import (
    content_generation_tool,
    template_selection_tool,
    content_refinement_tool,
    seo_optimization_tool,
)

from .affiliate_tools import (
    affiliate_link_database_tool,
    link_insertion_tool,
    link_validation_tool,
    tracking_parameter_tool,
)

from .image_tools import (
    image_search_tool,
    image_selection_tool,
    alt_text_generation_tool,
    wordpress_image_upload_tool,
)

from .publishing_tools import (
    wordpress_publish_tool,
    site_registry_tool,
    content_adaptation_tool,
)

__all__ = [
    # Research tools
    "web_search_tool",
    "comprehensive_research_tool",
    "screenshot_tool",
    "casino_intelligence_tool",
    # Writing tools
    "content_generation_tool",
    "template_selection_tool",
    "content_refinement_tool",
    "seo_optimization_tool",
    # Affiliate tools
    "affiliate_link_database_tool",
    "link_insertion_tool",
    "link_validation_tool",
    "tracking_parameter_tool",
    # Image tools
    "image_search_tool",
    "image_selection_tool",
    "alt_text_generation_tool",
    "wordpress_image_upload_tool",
    # Publishing tools
    "wordpress_publish_tool",
    "site_registry_tool",
    "content_adaptation_tool",
]

