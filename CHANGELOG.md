# Changelog

All notable changes to the Universal RAG LCEL Chain project will be documented in this file.

## [2.0.0] - 2025-06-26

### 🚀 Major Features Added

#### **LangChain Hub Integration**
- **Native Template System**: Replaced complex ImprovedTemplateManager with simple dictionary-based local hub
- **ChatPromptTemplate Support**: Templates now return proper ChatPromptTemplate objects
- **Community Integration**: Support for pulling templates from LangChain Hub with fallbacks
- **Local Hub Pattern**: Easy template management following LangChain best practices

#### **WordPress Integration Rules**
- **Cursor Rules**: Added comprehensive .cursor/rules/wordpress-integration.md
- **Native Patterns**: Simple chain composition over complex classes
- **PydanticOutputParser**: Structured WordPress content generation
- **LCEL Integration**: WordPress publishing as RunnableLambda in main chain
- **Error Handling**: Graceful fallbacks with OutputFixingParser

### 🔧 Critical ROOT Fixes

#### **Template System v2.0 Actually Working**
- **FIXED**: _select_optimal_template was using hardcoded templates, never calling Template System v2.0
- **NEW**: Actually calls _get_enhanced_template_v2 when enabled
- **IMPROVED**: Proper template selection based on query analysis

#### **WordPress Auto-Publishing**
- **FIXED**: Publishing required manual publish_to_wordpress=True parameter
- **NEW**: Auto-publishing for casino reviews when WordPress enabled
- **IMPROVED**: Intelligent content-type detection for automatic publishing

#### **Casino-Specific Content Generation**
- **FIXED**: System generated generic "casino" content instead of Betsson-specific
- **NEW**: Proper casino name extraction from queries
- **IMPROVED**: Casino intelligence summary uses actual casino names

### ⚡ Performance Improvements
- **NEW**: Proper LCEL chain with ChatPromptTemplate integration
- **NEW**: Native LangChain RedisSemanticCache integration
- **DEPRECATED**: Custom QueryAwareCache (will be removed in future version)

### 📝 Code Quality Improvements
- **RULE**: Simple chain composition over complex classes
- **RULE**: Use LangChain Hub instead of custom templates
- **RULE**: PydanticOutputParser for structured data
- **RULE**: Native LangChain primitives over custom implementations

---

**Built with ❤️ using native LangChain patterns and modern RAG techniques.**
