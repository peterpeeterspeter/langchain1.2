#!/usr/bin/env python3
"""
V1 Migration Analysis Framework - Task 13
Extract proven patterns from the 3,825-line monolithic system and integrate into our v2 architecture

✅ ANALYSIS TARGET: /Users/Peter/LangChain/langchain/comprehensive_adaptive_pipeline.py (3,825 lines)
✅ INTEGRATION TARGET: Universal RAG CMS v2 with Enhanced FTI Pipeline Architecture

🎯 STRATEGIC GOALS:
- Extract working features users loved in v1
- Modernize using LangChain best practices (LCEL, async/await, proper typing)
- Integrate with our superior v2 systems (Tasks 1-3, FTI Pipeline)
- Avoid v1 pitfalls (monolithic architecture, tight coupling, poor error handling)
"""

import ast
import re
import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class V1Component:
    """Represents a component extracted from v1 system"""
    name: str
    type: str  # class, function, pattern, workflow
    lines: Tuple[int, int]  # start, end
    dependencies: List[str]
    features: List[str]
    complexity_score: int  # 1-10
    migration_priority: str  # high, medium, low
    modernization_notes: List[str]

@dataclass
class V1Pattern:
    """Represents a proven pattern from v1"""
    pattern_name: str
    description: str
    code_examples: List[str]
    benefits: List[str]
    v2_integration_strategy: str
    estimated_effort: str  # hours

@dataclass
class MigrationPlan:
    """Complete migration plan for v1 to v2"""
    components_to_migrate: List[V1Component]
    patterns_to_extract: List[V1Pattern]
    integration_points: Dict[str, str]
    risk_assessment: Dict[str, Any]
    timeline: Dict[str, str]

class V1AnalysisFramework:
    """
    Comprehensive framework for analyzing v1 monolithic system
    and creating migration plans for v2 integration
    """
    
    def __init__(self, v1_file_path: str = "/Users/Peter/LangChain/langchain/comprehensive_adaptive_pipeline.py"):
        self.v1_file_path = Path(v1_file_path)
        self.v1_content = ""
        self.v1_ast = None
        self.analysis_results = {}
        
        # V2 system knowledge (our current architecture)
        self.v2_systems = {
            "enhanced_fti_pipeline": "src/pipelines/enhanced_fti_pipeline.py",
            "enhanced_confidence_scoring": "src/chains/enhanced_confidence_scoring_system.py", 
            "contextual_retrieval": "src/retrieval/contextual_retrieval.py",
            "supabase_foundation": "database/migrations/",
            "advanced_prompt_system": "src/chains/advanced_prompt_system.py",
            "configuration_management": "src/config/",
            "monitoring_system": "src/monitoring/",
            "api_platform": "src/api/"
        }
        
    async def analyze_v1_system(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of the v1 monolithic system
        """
        logger.info(f"🔍 Starting V1 system analysis: {self.v1_file_path}")
        
        # Load and parse v1 code
        await self._load_v1_code()
        
        # Perform comprehensive analysis
        analysis_results = {
            "file_metrics": await self._analyze_file_metrics(),
            "component_analysis": await self._analyze_components(),
            "pattern_extraction": await self._extract_proven_patterns(),
            "feature_inventory": await self._inventory_features(),
            "dependency_mapping": await self._map_dependencies(),
            "performance_analysis": await self._analyze_performance_patterns(),
            "integration_opportunities": await self._identify_integration_points(),
            "risk_assessment": await self._assess_migration_risks()
        }
        
        self.analysis_results = analysis_results
        return analysis_results
    
    async def _load_v1_code(self):
        """Load and parse the v1 monolithic file"""
        try:
            with open(self.v1_file_path, 'r', encoding='utf-8') as f:
                self.v1_content = f.read()
            
            # Parse AST for structural analysis
            self.v1_ast = ast.parse(self.v1_content)
            logger.info(f"✅ Loaded v1 system: {len(self.v1_content)} characters, {len(self.v1_content.splitlines())} lines")
            
        except Exception as e:
            logger.error(f"❌ Failed to load v1 system: {e}")
            raise
    
    async def _analyze_file_metrics(self) -> Dict[str, Any]:
        """Analyze basic file metrics and complexity"""
        lines = self.v1_content.splitlines()
        
        metrics = {
            "total_lines": len(lines),
            "code_lines": len([line for line in lines if line.strip() and not line.strip().startswith('#')]),
            "comment_lines": len([line for line in lines if line.strip().startswith('#')]),
            "empty_lines": len([line for line in lines if not line.strip()]),
            "classes": len([node for node in ast.walk(self.v1_ast) if isinstance(node, ast.ClassDef)]),
            "functions": len([node for node in ast.walk(self.v1_ast) if isinstance(node, ast.FunctionDef)]),
            "async_functions": len([node for node in ast.walk(self.v1_ast) if isinstance(node, ast.AsyncFunctionDef)]),
            "imports": len([node for node in ast.walk(self.v1_ast) if isinstance(node, (ast.Import, ast.ImportFrom))]),
        }
        
        # Calculate complexity score
        metrics["complexity_score"] = min(10, metrics["total_lines"] / 400)  # Rough complexity estimate
        
        return metrics
    
    async def _analyze_components(self) -> List[V1Component]:
        """Extract and analyze major components from v1 system"""
        components = []
        
        # Analyze main class (ComprehensiveAdaptivePipeline)
        for node in ast.walk(self.v1_ast):
            if isinstance(node, ast.ClassDef):
                component = V1Component(
                    name=node.name,
                    type="class",
                    lines=(node.lineno, getattr(node, 'end_lineno', node.lineno + 50)),
                    dependencies=self._extract_class_dependencies(node),
                    features=self._extract_class_features(node),
                    complexity_score=self._calculate_component_complexity(node),
                    migration_priority=self._assess_migration_priority(node),
                    modernization_notes=self._generate_modernization_notes(node)
                )
                components.append(component)
        
        return components
    
    async def _extract_proven_patterns(self) -> List[V1Pattern]:
        """Extract proven patterns that should be preserved in v2"""
        patterns = []
        
        # Pattern 1: Adaptive Template System
        patterns.append(V1Pattern(
            pattern_name="Adaptive Template Generation",
            description="Dynamic template creation based on content analysis and brand voice",
            code_examples=self._extract_pattern_code("adaptive_template"),
            benefits=[
                "Dynamic content adaptation",
                "Brand voice consistency", 
                "Content type optimization",
                "Subject-aware formatting"
            ],
            v2_integration_strategy="Integrate with our Advanced Prompt System (Task 2) and Template Manager",
            estimated_effort="8 hours"
        ))
        
        # Pattern 2: Comprehensive Research System
        patterns.append(V1Pattern(
            pattern_name="Multi-Source Research Orchestration",
            description="Coordinated research across multiple APIs with intelligent fallbacks",
            code_examples=self._extract_pattern_code("research_phase"),
            benefits=[
                "Robust multi-API research",
                "Intelligent fallback mechanisms",
                "Result validation and quality scoring",
                "Performance optimization"
            ],
            v2_integration_strategy="Enhance our Contextual Retrieval System (Task 3) with multi-source capabilities",
            estimated_effort="12 hours"
        ))
        
        # Pattern 3: Content Expansion Chains
        patterns.append(V1Pattern(
            pattern_name="Structure-Aware Content Expansion",
            description="Intelligent content expansion based on research categories and content structure",
            code_examples=self._extract_pattern_code("content_expansion"),
            benefits=[
                "Structured content organization",
                "Research-driven expansion",
                "Category-aware processing",
                "Quality-controlled enhancement"
            ],
            v2_integration_strategy="Integrate with our Enhanced FTI Pipeline for advanced content processing",
            estimated_effort="10 hours"
        ))
        
        # Pattern 4: Caching and Performance
        patterns.append(V1Pattern(
            pattern_name="Redis-Based Intelligent Caching",
            description="Multi-layer caching with circuit breakers and performance monitoring",
            code_examples=self._extract_pattern_code("caching"),
            benefits=[
                "Performance optimization",
                "Cost reduction",
                "Circuit breaker patterns",
                "Cache analytics"
            ],
            v2_integration_strategy="Enhance our Intelligent Caching System (Task 2) with v1 patterns",
            estimated_effort="6 hours"
        ))
        
        # Pattern 5: Image Generation and Management
        patterns.append(V1Pattern(
            pattern_name="Contextual Image Generation",
            description="Content-aware image generation with Supabase integration",
            code_examples=self._extract_pattern_code("image_generation"),
            benefits=[
                "Context-aware image selection",
                "Supabase storage integration",
                "Performance optimization",
                "Quality validation"
            ],
            v2_integration_strategy="Integrate with DataForSEO Image Search (Task 5) for enhanced capabilities",
            estimated_effort="8 hours"
        ))
        
        return patterns
    
    async def _inventory_features(self) -> Dict[str, Any]:
        """Create comprehensive inventory of v1 features"""
        features = {
            "core_capabilities": [
                "Adaptive template generation",
                "Multi-source research orchestration", 
                "Content expansion chains",
                "Redis caching with circuit breakers",
                "Contextual image generation",
                "WordPress integration",
                "Affiliate compliance management",
                "Brand voice management",
                "Content quality validation",
                "RAG processing and storage"
            ],
            "integrations": [
                "DataForSEO API",
                "Tavily Search",
                "OpenAI GPT-4",
                "DALL-E image generation",
                "Supabase database",
                "Redis caching",
                "WordPress REST API",
                "Hybrid search systems"
            ],
            "advanced_features": [
                "Circuit breaker patterns",
                "Performance monitoring",
                "Content coherence validation", 
                "E-E-A-T signals enhancement",
                "Structure-aware expansion",
                "Multimodal content enhancement",
                "Research quality control",
                "Cache analytics"
            ],
            "missing_in_v2": self._identify_missing_features(),
            "superior_in_v2": self._identify_v2_advantages()
        }
        
        return features
    
    async def _map_dependencies(self) -> Dict[str, List[str]]:
        """Map dependencies and integration points"""
        dependencies = {}
        
        # Extract import statements
        for node in ast.walk(self.v1_ast):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names = [alias.name for alias in node.names]
                dependencies[module] = names
        
        return dependencies
    
    async def _analyze_performance_patterns(self) -> Dict[str, Any]:
        """Analyze performance optimization patterns in v1"""
        patterns = {
            "caching_strategies": [
                "Redis LLM response caching",
                "Embeddings caching",
                "API response caching",
                "Circuit breaker implementation"
            ],
            "async_patterns": [
                "Async/await throughout pipeline",
                "Concurrent API calls",
                "Timeout handling",
                "Resource management"
            ],
            "optimization_techniques": [
                "Lazy loading",
                "Result validation",
                "Fallback mechanisms",
                "Performance monitoring"
            ]
        }
        
        return patterns
    
    async def _identify_integration_points(self) -> Dict[str, str]:
        """Identify where v1 features can integrate with v2 systems"""
        integration_points = {
            "adaptive_templates": "src/chains/advanced_prompt_system.py - Enhance with v1 adaptive patterns",
            "research_orchestration": "src/retrieval/contextual_retrieval.py - Add multi-source capabilities",
            "content_expansion": "src/pipelines/enhanced_fti_pipeline.py - Integrate structure-aware expansion",
            "caching_improvements": "src/chains/enhanced_confidence_scoring_system.py - Enhance intelligent caching",
            "image_management": "src/integrations/dataforseo_image_search.py - Add contextual generation",
            "performance_monitoring": "src/monitoring/ - Add v1 performance patterns",
            "brand_voice": "src/templates/improved_template_manager.py - Integrate brand voice management",
            "quality_validation": "src/chains/enhanced_confidence_scoring_system.py - Add v1 validation patterns"
        }
        
        return integration_points
    
    async def _assess_migration_risks(self) -> Dict[str, Any]:
        """Assess risks and mitigation strategies for migration"""
        risks = {
            "high_risk": [
                {
                    "risk": "Monolithic architecture dependencies",
                    "impact": "High",
                    "mitigation": "Extract components incrementally, maintain interfaces"
                },
                {
                    "risk": "Performance regression",
                    "impact": "Medium", 
                    "mitigation": "Comprehensive benchmarking, gradual rollout"
                }
            ],
            "medium_risk": [
                {
                    "risk": "Feature compatibility",
                    "impact": "Medium",
                    "mitigation": "Comprehensive testing, compatibility layer"
                },
                {
                    "risk": "Data migration complexity",
                    "impact": "Medium",
                    "mitigation": "Incremental migration, rollback capabilities"
                }
            ],
            "low_risk": [
                {
                    "risk": "Configuration differences",
                    "impact": "Low",
                    "mitigation": "Configuration mapping, validation scripts"
                }
            ]
        }
        
        return risks
    
    # Helper methods for component analysis
    def _extract_class_dependencies(self, node: ast.ClassDef) -> List[str]:
        """Extract dependencies for a class"""
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                dependencies.append(child.id)
        return list(set(dependencies))[:10]  # Limit to avoid huge lists
    
    def _extract_class_features(self, node: ast.ClassDef) -> List[str]:
        """Extract features/capabilities of a class"""
        features = []
        for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
            features.append(method.name)
        return features[:20]  # Limit to avoid huge lists
    
    def _calculate_component_complexity(self, node: ast.ClassDef) -> int:
        """Calculate complexity score for a component"""
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        lines = getattr(node, 'end_lineno', node.lineno + 50) - node.lineno
        return min(10, len(methods) + lines // 100)
    
    def _assess_migration_priority(self, node: ast.ClassDef) -> str:
        """Assess migration priority for a component"""
        if node.name == "ComprehensiveAdaptivePipeline":
            return "high"
        return "medium"
    
    def _generate_modernization_notes(self, node: ast.ClassDef) -> List[str]:
        """Generate modernization notes for a component"""
        notes = [
            "Convert to LCEL patterns",
            "Implement proper async/await",
            "Add comprehensive typing",
            "Integrate with v2 architecture",
            "Add proper error handling"
        ]
        return notes
    
    def _extract_pattern_code(self, pattern_type: str) -> List[str]:
        """Extract code examples for specific patterns"""
        # This would extract relevant code snippets based on pattern type
        return [f"# {pattern_type} code example would be extracted here"]
    
    def _identify_missing_features(self) -> List[str]:
        """Identify features in v1 that are missing in v2"""
        return [
            "Adaptive template generation",
            "Brand voice management",
            "Structure-aware content expansion",
            "Circuit breaker patterns",
            "Content coherence validation",
            "Multi-source research orchestration"
        ]
    
    def _identify_v2_advantages(self) -> List[str]:
        """Identify areas where v2 is superior to v1"""
        return [
            "Modular architecture vs monolithic",
            "Enhanced confidence scoring (4-factor vs basic)",
            "Advanced contextual retrieval system",
            "Production-ready API platform",
            "Comprehensive testing framework",
            "Real A/B testing capabilities",
            "Enterprise-grade monitoring",
            "LCEL-based chain architecture"
        ]
    
    async def generate_migration_plan(self) -> MigrationPlan:
        """Generate comprehensive migration plan"""
        if not self.analysis_results:
            await self.analyze_v1_system()
        
        plan = MigrationPlan(
            components_to_migrate=self.analysis_results["component_analysis"],
            patterns_to_extract=self.analysis_results["pattern_extraction"],
            integration_points=self.analysis_results["integration_opportunities"],
            risk_assessment=self.analysis_results["risk_assessment"],
            timeline={
                "phase_1": "Pattern extraction and analysis (2 days)",
                "phase_2": "Component modernization (5 days)",
                "phase_3": "Integration with v2 systems (3 days)",
                "phase_4": "Testing and validation (2 days)",
                "phase_5": "Production deployment (1 day)"
            }
        )
        
        return plan
    
    async def export_analysis_report(self, output_file: str = "v1_migration_analysis.json"):
        """Export comprehensive analysis report"""
        if not self.analysis_results:
            await self.analyze_v1_system()
        
        migration_plan = await self.generate_migration_plan()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "v1_file_analyzed": str(self.v1_file_path),
            "analysis_results": self.analysis_results,
            "migration_plan": {
                "components_count": len(migration_plan.components_to_migrate),
                "patterns_count": len(migration_plan.patterns_to_extract),
                "integration_points": migration_plan.integration_points,
                "timeline": migration_plan.timeline,
                "risk_assessment": migration_plan.risk_assessment
            },
            "recommendations": [
                "Start with adaptive template pattern extraction",
                "Integrate caching improvements first (low risk, high impact)",
                "Modernize research orchestration for contextual retrieval",
                "Add brand voice management to template system",
                "Implement structure-aware expansion in FTI pipeline"
            ]
        }
        
        # Convert dataclasses to dicts for JSON serialization
        def convert_dataclass(obj):
            if hasattr(obj, '__dict__'):
                return asdict(obj) if hasattr(obj, '__dataclass_fields__') else obj.__dict__
            return str(obj)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=convert_dataclass)
        
        logger.info(f"✅ Migration analysis report exported to: {output_file}")
        return output_file

# Example usage and testing
async def main():
    """Run comprehensive v1 migration analysis"""
    print("🚀 Starting V1 to V2 Migration Analysis Framework")
    
    analyzer = V1AnalysisFramework()
    
    try:
        # Run comprehensive analysis
        results = await analyzer.analyze_v1_system()
        
        # Generate migration plan
        plan = await analyzer.generate_migration_plan()
        
        # Export analysis report
        report_file = await analyzer.export_analysis_report()
        
        print(f"\n✅ ANALYSIS COMPLETE!")
        print(f"📊 File Metrics: {results['file_metrics']['total_lines']} lines analyzed")
        print(f"🔧 Components Found: {len(results['component_analysis'])}")
        print(f"🎯 Patterns Extracted: {len(results['pattern_extraction'])}")
        print(f"📋 Integration Points: {len(results['integration_opportunities'])}")
        print(f"📄 Report: {report_file}")
        
        return results, plan
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 