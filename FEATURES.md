# Universal RAG Chain: Core Features

This document outlines the key features and capabilities of the Universal RAG Chain.

## 🚀 Key Integrated Systems

The chain integrates a comprehensive suite of advanced systems to deliver high-quality, structured content.

-   **Contextual Retrieval System**: Employs a sophisticated hybrid search (vector + keyword), multi-query generation, and Maximal Marginal Relevance (MMR) to find the most relevant information.
-   **Comprehensive Web Research**: Actively researches authoritative sources like AskGamblers and Casino.Guru, augmented by real-time Tavily web searches, to build a deep understanding of the target casino.
-   **Screenshot Evidence Engine**: Captures, stores, and associates visual evidence from web research directly with the generated content, ensuring claims can be verified.
-   **95-Field Intelligence Schema**: Extracts and structures data into a detailed 95-point Pydantic model covering trustworthiness, games, bonuses, payments, and more.
-   **Authoritative Hyperlink Engine**: Enriches content by automatically inserting relevant, high-authority outbound links based on context.
-   **WordPress Publishing**: Fully automates publishing content to WordPress, including support for custom post types (e.g., MT Casino), metadata, featured images, and embedded media.
-   **Advanced Confidence Scoring**: Provides a multi-factor confidence score for each response, assessing source quality, data consistency, and query relevance.
-   **Security & Compliance**: Includes built-in checks for enterprise-grade security and compliance.

## 🎰 Affiliate Program Intelligence (New)

A new `AffiliateProgramIntelligenceCategory` has been added to the core `CasinoIntelligence` schema.

-   **15 New Data Points**: Captures critical affiliate T&C details, including:
    -   **Commission Models**: Revenue Share, CPA, Hybrid models.
    -   **Marketing Rules**: Policies on keywords, social media, and email.
    -   **Payment Terms**: Minimum thresholds and payment schedules.
-   **Automated Extraction**: The system is designed to automatically populate these fields during the web research phase.
-   **Purpose**: This feature enables the generation of content that is not only informative for players but also compliant with affiliate partnership requirements. 