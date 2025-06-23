# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample demonstrating ArrayIteratorAgent usage patterns with realistic data flow."""

from google.adk.agents import LlmAgent, SequentialAgent
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Import ArrayIteratorAgent - try package first, fallback to local
try:
    from google.adk.agents.array_iterator_agent import ArrayIteratorAgent
except ImportError:
    # If package import fails, try local import
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))
    from google.adk.agents.array_iterator_agent import ArrayIteratorAgent


# === Pydantic Models for Structured Output ===

class DocumentMetadata(BaseModel):
    """Structured output for document discovery."""
    title: str = Field(..., description="Document title")
    content: str = Field(..., description="Document content") 
    url: str = Field(..., description="Document URL")
    importance: int = Field(..., description="Importance score 1-10")

class DocumentAnalysis(BaseModel):
    """Structured output for document analysis."""
    title: str = Field(..., description="Extracted title")
    summary: str = Field(..., description="2-sentence summary")
    key_topics: List[str] = Field(..., description="List of key topics")
    sentiment: str = Field(..., description="Overall sentiment")

class CustomerData(BaseModel):
    """Structured output for customer data."""
    name: str = Field(..., description="Customer name")
    email: str = Field(..., description="Customer email")
    annual_spend: float = Field(..., description="Annual spending amount")
    tier: str = Field(..., description="Customer tier")


# === Agent Workflows with Realistic Data Flow ===

def create_document_discovery_and_processing_workflow():
    """
    REALISTIC WORKFLOW: Document discovery → Processing
    
    1. Document Finder Agent discovers documents (produces array)
    2. ArrayIteratorAgent processes each document
    """
    
    # Step 1: Agent that discovers/fetches documents and produces structured array
    document_finder = LlmAgent(
        name="document_finder",
        model="gemini-2.0-flash",
        instruction="""
        Based on the user's query in {user_query}, find and list relevant documents.
        
        For each document, provide:
        - title: Document title
        - content: Brief content excerpt  
        - url: Document URL
        - importance: Relevance score 1-10
        
        Return as a JSON array of document objects.
        """,
        output_schema=List[DocumentMetadata],  # Produces structured array
        output_key="discovered_documents"  # Stored in session state
    )
    
    # Step 2: Document analyzer (processes individual documents)
    document_analyzer = LlmAgent(
        name="document_analyzer", 
        model="gemini-2.0-flash",
        instruction="""
        Analyze the document provided in {current_document}.
        
        Extract:
        - title: Clean document title
        - summary: Exactly 2 sentences summarizing the content
        - key_topics: List of 3-5 main topics/keywords
        - sentiment: positive/negative/neutral
        """,
        output_schema=DocumentAnalysis,  # Structured output per document
        output_key="document_analysis"
    )
    
    # Step 3: Array iterator processes the discovered documents
    document_processor = ArrayIteratorAgent(
        name="document_processor",
        array_key="discovered_documents",  # Array from document_finder
        item_key="current_document",       # Current doc for analyzer
        output_key="document_analyses",    # Collected analyses
        sub_agents=[document_analyzer]
    )
    
    # Step 4: Complete workflow
    workflow = SequentialAgent(
        name="document_workflow",
        description="Discovers documents then processes each one",
        sub_agents=[
            document_finder,      # Produces array in session state
            document_processor    # Processes the array
        ]
    )
    
    return workflow


def create_customer_segmentation_workflow():
    """
    REALISTIC WORKFLOW: Customer data ingestion → Segmentation
    
    1. CRM Data Agent fetches customer data (produces nested array)
    2. ArrayIteratorAgent processes each customer for segmentation
    """
    
    # Step 1: Agent that fetches customer data from CRM/database
    crm_data_agent = LlmAgent(
        name="crm_data_agent",
        model="gemini-2.0-flash",
        instruction="""
        Based on the company ID in {company_id}, fetch customer data from CRM.
        
        Return company info with customer array:
        {
          "company": {
            "name": "Company Name",
            "industry": "Industry Type", 
            "customers": [
              {"name": "Customer Name", "email": "email", "annual_spend": 0000}
            ]
          }
        }
        
        Include 3-5 customers with realistic data.
        """,
        output_key="company_data"  # Creates nested structure
    )
    
    # Step 2: Customer segmentation processor
    customer_segmenter = LlmAgent(
        name="customer_segmenter",
        model="gemini-2.0-flash",
        instruction="""
        Process the customer data in {current_customer}.
        
        Analyze and determine:
        - Tier level based on annual_spend:
          * VIP: > $10,000 annual spend  
          * Premium: $5,000-$10,000 annual spend
          * Standard: < $5,000 annual spend
        - Personalized greeting
        - Recommended actions
        
        Return structured customer analysis.
        """,
        output_schema=CustomerData,  # Structured output per customer
        output_key="customer_segment"
    )
    
    # Step 3: Array iterator processes customers from nested path
    customer_processor = ArrayIteratorAgent(
        name="customer_processor",
        array_key="company_data.company.customers",  # Nested array access
        item_key="current_customer",
        output_key="company_data.company.segmented_customers",  # Nested output
        sub_agents=[customer_segmenter]
    )
    
    # Step 4: Complete workflow
    segmentation_workflow = SequentialAgent(
        name="customer_segmentation_workflow",
        description="Fetches customer data then segments each customer",
        sub_agents=[
            crm_data_agent,       # Produces nested data structure
            customer_processor    # Processes nested customer array
        ]
    )
    
    return segmentation_workflow


def create_data_ingestion_and_processing_workflow():
    """
    REALISTIC WORKFLOW: Data ingestion → ETL processing
    
    1. Data Collector Agent fetches raw data (produces array)
    2. ArrayIteratorAgent processes each record through ETL pipeline
    """
    
    # Step 1: Data collector that fetches raw events/records
    data_collector = LlmAgent(
        name="data_collector",
        model="gemini-2.0-flash", 
        instruction="""
        Based on the data source in {data_source}, collect raw data records.
        
        Return array of raw data records:
        [
          {"source": "API", "data": "raw_event_data", "timestamp": "2024-01-01"},
          {"source": "DB", "data": "database_record", "timestamp": "2024-01-02"}
        ]
        
        Include 5-10 records with various sources and realistic data.
        """,
        output_key="raw_data"  # Produces array for processing
    )
    
    # ETL Pipeline Steps
    
    # Step 2a: Extract data
    extractor = LlmAgent(
        name="data_extractor",
        model="gemini-2.0-flash",
        instruction="""
        Extract structured data from {current_item}.
        Parse the raw data and extract key fields like ID, type, value, etc.
        """,
        output_key="extracted_data"
    )
    
    # Step 2b: Validate data
    validator = LlmAgent(
        name="data_validator",
        model="gemini-2.0-flash",
        instruction="""
        Validate the extracted data in {extracted_data}.
        Check for completeness, format, and business rules.
        Return validation status and cleaned data.
        """,
        output_key="validation_result"
    )
    
    # Step 2c: Transform data
    transformer = LlmAgent(
        name="data_transformer",
        model="gemini-2.0-flash",
        instruction="""
        Transform validated data {validation_result} into final format.
        Apply business logic, standardize formats, enrich with metadata.
        """,
        output_key="transformed_data"
    )
    
    # Step 3: Sequential ETL pipeline
    etl_pipeline = SequentialAgent(
        name="etl_pipeline",
        description="Extract → Transform → Load pipeline for single record",
        sub_agents=[extractor, validator, transformer]
    )
    
    # Step 4: Array iterator applies ETL pipeline to each raw record
    batch_processor = ArrayIteratorAgent(
        name="batch_processor",
        array_key="raw_data",           # Array from data_collector
        item_key="current_item",        # Current record for ETL
        output_key="processed_batch",   # Final processed results
        sub_agents=[etl_pipeline]       # ETL pipeline as single sub-agent
    )
    
    # Step 5: Complete data processing workflow
    data_workflow = SequentialAgent(
        name="data_processing_workflow",
        description="Collects raw data then processes each record through ETL",
        sub_agents=[
            data_collector,     # Produces raw data array
            batch_processor     # Processes each record through ETL
        ]
    )
    
    return data_workflow


def create_quality_assurance_workflow():
    """
    REALISTIC WORKFLOW: Content generation → Quality check
    
    1. Content Generator creates articles (produces array)
    2. ArrayIteratorAgent runs QA checks on each article
    3. Handles escalation when quality issues are found
    """
    
    # Step 1: Content generator that creates articles
    content_generator = LlmAgent(
        name="content_generator",
        model="gemini-2.0-flash",
        instruction="""
        Based on the topic list in {topics}, generate article drafts.
        
        Return array of article objects:
        [
          {"title": "Article Title", "content": "Article content...", "status": "draft"},
          {"title": "Another Title", "content": "More content...", "status": "draft"}
        ]
        
        Include 4-5 articles. Occasionally include problematic content with 
        "PLAGIARISM" or "LOW_QUALITY" markers to test QA escalation.
        """,
        output_key="generated_articles"  # Produces article array
    )
    
    # Step 2: Quality assurance checker
    qa_checker = LlmAgent(
        name="qa_checker",
        model="gemini-2.0-flash",
        instruction="""
        Review the article in {current_article} for quality issues.
        
        Check for:
        - Plagiarism indicators (if content contains "PLAGIARISM")
        - Quality issues (if content contains "LOW_QUALITY") 
        - Grammar and coherence
        
        If serious issues found (PLAGIARISM/LOW_QUALITY), escalate to stop processing.
        Otherwise, return quality score and recommendations.
        """,
        output_key="qa_result"
    )
    
    # Step 3: Array iterator with escalation handling
    quality_processor = ArrayIteratorAgent(
        name="quality_processor", 
        array_key="generated_articles",     # Array from content_generator
        item_key="current_article",         # Current article for QA
        output_key="qa_results",            # QA results (until escalation)
        sub_agents=[qa_checker]
    )
    
    # Step 4: Complete QA workflow
    qa_workflow = SequentialAgent(
        name="content_qa_workflow",
        description="Generates content then runs QA checks with escalation",
        sub_agents=[
            content_generator,    # Produces article array
            quality_processor     # QA checks each article (stops on issues)
        ]
    )
    
    return qa_workflow


# === Session State Examples ===
# In real scenarios, this data comes from previous agents, not hardcoded constants!

def create_standalone_iterator_example():
    """
    EXAMPLE: Using ArrayIteratorAgent with pre-populated session state
    (For testing when you already have array data)
    """
    
    # Simple processor for when data is already in session state
    simple_processor = LlmAgent(
        name="simple_processor",
        model="gemini-2.0-flash",
        instruction="Process the item in {current_item} and return a summary",
        output_key="item_summary"
    )
    
    # Array iterator (assumes data already in session state)
    standalone_iterator = ArrayIteratorAgent(
        name="standalone_iterator",
        array_key="existing_data",       # Must exist in session state
        item_key="current_item",
        output_key="processing_results",
        sub_agents=[simple_processor]
    )
    
    return standalone_iterator


# === How Session State Gets Populated ===

EXAMPLE_SESSION_STATES = {
    "document_workflow": {
        # Initial state - user provides query
        "user_query": "Find articles about AI in healthcare",
        # After document_finder runs:
        "discovered_documents": [
            {"title": "AI in Medical Diagnosis", "content": "AI systems are...", "url": "https://...", "importance": 9},
            {"title": "ML for Drug Discovery", "content": "Machine learning...", "url": "https://...", "importance": 8}
        ],
        # After ArrayIteratorAgent runs:
        "document_analyses": [
            {"title": "AI in Medical Diagnosis", "summary": "...", "key_topics": ["AI", "medical"], "sentiment": "positive"},
            {"title": "ML for Drug Discovery", "summary": "...", "key_topics": ["ML", "pharma"], "sentiment": "positive"}
        ]
    },
    
    "customer_workflow": {
        # Initial state
        "company_id": "CORP-123",
        # After crm_data_agent runs:
        "company_data": {
            "company": {
                "name": "TechCorp Inc",
                "industry": "Software",
                "customers": [
                    {"name": "Alice Johnson", "email": "alice@example.com", "annual_spend": 15000},
                    {"name": "Bob Smith", "email": "bob@example.com", "annual_spend": 7500}
                ],
                # After ArrayIteratorAgent runs:
                "segmented_customers": [
                    {"name": "Alice Johnson", "email": "alice@example.com", "annual_spend": 15000, "tier": "VIP"},
                    {"name": "Bob Smith", "email": "bob@example.com", "annual_spend": 7500, "tier": "Premium"}
                ]
            }
        }
    },
    
    "data_processing_workflow": {
        # Initial state
        "data_source": "production_logs",
        # After data_collector runs:
        "raw_data": [
            {"source": "API", "data": "user_login_event", "timestamp": "2024-01-01T10:00:00Z"},
            {"source": "DB", "data": "order_created", "timestamp": "2024-01-01T10:05:00Z"}
        ],
        # After ArrayIteratorAgent runs:
        "processed_batch": [
            {"event_type": "login", "user_id": "123", "processed_at": "2024-01-01T10:00:00Z", "status": "valid"},
            {"event_type": "order", "order_id": "456", "processed_at": "2024-01-01T10:05:00Z", "status": "valid"}
        ]
    }
}


if __name__ == "__main__":
    print("🔄 ArrayIteratorAgent: Realistic Workflow Examples")
    print("=" * 60)
    
    print("\n📋 Available Workflows:")
    print("1. Document Discovery & Processing:", create_document_discovery_and_processing_workflow().name)
    print("2. Customer Segmentation:", create_customer_segmentation_workflow().name) 
    print("3. Data Ingestion & ETL:", create_data_ingestion_and_processing_workflow().name)
    print("4. Content QA Pipeline:", create_quality_assurance_workflow().name)
    print("5. Standalone Iterator:", create_standalone_iterator_example().name)
    
    print("\n🔄 How ArrayIteratorAgent Works:")
    print("┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐")
    print("│   Agent A       │───▶│  Session State   │───▶│ ArrayIterator   │")
    print("│ (Produces Array)│    │ {array_key: [...]}│    │ (Processes Each)│")
    print("└─────────────────┘    └──────────────────┘    └─────────────────┘")
    
    print("\n📊 Session State Evolution Example:")
    print("Initial:  {user_query: 'Find AI articles'}")
    print("After Agent A: {user_query: '...', discovered_docs: [doc1, doc2, doc3]}")
    print("After Iterator: {user_query: '...', discovered_docs: [...], analyses: [analysis1, analysis2, analysis3]}")
    
    print("\n🚀 Key Benefits:")
    print("✅ Structured data flow between agents")
    print("✅ Automatic result collection") 
    print("✅ Nested key support for complex data")
    print("✅ Escalation handling for quality control")
    print("✅ Reusable iteration patterns")
    
    print("\n💡 Usage: Combine agents in SequentialAgent for complete workflows")
    print("   Example: DataCollector → ArrayIteratorAgent → ResultProcessor") 