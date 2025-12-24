from dataclasses import dataclass, field, fields
from typing import Any, Optional, Literal, List
from langchain_core.runnables import RunnableConfig

RESET = "RESET"

@dataclass
class Configuration:
    """The configuration for the agent."""
    
    allow_clarification: bool = True
    max_structured_output_retries: int = 3
    
    # Model Configs
    research_model: str = "ollama:qwen2.5:7b"  # Best available for reasoning
    research_model_max_tokens: int = 32000
    
    compression_model: str = "ollama:qwen2.5:3b" # Fast and capable for summarizing
    compression_model_max_tokens: int = 32000
    max_content_length: int = 4000  # For summarization
    
    final_report_model: str = "ollama:qwen2.5:7b" # Strong writer
    final_report_model_max_tokens: int = 32000
    
    summarization_model: str = "ollama:qwen2.5:3b"
    summarization_model_max_tokens: int = 32000
    
    # Research Limits
    max_concurrent_research_units: int = 2
    max_researcher_iterations: int = 5
    max_react_tool_calls: int = 6
    
    # External Tools
    search_api: str = "tavily"  # tavily, firecrawl, etc.
    mcp_prompt: str = ""
    
    # MCP Configuration
    mcp_config: Optional[Any] = None

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        if config is None:
            return cls()
            
        configurable = config.get("configurable", {})
        
        # Extract fields from configurable dict that match Configuration dataclass
        # Use simple import inside or top level. We imported fields at top level now.
        valid_fields = {f.name for f in fields(cls)}
        
        values = {}
        for key, value in configurable.items():
            if key in valid_fields:
                values[key] = value
                
        return cls(**values)

class SearchAPI:
    """Enum for Search APIs"""
    TAVILY = "tavily"
    FIRECRAWL = "firecrawl"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    NONE = "none"

    def __init__(self, value):
        self.value = value
