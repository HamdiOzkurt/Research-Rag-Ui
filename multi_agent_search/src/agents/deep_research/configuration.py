"""Configuration management for the Deep Research system - Adapted for Ollama."""

import os
from enum import Enum
from typing import Any, List, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    """Enumeration of available search API providers."""
    
    TAVILY = "tavily"
    NONE = "none"

class MCPConfig(BaseModel):
    """Configuration for Model Context Protocol (MCP) servers."""
    
    url: Optional[str] = Field(default=None)
    tools: Optional[List[str]] = Field(default=None)
    auth_required: Optional[bool] = Field(default=False)

class Configuration(BaseModel):
    """Main configuration class for the Deep Research agent."""
    
    # General Configuration
    max_structured_output_retries: int = Field(default=3)
    allow_clarification: bool = Field(default=False)  # Disabled for faster testing
    max_concurrent_research_units: int = Field(default=2)  # Limited for Ollama
    
    # Research Configuration
    search_api: SearchAPI = Field(default=SearchAPI.TAVILY)
    max_researcher_iterations: int = Field(default=2)  # Reduced to prevent loops
    max_react_tool_calls: int = Field(default=6)
    
    # Model Configuration - OLLAMA ONLY (No API limits, just slower)
    summarization_model: str = Field(default="ollama:qwen2.5:3b")
    summarization_model_max_tokens: int = Field(default=4096)
    max_content_length: int = Field(default=4000)
    
    compression_model: str = Field(default="ollama:qwen2.5:3b")
    compression_model_max_tokens: int = Field(default=4096)
    
    research_model: str = Field(default="ollama:qwen2.5:3b")
    research_model_max_tokens: int = Field(default=4096)
    
    final_report_model: str = Field(default="ollama:qwen2.5:7b")  # Bigger model for final report
    final_report_model_max_tokens: int = Field(default=8192)
    
    # MCP server configuration
    mcp_config: Optional[MCPConfig] = Field(default=None)
    mcp_prompt: Optional[str] = Field(default=None)

    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig] = None) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
