import operator
from typing import Annotated, List, Optional, TypedDict, Union
from langchain_core.messages import AnyMessage
from pydantic import BaseModel, Field

# =============================================================================
# INPUT / OUTPUT STATES
# =============================================================================

class AgentInputState(TypedDict):
    """Input state for the agent."""
    messages: List[AnyMessage]

class AgentState(TypedDict):
    """Overall state of the deep research agent."""
    messages: Annotated[List[AnyMessage], operator.add]
    research_brief: str
    notes: Annotated[List[str], operator.add]  # Accumulated notes from researchers

# =============================================================================
# SUPERVISOR STATE
# =============================================================================

class SupervisorState(TypedDict):
    """State for the Research Supervisor."""
    supervisor_messages: Annotated[List[AnyMessage], operator.add]
    research_brief: str
    research_iterations: int
    notes: Annotated[List[str], operator.add]  # Passed back to main state

# =============================================================================
# RESEARCHER STATE
# =============================================================================

class ResearcherState(TypedDict):
    """State for an individual Researcher."""
    researcher_messages: Annotated[List[AnyMessage], operator.add]
    research_topic: str
    tool_call_iterations: int

class ResearcherOutputState(TypedDict):
    """Output from a researcher back to supervisor."""
    compressed_research: str
    raw_notes: List[str]

# =============================================================================
# STRUCTURED OUTPUT MODELS (Pydantic)
# =============================================================================

class ClarifyWithUser(BaseModel):
    """Decision model for whether to ask the user for clarification."""
    need_clarification: bool = Field(description="Whether clarification is needed")
    question: str = Field(description="Clarifying question to ask, if needed")
    verification: str = Field(description="Verification message if no clarification needed")

class ResearchQuestion(BaseModel):
    """Structured research brief generation."""
    research_brief: str = Field(description="Detailed research brief/question")

class ConductResearch(BaseModel):
    """Tool to delegate a research task."""
    research_topic: str = Field(description="Specific research topic to investigate")

class ResearchComplete(BaseModel):
    """Tool to signal research completion."""
    reason: str = Field(description="Reason for completing research")

class Summary(BaseModel):
    """Structured summary of a webpage."""
    summary: str = Field(description="Concise summary of content")
    key_excerpts: str = Field(description="Key quotes or data points")
