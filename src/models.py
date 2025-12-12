from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class ResearchTask(BaseModel):
    """Araştırma görevi veri modeli"""
    query: str
    priority: int = 1
    depth: int = 3  # 1-5 arası derinlik
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Source(BaseModel):
    """Bulunan kaynak veri modeli"""
    url: str
    title: str
    content: str
    source_type: str  # web, reddit, github, etc.
    reliability_score: float = 0.0
    publish_date: Optional[str] = None
    key_points: List[str] = Field(default_factory=list)

class ResearchResult(BaseModel):
    """Nihai araştırma sonucu"""
    task: ResearchTask
    report: str
    sources: List[Source]
    metadata: Dict[str, Any] = Field(default_factory=dict)
