# Deep Research & RAG Assistant

An advanced multi-agent research platform designed for deep web analysis and intelligent document interaction. Built with LangGraph, Next.js, and Ollama, this system provides a unified interface for both conducting comprehensive internet research and querying internal knowledge bases.

![Architecture Diagram](https://github.com/HamdiOzkurt/Research-Rag-UI/blob/main/multi_agent_search/assets/architecture.png?raw=true)

## System Capabilities

The platform operates using two primary architectural pipelines.

| Feature | Deep Research Agent | RAG (Document Chat) |
|---------|---------------------|---------------------|
| **Primary Function** | Autonomous deep web research and report generation | Context-aware Q&A based on uploaded documents |
| **Model Architecture** | Supervisor-Worker hierarchical agent swarm | Hybrid Search (Vector + Keyword) with Re-ranking |
| **Data Source** | Real-time Web (via Tavily/Firecrawl) | PDF, DOCX, TXT files (via ChromaDB) |
| **Output** | Comprehensive, citation-backed markdown reports | Precise answers with direct references to document chunks |
| **Key Capability** | Recursive query planning and self-correction | Evidence-based reasoning and hallucination prevention |

## Technology Stack

| Component | Technology | Description |
|-----------|------------|-------------|
| **Orchestration** | LangGraph | State management and agentic workflow coordination |
| **Frontend** | Next.js 14 | Responsive, modern React framework with Server Actions |
| **Backend** | FastAPI | High-performance Python API for agent communication |
| **LLM Inference** | Ollama / Groq | Hybrid inference engine supporting local (Llama/Qwen) and cloud models |
| **Vector DB** | ChromaDB | Local embedding storage for efficient document retrieval |
| **Search Tools** | Tavily / Firecrawl | Optimized search APIs for LLM consumption |

## Getting Started

Follow these instructions to set up the environment and launch the application.

### 1. Backend Setup (Python)

Navigate to the project root and activate the environment.

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Start the FastAPI server
python -m uvicorn src.simple_copilot_backend:app --reload --port 8000
```

### 2. Frontend Setup (Next.js)

Open a new terminal window and navigate to the UI directory.

```powershell
cd copilotkit-ui

# Install Node modules
npm install

# Start the development server
npm run dev
```

Access the application at [http://localhost:3000](http://localhost:3000).

## Configuration

The system requires environmental variables to be set in the `.env` file.

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_API_KEYS` | List of Gemini API keys for rotation (prevents 429 errors) | `key1,key2,key3` |
| `FIRECRAWL_API_KEY` | Required for the web search capability | `fc-your-key` |
| `DEFAULT_MODEL` | Primary model for general tasks | `ollama:qwen2.5:7b` |
| `TAVILY_API_KEY` | Alternative search provider key | `tvly-your-key` |

## Project Structure

```
multi_agent_search/
├── src/
│   ├── simple_copilot_backend.py      # Main API Entry point
│   ├── agents/
│   │   ├── deep_research/             # Deep Research Agent Logic
│   │   │   ├── deep_researcher.py     # Supervisor Agent Implementation
│   │   │   └── graph.py               # Workflow Graph Definition
│   │   ├── rag_agent.py               # RAG Agent Implementation
│   │   └── agentic_chunker.py         # Advanced Document Processor
│   └── config/
│       └── settings.py                # Global Configuration
├── copilotkit-ui/                     # Next.js Frontend Source
│   ├── app/
│   │   ├── components/                # React Components
│   │   └── page.tsx                   # Main Entry Page
├── assets/                            # Static Assets and Diagrams
└── requirements.txt                   # Python Dependencies
```

## License

This project is open-source and available under the MIT License.
