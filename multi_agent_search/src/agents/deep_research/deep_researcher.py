"""Main LangGraph implementation for the Deep Research agent - FIXED VERSION"""

import asyncio
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from .configuration import Configuration
from .prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,
)
from .state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from .utils import (
    get_all_tools,
    get_api_key_for_model,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
)

# Initialize a configurable model that we will use throughout the agent
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key", "model_provider"),
)

async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """Analyze user messages and ask clarifying questions if the research scope is unclear."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")
    
    messages = state["messages"]
    model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    clarification_model = (
        configurable_model
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(model_config)
    )
    
    prompt_content = clarify_with_user_instructions.format(
        messages=get_buffer_string(messages), 
        date=get_today_str()
    )
    response = await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    
    if response.need_clarification:
        return Command(
            goto=END, 
            update={"messages": [AIMessage(content=response.question)]}
        )
    else:
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]}
        )


async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief."""
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    research_model = (
        configurable_model
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    prompt_content = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    )
    response = await research_model.ainvoke([HumanMessage(content=prompt_content)])
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "notes": [],  # Reset notes
            "supervisor_messages": {
                "type": "override",
                "value": []  # Will be initialized by supervisor
            }
        }
    )


async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Lead research supervisor that plans research strategy and delegates to researchers."""
    configurable = Configuration.from_runnable_config(config)
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    # Check if supervisor_messages needs initialization
    supervisor_messages = state.get("supervisor_messages", [])
    if not supervisor_messages:
        supervisor_system_prompt = lead_researcher_prompt.format(
            date=get_today_str(),
            max_concurrent_research_units=configurable.max_concurrent_research_units,
            max_researcher_iterations=configurable.max_researcher_iterations
        )
        supervisor_messages = [
            SystemMessage(content=supervisor_system_prompt),
            HumanMessage(content=state.get("research_brief", "Conduct research"))
        ]

    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    
    research_model = (
        configurable_model
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    response = await research_model.ainvoke(supervisor_messages)
    
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Execute tools called by the supervisor."""
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]
    
    exceeded_allowed_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete_tool_call = any(
        tool_call["name"] == "ResearchComplete" 
        for tool_call in most_recent_message.tool_calls
    )
    
    if exceeded_allowed_iterations or no_tool_calls or research_complete_tool_call:
        all_notes = get_notes_from_tool_calls(supervisor_messages)
        return Command(
            goto=END,
            update={
                "notes": all_notes,
                "raw_notes": state.get("raw_notes", []),
                "research_brief": state.get("research_brief", "")
            }
        )
    
    all_tool_messages = []
    new_raw_notes = []
    
    # Handle think_tool
    think_tool_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "think_tool"]
    for tool_call in think_tool_calls:
        reflection_content = tool_call["args"]["reflection"]
        all_tool_messages.append(ToolMessage(
            content=f"Reflection recorded: {reflection_content}",
            name="think_tool",
            tool_call_id=tool_call["id"]
        ))
    
    # Handle ConductResearch
    conduct_research_calls = [tc for tc in most_recent_message.tool_calls if tc["name"] == "ConductResearch"]
    
    if conduct_research_calls:
        allowed_calls = conduct_research_calls[:configurable.max_concurrent_research_units]
        
        # Parallel execution of researchers
        research_tasks = [
            researcher_subgraph.ainvoke({
                "researcher_messages": [
                    HumanMessage(content=tool_call["args"]["research_topic"])
                ],
                "research_topic": tool_call["args"]["research_topic"]
            }, config) 
            for tool_call in allowed_calls
        ]
        
        tool_results = await asyncio.gather(*research_tasks)
        
        for observation, tool_call in zip(tool_results, allowed_calls):
            content = observation.get("compressed_research", "Error synthesizing")
            all_tool_messages.append(ToolMessage(
                content=content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            ))
            new_raw_notes.extend(observation.get("raw_notes", []))

    # Update state
    update_payload = {"supervisor_messages": all_tool_messages}
    if new_raw_notes:
        update_payload["raw_notes"] = new_raw_notes
        
    return Command(
        goto="supervisor",
        update=update_payload
    ) 

# Supervisor Graph
supervisor_builder = StateGraph(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()

async def researcher(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher_tools"]]:
    """Individual researcher logic."""
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    
    tools = await get_all_tools(config)
    
    research_model_config = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    researcher_prompt = research_system_prompt.format(
        mcp_prompt=configurable.mcp_prompt or "", 
        date=get_today_str()
    )
    
    research_model = (
        configurable_model
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
        .with_config(research_model_config)
    )
    
    # Prepend system prompt if not present
    if not researcher_messages or not isinstance(researcher_messages[0], SystemMessage):
        messages_to_send = [SystemMessage(content=researcher_prompt)] + researcher_messages
    else:
        messages_to_send = researcher_messages

    response = await research_model.ainvoke(messages_to_send)
    
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1
        }
    )

async def execute_tool_safely(tool, args, config):
    try:
        return await tool.ainvoke(args, config)
    except Exception as e:
        return f"Error executing tool: {str(e)}"

async def researcher_tools(state: ResearcherState, config: RunnableConfig) -> Command[Literal["researcher", "compress_research"]]:
    """Execute researcher tools."""
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = researcher_messages[-1]
    
    if not most_recent_message.tool_calls:
        return Command(goto="compress_research")
    
    tools = await get_all_tools(config)
    tools_by_name = {t.name: t for t in tools}
    
    tool_calls = most_recent_message.tool_calls
    def get_tool(name):
        return tools_by_name.get(name)

    tasks = []
    for tc in tool_calls:
        t = get_tool(tc["name"])
        if t:
            tasks.append(execute_tool_safely(t, tc["args"], config))
        else:
            tasks.append(asyncio.sleep(0, result=f"Tool {tc['name']} not found"))

    observations = await asyncio.gather(*tasks)
    
    tool_outputs = [
        ToolMessage(content=obs, name=tc["name"], tool_call_id=tc["id"])
        for obs, tc in zip(observations, tool_calls)
    ]
    
    if state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls:
         return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
         
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})

async def compress_research(state: ResearcherState, config: RunnableConfig):
    """Summarize researcher findings."""
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = configurable_model.with_config({
        "model": configurable.compression_model,
        "max_tokens": configurable.compression_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.compression_model, config),
        "tags": ["langsmith:nostream"]
    })
    
    researcher_messages = state.get("researcher_messages", [])
    
    try:
        compression_prompt = compress_research_system_prompt.format(date=get_today_str())
        messages = [SystemMessage(content=compression_prompt)] + researcher_messages + [HumanMessage(content=compress_research_simple_human_message)]
        
        response = await synthesizer_model.ainvoke(messages)
        compressed = response.content
    except Exception as e:
        compressed = f"Error compressing: {e}"
        
    # Extract raw notes
    raw_notes = []
    for m in researcher_messages:
        if isinstance(m, ToolMessage):
            raw_notes.append(str(m.content))
        if isinstance(m, AIMessage):
            if m.content: raw_notes.append(str(m.content))
            
    return {
        "compressed_research": str(compressed),
        "raw_notes": raw_notes
    }

# Researcher Graph
researcher_builder = StateGraph(ResearcherState, output=ResearcherOutputState, config_schema=Configuration)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()

async def final_report_generation(state: AgentState, config: RunnableConfig):
    """Generate final report."""
    notes = state.get("notes", [])
    findings = "\n\n".join(notes)
    
    configurable = Configuration.from_runnable_config(config)
    writer_model_config = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"]
    }
    
    final_report_prompt = final_report_generation_prompt.format(
        research_brief=state.get("research_brief", ""),
        messages=get_buffer_string(state.get("messages", [])),
        findings=findings,
        date=get_today_str()
    )
    
    final_report = await configurable_model.with_config(writer_model_config).ainvoke([
        HumanMessage(content=final_report_prompt)
    ])
    
    return {
        "messages": [final_report]
    }


# Main Graph
deep_researcher_builder = StateGraph(AgentState, input=AgentInputState, config_schema=Configuration)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)  # Use subgraph directly
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# Export graph
graph = deep_researcher_builder.compile()
