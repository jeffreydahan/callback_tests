# ADK Imports
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types # For types.Content
from typing import Optional, Any


# --- Agent Callbacks ---
def before_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    print(f"\n[Callback] ==> BEFORE AGENT: {callback_context.agent_name}")
    print(f"    State: {callback_context.state.to_dict()}")
    return None # Return None to allow agent to run

def after_agent_callback(callback_context: CallbackContext) -> Optional[types.Content]:
    print(f"\n[Callback] <== AFTER AGENT: {callback_context.agent_name}")
    print(f"    State: {callback_context.state.to_dict()}")
    return None # Return None to allow agent to run

# --- Model Callbacks ---
def before_model_callback(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmResponse]:
    print(f"\n[Callback] --> BEFORE MODEL CALL for agent: {callback_context.agent_name}")
    content_types = []
    for content in llm_request.contents:
        for part in content.parts:
            if part.text:
                content_types.append("text")
            if part.inline_data and part.inline_data.mime_type:
                content_types.append(part.inline_data.mime_type)
            if part.file_data and part.file_data.mime_type:
                content_types.append(part.file_data.mime_type)
    if content_types:
        print(f"    Content Types: {', '.join(sorted(list(set(content_types))))}")
    return None # Return None to proceed with the model call

def after_model_callback(callback_context: CallbackContext, llm_response: LlmResponse) -> Optional[LlmResponse]:
    if llm_response.content and llm_response.content.parts:
        # Check if there is a transfer_to_agent call in the response
        transfer_call_part = None
        for part in llm_response.content.parts:
            if part.function_call and part.function_call.name == "transfer_to_agent":
                transfer_call_part = part
                break

        # If a transfer is happening, decide if we need to add an explanation
        if transfer_call_part:
            # Check if the LLM already provided a text explanation
            has_text_reason = any(
                part.text.strip() for part in llm_response.content.parts if part.text
            )

            if not has_text_reason:
                # The LLM didn't provide a reason, so we'll add a generic one.
                try:
                    target_agent = transfer_call_part.args["agent_name"]
                    explanation_text = (
                        f"Handing off to `{target_agent}` to complete the request."
                    )
                    explanation_part = types.Part(text=explanation_text)
                    new_parts = [explanation_part] + list(llm_response.content.parts)
                    new_content = types.Content(
                        parts=new_parts, role=llm_response.content.role
                    )
                    # Return the modified response
                    return LlmResponse(
                        content=new_content,
                        usage_metadata=llm_response.usage_metadata,
                    )
                except (KeyError, TypeError):
                    # Fall through if args are not as expected
                    pass

    # Default behavior for all other cases (no transfer or transfer with reason)
    print(f"\n[Callback] <-- AFTER MODEL CALL for agent: {callback_context.agent_name}")
    if llm_response.usage_metadata:
        usage = llm_response.usage_metadata
        print("    Token Usage:")
        if usage.prompt_token_count:
            print(f"        Prompt: {usage.prompt_token_count}")
        if usage.candidates_token_count:
            print(f"        Candidates: {usage.candidates_token_count}")
        if usage.total_token_count:
            print(f"        Total: {usage.total_token_count}")
    return None

# --- Tool Callbacks ---
def before_tool_callback(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext) -> Optional[dict]:
    print(f"\n[Callback] --> BEFORE TOOL CALL: {tool.name}")
    print(f"    Agent: {tool_context.agent_name}")
    print(f"    Args: {args}")
    return None # Return None to proceed with the tool call

def after_tool_callback(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: dict) -> Optional[dict]:
    print(f"\n[Callback] <-- AFTER TOOL CALL: {tool.name}")
    print(f"    Agent: {tool_context.agent_name}")
    print(f"    Response: {tool_response}")
    return None # Return None to use the original tool response

search_format_agent = LlmAgent(
    name="search_format_agent",
    model="gemini-2.0-flash",
    description="searches the internet using google and formats outputs from other agents",
    instruction="""
    You MUST use the google_search tool to find the most recent stock price for
    the given stock ticker.

    Once you have the information, provide the stock price along with the most
    recent date in JSON format like this:
    {
        "ticker": "TSLA",
        "price": "173.95",
        "date": "2024-02-23"
    }
    """,

    # All before callbacks
    before_agent_callback=before_agent_callback,
    before_model_callback=before_model_callback,
    before_tool_callback=before_tool_callback,

    # All after callbacks
    after_agent_callback=after_agent_callback,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,

    # Tool
    tools=[google_search],

    # Transfer control
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# define root_agent
root_agent = LlmAgent(
    name="root_agent",
    model="gemini-2.0-flash",
    description="You are an agent that provides realtime stock quotes using the latest google data",
    instruction="""Your job is to delegate tasks to the correct sub-agent based on its description.
When you decide to transfer to a sub-agent, you MUST first output a brief, user-facing sentence explaining your reasoning for the transfer.
After the explanation, you MUST call the `transfer_to_agent` function.

For example, if the user asks for a stock price, you might say:
"I need to find the latest stock information, so I'll hand this over to the search agent."
Then you would call the `transfer_to_agent` function.
""",

    # All before callbacks
    before_agent_callback=before_agent_callback,
    before_model_callback=before_model_callback,
    before_tool_callback=before_tool_callback,

    # All after callbacks
    after_agent_callback=after_agent_callback,
    after_model_callback=after_model_callback,
    after_tool_callback=after_tool_callback,

    # Sub-Agent
    sub_agents=[search_format_agent],
)