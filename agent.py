import litellm
import re
import subprocess
import os
import shutil
import traceback
import json
from config import Config
import warnings

Config.validate()
litellm.drop_params = True
#litellm._turn_on_debug()
litellm.modify_params = True # Fix Bedrock Converse API message format issues

warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "Why the command will be used."
                    },
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that the task is complete",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "The reason for finishing the task"
                    }
                },
                "required": ["reason"]
            }
        }
    }
]



def clean_messages_for_pcss(messages):
    """
    Convert tool_calls and tool messages to plain text for PCSS.
    PCSS's vLLM can't handle tool_calls dicts in message history.
    """
    cleaned = []
    for msg in messages:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            # Convert tool_calls to readable text
            tool_text = "\n".join([
                f"Called {tc['function']['name']}({tc['function']['arguments'][:200]}...)" 
                if len(tc['function']['arguments']) > 200 
                else f"Called {tc['function']['name']}({tc['function']['arguments']})"
                for tc in msg["tool_calls"]
            ])
            content = msg.get("content") or ""
            cleaned.append({
                "role": "assistant",
                "content": (content + "\n" + tool_text).strip()
            })
        else:
            cleaned.append(msg)
    return cleaned

def query_lm(messages):
    """Query the language model and return the full message object."""
    model_name = Config.MODEL_NAME
    provider = model_name.split('/')[0]
    
    # Build base completion kwargs
    completion_kwargs = {
        "messages": messages,
        "tools": TOOLS,
    }
    
    if provider == "pcss":
        model_suffix = model_name.split('/')[1]
        is_ollama_model = "codellama" in model_suffix
        
        completion_kwargs.update({
            "model": f"openai/{model_suffix}",
            "tool_choice": "required",
            "api_key": Config.PCSS_API_KEY,
            "api_base": Config.PCSS_API_BASE,
        })
        
        # Ollama models via PCSS don't support parallel_tool_calls
        if not is_ollama_model:
            completion_kwargs["parallel_tool_calls"] = False
        
        # DeepSeek-V3.1 needs cleaned messages (no tools in history)
        if model_name == "pcss/DeepSeek-V3.1-vLLM":
            completion_kwargs["messages"] = clean_messages_for_pcss(messages)
    elif provider == "bedrock":
        completion_kwargs.update({
            "model": model_name,
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "api_key": Config.AWS_BEARER_TOKEN_BEDROCK,
            "api_base": Config.AWS_BEDROCK_API_BASE,
        })
    else:
        completion_kwargs.update({
            "model": model_name,
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        })
    
    response = litellm.completion(**completion_kwargs)
    return response.choices[0].message

env_vars = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1"
}

def save_readable_messages(messages: list, working_dir: str):
    """Save messages as readable JSON with unescaped newlines."""
    log_file = os.path.join(working_dir, "messages_readable.json")
    
    json_str = json.dumps(messages, indent=4)
    
    json_str = json_str.replace('\\n', '\n')
    
    with open(log_file, "w") as f:
        f.write(json_str)

def debug_print(label: str, content: str, step: int = None):
    sep = "=" * 60
    step_str = f" [STEP {step}]" if step is not None else ""
    indented = "\n".join("  " + line for line in content.splitlines())
    print(f"\n{sep}")
    print(f"{label}{step_str}")
    print(f"{sep}")
    print(indented)
    print()

def execute_action(command: str, cwd: str = None) -> tuple[str, int]:
    """Execute action, return (output, return_code)"""
    result = subprocess.run(
        command,
        shell=True,
        text=True,
        env=os.environ | env_vars,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=Config.COMMAND_TIMEOUT,
        cwd=cwd,
    )
    return result.stdout, result.returncode

def execute_tool_call(tool_call, working_dir: str = None) -> tuple[str, bool]:
    """
    Execute a tool call and return (result, should_terminate).
    """
    function_name = tool_call.function.name
    function_args = json.loads(tool_call.function.arguments)
    
    if function_name == "run_shell":
        command = function_args.get("command", "")
        output, return_code = execute_action(command, cwd=working_dir)
        
        if return_code != 0:
            result = f"[RETURN CODE: {return_code}]\n{output}"
        elif output.strip():
            result = output
        else:
            # Provide explicit success feedback for silent commands
            result = "[Command completed successfully with no output]"
        return result, False

    elif function_name == "finish":
        reason = function_args.get("reason", "Task completed")
        return f"Task finished: {reason}", True
    
    else:
        return f"Unknown tool: {function_name}", False


class TerminatingException(Exception):
    """Exception that signals the agent should stop."""

class NonTerminatingException(Exception):
    """Exception that the agent can recover from."""

class FormatError(NonTerminatingException):
    """Error in parsing LLM output format."""


def get_system_prompt(prompt_version: str = "v2") -> str:
    """
    """
    if prompt_version == "v1":
        from prompts_v1 import AGENT_SYSTEM_PROMPT_SHORT
    else:
        from prompts_v2 import AGENT_SYSTEM_PROMPT_SHORT
    return AGENT_SYSTEM_PROMPT_SHORT


def create_initial_messages(system_prompt: str, user_task: str = None) -> list[dict]:
    """
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    if user_task:
        messages.append({"role": "user", "content": user_task})
    
    return messages


def run_agent(
    task: str,
    prompt_version: str = "v2",
    working_dir: str = None,
    max_steps: int = None,
) -> list[dict]:
    """
    Run the agent with the given task.
    
    Args:
        task: The task description for the agent
        prompt_version: "v1" or "v2"
        working_dir: Override working directory (uses Config.WORKING_DIR if None)
        max_steps: Override max steps (uses Config.MAX_STEPS if None)
    
    Returns:
        The conversation messages list
    """
    # Get configuration with overrides
    working_dir = working_dir if working_dir is not None else Config.WORKING_DIR
    max_steps = max_steps if max_steps is not None else Config.MAX_STEPS

    system_prompt = get_system_prompt(prompt_version)
    
    messages = create_initial_messages(system_prompt, user_task=task)

    os.makedirs(working_dir, exist_ok=True)
    if os.path.exists(working_dir) and os.listdir(working_dir):
        shutil.rmtree(working_dir)
        os.makedirs(working_dir, exist_ok=True)

    # Run the agent loop
    step = 0
    while step < max_steps:
        step += 1
        try:
            message = query_lm(messages)
            
            # Build assistant message for history
            assistant_message = {"role": "assistant"}
            if message.content:
                assistant_message["content"] = message.content
            if message.tool_calls and len(message.tool_calls) > 0:
                assistant_message["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            messages.append(assistant_message)
            
            # Debug print
            if message.content:
                debug_print("LM OUTPUT", message.content, step)
            if message.tool_calls:
                tool_calls_str = "\n".join([
                    f"- {tc.function.name}:\n"
                    + "\n".join(
                        f"    {k}: {v}"
                        for k, v in json.loads(tc.function.arguments).items()
                    )
                    for tc in message.tool_calls
                ])
                debug_print("TOOL CALLS", tool_calls_str, step)            
            
            # Handle tool calls
            if message.tool_calls:
                tool_results = []
                should_terminate = False
                
                for tool_call in message.tool_calls:
                    result, terminate = execute_tool_call(tool_call, working_dir=working_dir)
                    tool_results.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
                    if terminate:
                        should_terminate = True
                        debug_print("FINISH", result, step)
                
                messages.extend(tool_results)
                save_readable_messages(messages, working_dir)

                # Debug print tool results
                tool_call_map = {tc.id: tc for tc in message.tool_calls}
                for result in tool_results:
                    tool_call = tool_call_map.get(result["tool_call_id"])
                    if tool_call and tool_call.function.name == "run_shell":
                        debug_print("COMMAND OUTPUT", result["content"], step)
                
                if should_terminate:
                    break
            else:
                messages.append({"role": "user", "content": "Please provide a response with tool calls or content."})
                
        except NonTerminatingException as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            debug_print("ERROR (NON-TERMINATING)", error_msg, step)
            messages.append({"role": "user", "content": str(e)})
        except TerminatingException as e:
            debug_print("STOPPING", f"{type(e).__name__}: {str(e)}", step)
            break
        except Exception as e:
            error_details = f"{type(e).__name__}: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            debug_print("UNEXPECTED ERROR", error_details, step)
            messages.append({"role": "user", "content": f"Error executing action: {type(e).__name__}: {str(e)}"})
    
    if step >= max_steps:
        debug_print("STOPPING", f"Reached maximum steps limit ({max_steps})", step)

    # Save messages
    messages_file = os.path.join(working_dir, "messages.json")
    with open(messages_file, "w") as f:
        json.dump(messages, f, indent=4)
    
    return messages

if __name__ == "__main__":
    run_agent(
        task="What is the weather right now in Warsaw?",
        prompt_version="v2",
    )
