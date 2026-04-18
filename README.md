# Minimal coding agent

A small **tool-calling agent** built with [LiteLLM](https://github.com/BerriAI/litellm). It runs a loop: the model proposes `run_shell` or `finish` tool calls; commands execute in a working directory and results are fed back until the model finishes or a step limit is hit.

## Contents

| File | Role |
|------|------|
| `agent.py` | Agent loop, LiteLLM integration, tool execution |
| `config.py` | `MODEL_NAME`, API keys, timeouts, paths (from env) |
| `prompts_v1.py` / `prompts_v2.py` | System prompts (`v1` shorter, `v2` more guidance) |
| `prepare_env.py` | Clone repos for SWE-bench (`prepare_SWE_env`) or generic Git (`prepare_custom_env`) |
| `run_in_container.py` | Run `agent.py` inside Docker (custom Ubuntu image or SWE-bench image) |

## Requirements

- Python 3.10+
- **Git** (for `prepare_env.py`)  
- **Docker** (only if you use `run_in_container.py`)

## Setup

### Environment variables

The agent reads configuration from the environment. At minimum you need a model name and the matching API key for that provider.

- **`MODEL_NAME`** — LiteLLM-style id, e.g. `openai/gpt-5-mini-2025-08-07`, `anthropic/claude-…`, `openrouter/…`, `bedrock/…`, `pcss/…` (see `config.py` for supported providers and key names).
- **Provider API key** — e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, etc., depending on the provider prefix in `MODEL_NAME`.

Optional:

- `AGENT_WORKING_DIR` — where the agent runs shell commands (default `./working_dir`)
- `AGENT_MAX_STEPS` — max LLM steps (default `20`)
- `AGENT_COMMAND_TIMEOUT` — subprocess timeout in seconds (default `30`)
- `AGENT_PROMPT_VERSION` — `v1` or `v2` (default `v2`)

`config.py` validates keys on import.
