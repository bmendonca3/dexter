import json
import os
import time
from typing import Any, List, Optional, Type, Union

import requests
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ValidationError

from dexter.prompts import DEFAULT_SYSTEM_PROMPT


MAX_RETRIES = 3
DEFAULT_TIMEOUT = 60  # seconds


def _get_llm_config() -> dict[str, str]:
    """
    Resolve the model provider configuration. Defaults to xAI Grok 4.
    """
    provider = os.getenv("DEXTER_LLM_PROVIDER", "xai").strip().lower()

    if provider == "openai":
        return {
            "provider": provider,
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1"),
        }

    # Default to xAI / Grok 4
    return {
        "provider": "xai",
        "api_key": os.getenv("XAI_API_KEY", ""),
        "base_url": os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        "model": os.getenv("XAI_MODEL", "grok-4-fast-reasoning"),
    }


def _tool_to_openai_schema(tool: BaseTool) -> dict[str, Any]:
    """Convert a LangChain tool into an OpenAI-compatible function schema."""
    args_schema = getattr(tool, "args_schema", None)
    if args_schema is not None:
        if hasattr(args_schema, "model_json_schema"):
            parameters = args_schema.model_json_schema()
        elif hasattr(args_schema, "schema"):
            parameters = args_schema.schema()
        else:
            parameters = {"type": "object", "properties": {}}
    else:
        parameters = {"type": "object", "properties": {}}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": getattr(tool, "description", "") or "",
            "parameters": parameters,
        },
    }


def _prepare_messages(prompt: str, system_prompt: Optional[str], schema: Optional[Type[BaseModel]]) -> List[dict[str, str]]:
    """Create a chat history payload compatible with the OpenAI/xAI APIs."""
    final_system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
    messages: List[dict[str, str]] = [{"role": "system", "content": final_system_prompt}]

    if schema is not None:
        schema_json = json.dumps(_schema_to_json(schema), indent=2)
        messages.append(
            {
                "role": "system",
                "content": (
                    "You must respond with a JSON object that strictly matches this schema. "
                    "Do not include any surrounding text or markdown fences.\n"
                    f"{schema_json}"
                ),
            }
        )

    messages.append({"role": "user", "content": prompt})
    return messages


def _schema_to_json(schema: Type[BaseModel]) -> dict[str, Any]:
    """Return the JSON schema for a Pydantic model."""
    if hasattr(schema, "model_json_schema"):
        return schema.model_json_schema()
    if hasattr(schema, "schema"):
        return schema.schema()
    raise ValueError("Provided schema does not expose a JSON schema representation.")


def _strip_code_fences(text: str) -> str:
    """Remove ```json fences if present."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def _parse_structured_output(raw_content: str, schema: Type[BaseModel]) -> BaseModel:
    """Parse JSON content into the requested Pydantic schema."""
    try:
        payload = json.loads(_strip_code_fences(raw_content))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse structured output as JSON: {exc}") from exc

    try:
        if hasattr(schema, "model_validate"):
            return schema.model_validate(payload)
        return schema.parse_obj(payload)  # type: ignore[attr-defined]
    except ValidationError as exc:
        raise ValueError(f"Llm output did not match expected schema: {exc}") from exc


def _to_ai_message(choice: dict[str, Any]) -> AIMessage:
    """Convert a chat completion choice into an AIMessage."""
    message = choice.get("message", {})
    content = message.get("content") or ""
    tool_calls_payload = message.get("tool_calls") or []
    tool_calls = []
    for tool_call in tool_calls_payload:
        if tool_call.get("type") != "function":
            continue
        function_call = tool_call.get("function", {})
        raw_args = function_call.get("arguments") or "{}"
        try:
            parsed_args: Union[dict[str, Any], str] = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = raw_args
        tool_calls.append(
            {
                "id": tool_call.get("id"),
                "name": function_call.get("name"),
                "args": parsed_args,
            }
        )

    return AIMessage(content=content, tool_calls=tool_calls, additional_kwargs={"finish_reason": choice.get("finish_reason")})


def call_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    tools: Optional[List[BaseTool]] = None,
) -> Union[AIMessage, BaseModel]:
    """
    Call the configured LLM provider (default: Grok 4 via xAI) and return either
    a LangChain AIMessage or a structured Pydantic model.
    """
    config = _get_llm_config()
    if not config["api_key"]:
        raise RuntimeError(
            "No API key configured. Set XAI_API_KEY for Grok or OPENAI_API_KEY when using OpenAI."
        )

    messages = _prepare_messages(prompt, system_prompt, output_schema)

    payload: dict[str, Any] = {
        "model": config["model"],
        "messages": messages,
        "temperature": float(os.getenv("DEXTER_TEMPERATURE", "0")),
    }

    if tools:
        payload["tools"] = [_tool_to_openai_schema(tool) for tool in tools]
        payload["tool_choice"] = "auto"

    url = f"{config['base_url'].rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=DEFAULT_TIMEOUT)
            if response.status_code >= 400:
                # retry on transient errors
                if response.status_code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES - 1:
                    time.sleep(1.5 ** attempt)
                    continue
                raise RuntimeError(f"LLM request failed ({response.status_code}): {response.text}")

            data = response.json()
            if not data.get("choices"):
                raise RuntimeError("LLM returned no choices.")

            choice = data["choices"][0]
            if output_schema is not None:
                message = choice.get("message", {})
                content = message.get("content") or ""
                return _parse_structured_output(content, output_schema)

            return _to_ai_message(choice)
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            last_error = exc
            if attempt == MAX_RETRIES - 1:
                break
            time.sleep(1.5 ** attempt)

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} attempts: {last_error}")
