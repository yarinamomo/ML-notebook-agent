import litellm
from minisweagent.models.litellm_model import LitellmModel

# Configure logging
from src.utils.log import logger
from src.notebook_tools import NOTEBOOK_TOOLS, parse_notebook_tool_actions, parse_tool_calls_from_content
from minisweagent.models import GLOBAL_MODEL_STATS
import time


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump(mode="json"))
    if hasattr(value, "dict"):
        return _json_safe(value.dict())
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return str(value)


class CustomToolLitellmModel(LitellmModel):
    """LitellmModel subclass that uses dedicated notebook tools instead of a single bash tool."""

    def __init__(self, *args, **kwargs):
        logger.info("Initializing CustomToolLitellmModel with args: %s, kwargs: %s", args, kwargs)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Override: send notebook-specific tools instead of BASH_TOOL
    # ------------------------------------------------------------------
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            return litellm.completion(
                model=self.config.model_name,
                messages=messages,
                tools=NOTEBOOK_TOOLS,
                **(self.config.model_kwargs | kwargs),
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("RAW ERROR:", repr(e))
            raise

    # ------------------------------------------------------------------
    # Override: parse structured notebook tool calls
    # Falls back to content parsing for models (e.g. GLM) that embed
    # tool calls in the text instead of the tool_calls field.
    # ------------------------------------------------------------------
    def _parse_actions(self, response) -> list[dict]:
        choices = getattr(response, "choices", []) or []

        # Ensure the provider choice that contains tool_calls (or has
        # finish_reason == 'tool_calls') is presented as choices[0] so the
        # base LitellmModel (which uses response.choices[0] to build the
        # stored assistant message) and our parsing stay in sync.
        tool_choice_idx = None
        for i, choice in enumerate(choices):
            choice_message = getattr(choice, "message", None)
            if getattr(choice, "finish_reason", None) == "tool_calls":
                tool_choice_idx = i
                break
            if choice_message is not None and getattr(choice_message, "tool_calls", None):
                tool_choice_idx = i
                break

        # If a tool-calling choice exists and isn't first, swap it into slot 0.
        if tool_choice_idx is not None and tool_choice_idx != 0:
            try:
                choices[0], choices[tool_choice_idx] = choices[tool_choice_idx], choices[0]
                response.choices = choices
            except (AttributeError, TypeError, IndexError) as _:
                # Best-effort swap; if the response object is immutable, fall back
                # to selecting the choice for parsing without mutating.
                logger.debug("Could not swap response choices; proceeding without mutation")

        chosen_choice = choices[0] if choices else None

        message = getattr(chosen_choice, "message", None)
        tool_calls = getattr(message, "tool_calls", []) or []

        raw_message = {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
            "reasoning_content": getattr(message, "reasoning_content", None),
            "finish_reason": getattr(chosen_choice, "finish_reason", None),
            "tool_calls": [
                {
                    "id": getattr(tool_call, "id", None),
                    "name": getattr(getattr(tool_call, "function", None), "name", None),
                    "arguments": getattr(getattr(tool_call, "function", None), "arguments", None),
                }
                for tool_call in tool_calls
            ],
        }

        full_resp = {
            "id": getattr(response, "id", None),
            "created": getattr(response, "created", None),
            "model": getattr(response, "model", None),
            "object": getattr(response, "object", None),
            "usage": _json_safe(getattr(response, "usage", None)),
            "chosen_choice_index": 0 if chosen_choice is not None else None,
            "choices": [],
        }

        for choice in choices:
            choice_message = getattr(choice, "message", None)
            choice_dict = {
                "index": getattr(choice, "index", None),
                "finish_reason": getattr(choice, "finish_reason", None),
                "message": {
                    "role": getattr(choice_message, "role", None) if choice_message is not None else None,
                    "content": getattr(choice_message, "content", None) if choice_message is not None else None,
                    "tool_calls": [],
                },
            }

            for tool_call in getattr(choice_message, "tool_calls", []) or []:
                choice_dict["message"]["tool_calls"].append(
                    {
                        "id": getattr(tool_call, "id", None),
                        "type": getattr(tool_call, "type", None),
                        "function": {
                            "name": getattr(getattr(tool_call, "function", None), "name", None),
                            "arguments": getattr(getattr(tool_call, "function", None), "arguments", None),
                        },
                    }
                )

            full_resp["choices"].append(choice_dict)

        raw_message["full_response"] = full_resp

        if not tool_calls and getattr(message, "content", None):
            tool_calls = parse_tool_calls_from_content(message.content)
            if tool_calls:
                logger.info(
                    "Extracted %d tool call(s) from content text (model did not use tool_calls field)",
                    len(tool_calls),
                )
            else:
                logger.warning(
                    "No tool_calls in response and could not parse any from content. content=%r, finish_reason=%r",
                    (message.content or "")[:200],
                    getattr(chosen_choice, "finish_reason", None),
                )

        return parse_notebook_tool_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
            raw_message=raw_message,
        )

    def query(self, *args, **kwargs):
        """Query the provider directly and build the final message so the
        stored assistant message and parsed actions are taken from the same
        provider choice (the one that actually contains tool_calls).
        This avoids mutating site-packages and ensures call_id alignment.
        """
        logger.debug("Calling CustomToolLitellmModel.query with args: %s, kwargs: %s", args, kwargs)
        messages = args[0] if args else kwargs.get("messages")
        # Prepare messages as the base class does
        prepared = self._prepare_messages_for_api(messages or [])

        # Call the provider
        response = self._query(prepared, **(kwargs or {}))

        # Calculate cost and update global stats
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])

        # Choose the provider choice that contains tool_calls (or has finish_reason)
        choices = getattr(response, "choices", []) or []
        chosen_index = 0
        for i, c in enumerate(choices):
            cm = getattr(c, "message", None)
            if getattr(c, "finish_reason", None) == "tool_calls":
                chosen_index = i
                break
            if cm is not None and getattr(cm, "tool_calls", None):
                chosen_index = i
                break

        chosen_choice = choices[chosen_index] if choices else None
        chosen_message_obj = getattr(chosen_choice, "message", None) if chosen_choice is not None else None

        # Build raw_message for parsing/debug
        raw_message = {
            "role": getattr(chosen_message_obj, "role", "assistant"),
            "content": getattr(chosen_message_obj, "content", None),
            "reasoning_content": getattr(chosen_message_obj, "reasoning_content", None),
            "finish_reason": getattr(chosen_choice, "finish_reason", None) if chosen_choice is not None else None,
            "tool_calls": [
                {
                    "id": getattr(tc, "id", None),
                    "name": getattr(getattr(tc, "function", None), "name", None),
                    "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                }
                for tc in (getattr(chosen_message_obj, "tool_calls", []) or [])
            ],
        }

        # Parse actions from the chosen choice only
        tool_calls = getattr(chosen_message_obj, "tool_calls", []) or []
        actions = []
        if tool_calls:
            actions = parse_notebook_tool_actions(tool_calls, format_error_template=self.config.format_error_template, raw_message=raw_message)
        else:
            # Fallback: try extracting from content
            content_calls = parse_tool_calls_from_content(getattr(chosen_message_obj, "content", "") or "")
            if content_calls:
                actions = parse_notebook_tool_actions(content_calls, format_error_template=self.config.format_error_template, raw_message=raw_message)

        # Build final message dict consistent with base class expectations
        message = chosen_message_obj.model_dump() if chosen_message_obj is not None else {"role": "assistant", "content": None}
        # Ensure the serialized message carries the provider tool_calls and finish_reason
        if chosen_message_obj is not None:
            message["tool_calls"] = [
                {
                    "id": getattr(tc, "id", None),
                    "type": getattr(tc, "type", None),
                    "function": {
                        "name": getattr(getattr(tc, "function", None), "name", None),
                        "arguments": getattr(getattr(tc, "function", None), "arguments", None),
                    },
                }
                for tc in (getattr(chosen_message_obj, "tool_calls", []) or [])
            ]
            message["finish_reason"] = getattr(chosen_choice, "finish_reason", None)
        message["extra"] = {
            "actions": actions,
            "response": response.model_dump(),
            **cost_output,
            "timestamp": time.time(),
        }
        return message
