import logging

import litellm
from minisweagent.models.litellm_model import LitellmModel

# Configure logging
from src.utils.log import logger
from src.notebook_tools import (
    NOTEBOOK_TOOLS,
    parse_notebook_tool_actions,
    parse_tool_calls_from_content,
)


class CustomToolLitellmModel(LitellmModel):
    """LitellmModel subclass that uses dedicated notebook tools instead of a single bash tool."""

    def __init__(self, *args, **kwargs):
        logger.info(
            "Initializing CustomToolLitellmModel with args: %s, kwargs: %s",
            args,
            kwargs,
        )
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

    # ------------------------------------------------------------------
    # Override: parse structured notebook tool calls
    # Falls back to content parsing for models (e.g. GLM) that embed
    # tool calls in the text instead of the tool_calls field.
    # ------------------------------------------------------------------
    def _parse_actions(self, response) -> list[dict]:
        message = response.choices[0].message
        tool_calls = message.tool_calls or []

        # Fallback: extract tool calls from content text
        if not tool_calls and message.content:
            tool_calls = parse_tool_calls_from_content(message.content)
            if tool_calls:
                logger.info(
                    "Extracted %d tool call(s) from content text (model did not use tool_calls field)",
                    len(tool_calls),
                )
            else:
                logger.warning(
                    "No tool_calls in response and could not parse any from content. "
                    "content=%r, finish_reason=%r",
                    (message.content or "")[:200],
                    response.choices[0].finish_reason,
                )

        return parse_notebook_tool_actions(
            tool_calls,
            format_error_template=self.config.format_error_template,
        )

    def query(self, *args, **kwargs):
        logger.debug("Calling query with args: %s, kwargs: %s", args, kwargs)
        result = super().query(*args, **kwargs)
        return result
