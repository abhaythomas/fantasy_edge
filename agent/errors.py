"""User-facing error helpers shared by the app interfaces."""

import re


def rate_limit_message(error: Exception) -> str:
    """Turn Groq's 429 details into a useful, non-technical next step."""
    lower_error = str(error).lower()

    retry_match = re.search(
        r"(?:try again in|retry after)\s*([\d.]+)\s*(ms|s|sec|seconds?)?",
        lower_error,
    )
    wait_text = "the limit window resets"
    if retry_match:
        amount, unit = retry_match.groups()
        wait_text = f"about {amount} {unit or 'seconds'}"

    if "tokens per minute" in lower_error or "tpm" in lower_error:
        limit_name = "the token-per-minute limit"
    elif "tokens per day" in lower_error or "tpd" in lower_error:
        limit_name = "the daily token limit"
    elif "requests per day" in lower_error or "rpd" in lower_error:
        limit_name = "the daily request limit"
    else:
        limit_name = "the request-per-minute limit"

    return (
        f"Groq's free-tier {limit_name} was reached. Wait until {wait_text}, "
        "then try again. Clearing a long conversation can also reduce token usage."
    )
