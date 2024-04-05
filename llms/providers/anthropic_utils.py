"""Tools to generate from Anthropic prompts."""

import anthropic
import os
import random
import time
from typing import Any

client = anthropic.Anthropic()  # defaults to os.environ.get("ANTHROPIC_API_KEY")


def retry_with_exponential_backoff(  # type: ignore
    func,
    initial_delay: float = 1,
    exponential_base: float = 1,
    jitter: bool = True,
    max_retries: int = 10,
    errors: tuple[Any] = (),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):  # type: ignore
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def generate_from_anthropic_chat_completion(
    messages: list[dict[str, str], str],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    context_length: int,
    stop_token: str | None = None,
) -> str:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable must be set when using Anthropic API."
        )
    assert type(messages[0]) == str, "First message must be a system prompt for Anthropic."
    response = client.messages.create(
        model=model,
        system=messages[0],
        max_tokens=max_tokens,
        messages=messages[1:],
        temperature=temperature,
        top_p=top_p
    )
    answer: str = response.content[0].text
    return answer

