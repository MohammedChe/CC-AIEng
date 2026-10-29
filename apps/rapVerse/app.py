"""Rap Verse: a minimal LangChain LCEL app with a Gradio frontend.

Mirrors the last code cell in
`notebooks/Prompt Engineering/03_Model_I_O.ipynb`:

    chain = prompt | model | output_parser
    chain.invoke({"topic": "..."})

The notebook chain is moved into this script; Gradio wraps a single `invoke` call.

LangChain frontend overview:
https://docs.langchain.com/oss/python/langchain/frontend/overview
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openrouter import ChatOpenRouter

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent.parent

VERSE_TEMPLATE = "Write a 4-bar rap verse about {topic}"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"


# --------------------------------------------------------------------------- #
# Environment setup
# --------------------------------------------------------------------------- #


def load_api_key() -> str:
    """Load OPENROUTER_API_KEY from the repo-root .env or the environment."""
    load_dotenv(REPO_ROOT / ".env", override=True)

    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        return api_key

    if sys.stdin.isatty():
        api_key = getpass.getpass("Enter your OpenRouter API key: ")
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
            return api_key

    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. Add it to a .env file at the repo root "
        "(see apps/rapVerse/README.md) or export it before running this app."
    )


# --------------------------------------------------------------------------- #
# LCEL chain (Model I/O)
# --------------------------------------------------------------------------- #


def build_chain():
    """Build prompt | ChatOpenRouter | StrOutputParser (same as the notebook)."""
    load_api_key()
    model_name = os.getenv("MODEL", DEFAULT_MODEL)

    prompt = ChatPromptTemplate.from_template(VERSE_TEMPLATE)
    model = ChatOpenRouter(model=model_name)
    output_parser = StrOutputParser()

    return prompt | model | output_parser


try:
    chain = build_chain()
except Exception as exc:  # noqa: BLE001 - one clear message for any startup failure
    print(
        f"\nFailed to start Rap Verse: {exc}\n"
        "Check that OPENROUTER_API_KEY is set (see apps/rapVerse/README.md) "
        "and try again.\n",
        file=sys.stderr,
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Gradio frontend
# --------------------------------------------------------------------------- #


def generate_verse(topic: str) -> str:
    """Run the LCEL chain for a topic string."""
    topic = (topic or "").strip()
    if not topic:
        return "Enter a topic for your verse."

    try:
        result = chain.invoke({"topic": topic})
        text = (result or "").strip()
        if not text:
            return "The model returned an empty response. Try another topic."
        return text
    except Exception as exc:  # noqa: BLE001 - surface failures in the UI, not the server
        return f"Something went wrong talking to the model: {exc}"


demo = gr.Interface(
    fn=generate_verse,
    inputs=gr.Textbox(
        label="Topic",
        placeholder="e.g. rainy days in Ireland",
        max_length=300,
        lines=1,
    ),
    outputs=gr.Textbox(label="Verse", lines=6, interactive=False),
    title="Rap Verse",
    description=(
        "This app is the last LCEL chain from "
        "`notebooks/Prompt Engineering/03_Model_I_O.ipynb`: "
        "**prompt** → **model** → **output parser**. "
        "Enter a topic and click Submit to run `chain.invoke({\"topic\": ...})`. "
        "See the [LangChain frontend overview](https://docs.langchain.com/oss/python/langchain/frontend/overview) "
        "for how UIs like this connect to chains."
    ),
    examples=[["rainy days in Ireland"]],
)

# Queue requests instead of firing unbounded concurrent calls at the API.
demo.queue()

# Only start the web server when this file is run directly (e.g. `python app.py`).
# `chain` and `demo` are still built above when the module is merely imported
# (e.g. by a test or `python -c "import app"`), but launch() won't fire in that
# case, so importing this module never opens a port.
if __name__ == "__main__":
    try:
        port = int(os.getenv("GRADIO_SERVER_PORT", "7861"))
    except ValueError:
        print(
            "GRADIO_SERVER_PORT must be an integer.",
            file=sys.stderr,
        )
        sys.exit(1)

    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=port,
    )
