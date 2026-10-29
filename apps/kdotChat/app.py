"""K-Dot Chat: a Retrieval-Augmented Generation chatbot over Kendrick Lamar's lyrics.

This mirrors the LangChain pattern taught in
`notebooks/Prompt Engineering/04_Retrieval.ipynb`: load the lyrics file, split it
into chunks, embed the chunks, store them in a FAISS vector store, and answer
questions with an LCEL retrieval chain. The only addition here is a Gradio
`ChatInterface` frontend on top, and a few production niceties (index caching,
error handling, streaming).

Frontend reference: https://gradio.app/guides/chatinterface-examples
LangChain frontend overview: https://docs.langchain.com/oss/python/langchain/frontend/overview
"""

from __future__ import annotations

import getpass
import hashlib
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
import gradio as gr

from langchain_openrouter import ChatOpenRouter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = (APP_DIR / ".." / ".." / "data" / "kdot.txt").resolve()
CACHE_DIR = APP_DIR / ".faiss_cache"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 50
RETRIEVER_K = 4
EMBEDDING_MODEL = "openai/text-embedding-3-small"

QA_TEMPLATE = """

Use the following pieces of context to answer the question at the end. Cite the Album name that helped shape your answer.

If you don't know the answer, just say 'Ah snap homie, I ain't gonna front. I don't know.', don't try to make up an answer.

Use three sentences maximum, relevant analogies, and keep the answer as concise as possible.

Use the active voice, and speak directly to the reader using concise language.
{context}

Question: {question}

Helpful Answer:

"""


# --------------------------------------------------------------------------- #
# Environment setup
# --------------------------------------------------------------------------- #


def load_api_key() -> str:
    """Load OPENROUTER_API_KEY from the environment, prompting only when interactive."""
    load_dotenv(override=True)

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
        "(see apps/kdotChat/README.md) or export it before running this app."
    )


# --------------------------------------------------------------------------- #
# Index building / caching
# --------------------------------------------------------------------------- #


def cache_key_for(path: Path) -> str:
    """A stable cache key derived from the source file's size + mtime.

    This is cheap to compute and invalidates automatically whenever kdot.txt
    is edited or replaced, without having to hash the whole file every start.
    """
    stat = path.stat()
    fingerprint = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]


def build_vectorstore(embeddings: OpenAIEmbeddings) -> FAISS:
    """Load+split kdot.txt and build (or load a cached) FAISS vector store."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find the lyrics file at '{DATA_FILE}'. Make sure "
            "data/kdot.txt exists at the repo root."
        )

    key = cache_key_for(DATA_FILE)
    index_dir = CACHE_DIR / key

    if index_dir.exists():
        print(f"Loading cached FAISS index from {index_dir} ...")
        start = time.perf_counter()
        vectorstore = FAISS.load_local(
            str(index_dir), embeddings, allow_dangerous_deserialization=True
        )
        print(f"Loaded cached index in {time.perf_counter() - start:.2f}s")
        return vectorstore

    print(f"No cached index found for this version of {DATA_FILE.name}; building one now...")
    start = time.perf_counter()

    # https://docs.langchain.com/oss/python/integrations/document_loaders
    loader = TextLoader(str(DATA_FILE), encoding="utf-8")
    documents = loader.load()

    # https://docs.langchain.com/oss/python/integrations/splitters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks")

    # https://docs.langchain.com/oss/python/langchain/knowledge-base#faiss
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Built and cached index in {time.perf_counter() - start:.2f}s -> {index_dir}")

    return vectorstore


def build_chain():
    """Build the LCEL retrieval chain: retriever + prompt + llm + output parser."""
    api_key = load_api_key()
    model = os.getenv("MODEL", "openai/gpt-6-luna")

    # Embeddings go through OpenRouter's OpenAI-compatible endpoint, so only
    # OPENROUTER_API_KEY is needed (same as the notebook).
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        check_embedding_ctx_length=False,
    )

    vectorstore = build_vectorstore(embeddings)

    qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)
    llm = ChatOpenRouter(model=model, temperature=0)

    setup_and_retrieval = RunnableParallel(
        {
            "context": vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K}),
            "question": RunnablePassthrough(),
        }
    )
    return setup_and_retrieval | qa_prompt | llm | StrOutputParser()


try:
    chain = build_chain()
except Exception as exc:  # noqa: BLE001 - we want one clear message for any startup failure
    print(
        f"\nFailed to start K-Dot Chat: {exc}\n"
        "Check that OPENROUTER_API_KEY is set, data/kdot.txt exists, and you "
        "have a working network connection, then try again.\n",
        file=sys.stderr,
    )
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Gradio frontend
# --------------------------------------------------------------------------- #
# https://gradio.app/guides/chatinterface-examples


def respond(message: str, history: list):
    """Stream an answer for `message`, guarding against empty input and errors."""
    message = (message or "").strip()
    if not message:
        yield "Ask me something about Kendrick Lamar's lyrics!"
        return

    try:
        partial = ""
        for token in chain.stream(message):
            partial += token
            yield partial
        if not partial:
            yield "Ah snap homie, I ain't gonna front. I don't know."
    except Exception as exc:  # noqa: BLE001 - surface any per-turn failure to the chat UI
        yield f"Something went wrong talking to the model: {exc}"


demo = gr.ChatInterface(
    fn=respond,
    title="K-Dot Chat",
    description=(
        "Ask questions about Kendrick Lamar's lyrics. Answers are grounded in "
        "`data/kdot.txt` via a LangChain Retrieval-Augmented Generation (RAG) "
        "chain — see `notebooks/Prompt Engineering/04_Retrieval.ipynb` for the "
        "underlying pattern and the "
        "[LangChain frontend overview](https://docs.langchain.com/oss/python/langchain/frontend/overview) "
        "for how frontends like this one fit in."
    ),
    examples=[
        "What do grief, fear, envy, and desire stem from?",
        "What is hope?",
        "How does Kendrick talk about family?",
    ],
)

# Queue requests instead of firing unbounded concurrent calls at the API.
demo.queue()


# Only start the web server when this file is run directly (e.g. `python app.py`).
# `chain` and `demo` are still built above when the module is merely imported
# (e.g. by a test or `python -c "import app"`), but launch() won't fire in that
# case, so importing this module never opens a port.
if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
    )
