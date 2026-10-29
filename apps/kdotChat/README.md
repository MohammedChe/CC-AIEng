# K-Dot Chat

A small Retrieval-Augmented Generation (RAG) chatbot that answers questions about
Kendrick Lamar's lyrics, with a [Gradio](https://gradio.app) chat frontend on top
of a [LangChain](https://docs.langchain.com) retrieval chain.

It uses the exact same pattern taught in
[`notebooks/Prompt Engineering/04_Retrieval.ipynb`](../../notebooks/Prompt%20Engineering/04_Retrieval.ipynb):
load → split → embed → store (FAISS) → retrieve → generate — just packaged as a
standalone script with a chat UI instead of notebook cells.

## How it works

1. **Load**: `data/kdot.txt` (Kendrick Lamar's lyrics) is loaded with LangChain's
   `TextLoader`.
2. **Split**: `RecursiveCharacterTextSplitter` breaks the file into ~1000-character
   chunks with a 50-character overlap.
3. **Embed & store**: each chunk is embedded with `text-embedding-3-small` (via
   OpenRouter's OpenAI-compatible endpoint) and stored in a `FAISS` vector store.
   The index is cached to `apps/kdotChat/.faiss_cache/` so restarting the app
   doesn't re-embed everything (and re-spend on embeddings) every time — the
   cache is automatically rebuilt if `data/kdot.txt` changes.
4. **Retrieve & generate**: an LCEL chain (`retriever | prompt | llm | output parser`)
   retrieves the most relevant chunks for a question and asks the model to answer
   using only that context, citing the album it came from.
5. **Frontend**: a [`gradio.ChatInterface`](https://gradio.app/guides/chatinterface-examples)
   streams the model's answer token-by-token as it's generated.

For a deeper look at how LangChain chains connect to frontends like this one, see
the [LangChain frontend overview](https://docs.langchain.com/oss/python/langchain/frontend/overview).

## Setup

From the repository root:

```bash
# Activate the project's virtual environment
source .venv/bin/activate

# Install repo dependencies (includes gradio, langchain, faiss-cpu, etc.)
pip install -r requirements.txt

# If your requirements.txt predates this app, or you just want gradio on its own:
pip install gradio
```

You'll need an `OPENROUTER_API_KEY` set, either in a `.env` file at the repo root:

```env
OPENROUTER_API_KEY=your-key-here
MODEL=openai/gpt-6-luna   # optional, this is the default
```

or exported in your shell. If no key is found and you're running in an
interactive terminal, the app will prompt you for one; otherwise it exits with
a clear error.

## Run

```bash
python apps/kdotChat/app.py
```

Then open the URL printed in the terminal (defaults to
[http://127.0.0.1:7860](http://127.0.0.1:7860)).

The first run will build and cache the FAISS index (this takes a little while
since it embeds the whole lyrics file); subsequent runs load the cached index
and start almost instantly.

Host/port can be overridden with environment variables:

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=8000 python apps/kdotChat/app.py
```

## Notes

- Answers are grounded only in `data/kdot.txt`. If the context doesn't contain
  the answer, the bot responds: *"Ah snap homie, I ain't gonna front. I don't know."*
- Each question is answered independently (conversation history isn't fed back
  into retrieval), matching the scope of the source notebook.
- Useful references:
  - [Gradio ChatInterface examples guide](https://gradio.app/guides/chatinterface-examples)
  - [LangChain frontend overview](https://docs.langchain.com/oss/python/langchain/frontend/overview)
  - [`notebooks/Prompt Engineering/04_Retrieval.ipynb`](../../notebooks/Prompt%20Engineering/04_Retrieval.ipynb) — the notebook this app is based on
