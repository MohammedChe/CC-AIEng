# Rap Verse

A minimal example of turning a LangChain chain from the course notebooks into a
small Python app with a [Gradio](https://gradio.app) frontend.

It implements the last code cell in
[`notebooks/Prompt Engineering/03_Model_I_O.ipynb`](../../notebooks/Prompt%20Engineering/03_Model_I_O.ipynb):

1. **Prompt** — `ChatPromptTemplate` with a `{topic}` placeholder  
2. **Model** — `ChatOpenRouter`  
3. **Output parser** — `StrOutputParser`  

Those three steps are wired with LCEL: `chain = prompt | model | output_parser`.
The Gradio UI calls `chain.invoke({"topic": ...})` when you submit a topic.

For background on connecting chains to UIs, see the
[LangChain frontend overview](https://docs.langchain.com/oss/python/langchain/frontend/overview).

## Setup

From the repository root, with your virtual environment activated:

```bash
# Course dependencies (LangChain, OpenRouter, etc.) — Gradio is not included yet
pip install -r requirements.txt

# Frontend library for this exercise
pip install gradio

# See pinned versions (optional: save them to your own requirements file)
pip freeze
# pip freeze > requirements.txt
```

Add your OpenRouter key to a `.env` file at the repo root:

```env
OPENROUTER_API_KEY=your-key-here
```

**Model name:** if you do not set `MODEL`, the app uses the notebook default
`nvidia/nemotron-3-ultra-550b-a55b:free`. To use a different model, add for example:

```env
MODEL=your-model-id-on-openrouter
```

See [OpenRouter models](https://openrouter.ai/models) for IDs.

If no key is found and you run the app in an interactive terminal, it will prompt
you once; otherwise it exits with a clear error.

## Run

```bash
python apps/rapVerse/app.py
```

Open the URL printed in the terminal (default
[http://127.0.0.1:7861](http://127.0.0.1:7861)).

Optional environment variables:

```bash
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=8000 python apps/rapVerse/app.py
```

## What to try

Use the example **rainy days in Ireland** or enter your own topic (short phrases
work best). Each submit runs one chain invocation — there is no chat history.

## References

- [Gradio quickstart](https://www.gradio.app/guides/quickstart)
- [LangChain frontend overview](https://docs.langchain.com/oss/python/langchain/frontend/overview)
- [`03_Model_I_O.ipynb`](../../notebooks/Prompt%20Engineering/03_Model_I_O.ipynb) — Model I/O and LCEL chaining
