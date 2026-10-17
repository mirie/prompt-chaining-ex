# prompt-chaining-ex

A Python prompt-chaining example managed with [uv](https://docs.astral.sh/uv/).

Requires Python 3.12 or newer and an OpenAI API key.

Includes LangChain, LangChain OpenAI, and LangGraph.

## Get started

```sh
uv sync
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then run:

```sh
uv run python main.py
```

The example makes two OpenAI requests: one extracts laptop specifications, and
the next transforms them into JSON. The final response is parsed and printed as
formatted JSON. API usage incurs charges. `OPENAI_MODEL` defaults to
`gpt-4.1-mini` and can be changed in `.env`.

The `.env` file is ignored by Git; keep your API key there or in your environment.

## Add code and dependencies

Create a Python file, then run it with `uv run your_file.py`.
Add dependencies with `uv add package-name`; commit both `pyproject.toml` and
`uv.lock` when dependencies change.
