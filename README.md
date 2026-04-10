# Local CrewAI Multi-Agent Demo (Ollama)

A simple, fully local multi-agent system built with CrewAI and Ollama.

## Architecture

Researcher Agent -> Writer Agent

1. Senior Researcher (Tech Research Analyst)

- Investigates Local Small Language Models (SLMs)
- Produces 3 detailed benefit bullet points

2. Creative Writer (Tech Blog Writer)

- Consumes the research bullets
- Produces a short, 2-paragraph Markdown blog post

The crew runs in sequential mode, so the writer task uses the research output from the previous step.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install crewai langchain-community langchain-ollama
```

## Pull a Local Model

Use either `phi3` or `llama3`:

```bash
ollama run phi3
# or
ollama run llama3
```

If Ollama is not already running as a service:

```bash
ollama serve
```

## Run

```bash
python main.py
```

## Notes

- The script is configured to use `phi3` by default at `http://localhost:11434`.
- To switch models, edit `model_name` in `main.py` (for example, change to `llama3`).
- No OpenAI API key is required.
