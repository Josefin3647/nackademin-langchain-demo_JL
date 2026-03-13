# LangChain Demo - Agent Examples

This project demonstrates various use cases and functionality in LangChain through practical agent examples.

## Getting Started

### Prerequisites
- Python 3.13
- Ollama server with access to Llama models

### Setup

1. Clone the project
2. Create a virtual environment and install dependencies:
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
3. Create a `.env` file with your configuration:
```bash
OLLAMA_BASE_URL=http://nackademin.icedc.se
OLLAMA_BEARER_TOKEN=your-bearer-token-here
```

### Running Examples

Make sure the virtual environment is activated and run from the project root:

```bash
source .venv/bin/activate
python -m examples.agent-lecture.simple_agent
```
