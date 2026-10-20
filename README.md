
# LangGraph Chatbot

A conversational AI chatbot built using **LangGraph, LangChain, Streamlit, and SQLite**, with an OpenAI-compatible LLM endpoint powered by Ashna API. The project demonstrates stateful conversations, tool calling, graph-based orchestration, and persistent chat history.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [How to Run](#how-to-run)
- [How It Works](#how-it-works)
- [Database and Memory](#database-and-memory)
- [Troubleshooting](#troubleshooting)
- [Security](#security)
- [Future Improvements](#future-improvements)

## Overview

This project implements a stateful AI chatbot using LangGraph and LangChain.

Unlike a basic chatbot that sends individual messages to a language model, this application uses a graph-based workflow to manage conversation state and route tool calls.

The application provides a Streamlit user interface and uses SQLite for local conversation checkpointing and related chat data.

The LLM is accessed through the Ashna API using LangChain's `ChatOpenAI` integration and an OpenAI-compatible API endpoint.

## Features

- **Conversational AI:** Interact with an LLM using natural language.
- **LangGraph workflow:** Organize chatbot execution using graph nodes and edges.
- **Conversation history:** Maintain messages across turns within a conversation thread.
- **SQLite persistence:** Store graph checkpoints locally.
- **Tool calling:** Allow the model to request supported tools.
- **Web search:** Search the web through DuckDuckGo when configured.
- **Calculator:** Perform calculations through a custom tool.
- **Stock lookup:** Retrieve stock-related information through Alpha Vantage if enabled.
- **Streaming responses:** Display model output incrementally in Streamlit.
- **Chat management:** Support conversation history and chat-management operations implemented in the frontend.
- **Python virtual environment:** Keep dependencies isolated from other projects.

> Available tools depend on which tools are registered in the backend and bound to the model.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.11 | Recommended Python runtime |
| Streamlit | Chat user interface |
| LangChain | LLM integration and tool abstractions |
| LangGraph | Graph orchestration and state management |
| langchain-openai | OpenAI-compatible chat model integration |
| langgraph-checkpoint-sqlite | SQLite-based graph checkpointing |
| SQLite | Local database storage |
| Ashna API | Language model access |
| DuckDuckGo | Web search, if enabled |
| Alpha Vantage | Stock data, if enabled |

## Project Structure

```text
langgraph_chatbot/
│
├── frontend_claude.py
│   └── Streamlit chat interface
│
├── database_rag_backend.py
│   └── LangGraph backend, model configuration,
│       tools, and graph execution
│
├── database.py
│   └── SQLite database helpers and chat records
│
├── requirements.txt
│   └── Python dependencies
│
├── .gitignore
│   └── Files excluded from Git
│
├── .env
│   └── Local API credentials (never commit)
│
└── chatbot.db
    └── Local SQLite data, depending on configuration
```

The `.env` and database files may be created locally and should not be uploaded to a public repository.

## Architecture

The chatbot follows a graph-based execution flow.

```text
        User
         |
         v
  Streamlit Frontend
         |
         v
  User Message + Thread ID
         |
         v
   LangGraph Chat Node
         |
         v
    Language Model
         |
         v
   tools_condition
         |
     +---+---+
     |       |
   Tool?    No Tool
     |       |
     v       v
  ToolNode  Response
     |
     v
  Chat Node
     |
     v
  Final Response
     |
     v
  Streamlit UI

SQLite Checkpointer
stores conversation state by thread ID.
```

## Prerequisites

Before starting, install:

- Python 3.11
- Git
- Visual Studio Code (optional)
- An Ashna API key with access to the configured model
- An Alpha Vantage API key if the stock lookup tool requires it

Check your Python version:

```powershell
py -3.11 --version
```

## Installation

### 1. Clone the repository

```powershell
git clone https://github.com/abhijeet6225/langgraph_chatbot.git

cd langgraph_chatbot
```

If the repository is private, sign in to GitHub and ensure you have access.

### 2. Create a virtual environment

```powershell
py -3.11 -m venv .venv
```

### 3. Activate the environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly.

### 4. Install dependencies

```powershell
python -m pip install --upgrade pip

python -m pip install -r requirements.txt
```

Verify the interpreter:

```powershell
python --version
```

The output should show Python 3.11.x.

## Environment Variables

Create a `.env` file in the project root if your backend loads environment variables from that file.

Example:

```dotenv
ASHNA_API_KEY=your_ashna_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
```

Only configure the variables actually required by your source code.

The model configuration used during development was:

```python
from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    model="gpt-4o-mini",
    base_url="https://api.ashna.ai/v1/api",
    api_key=os.getenv("ASHNA_API_KEY"),
    temperature=0
)
```

Check that the model name, endpoint, and environment-variable names match your current backend.

**Never commit API keys to GitHub.** Add `.env` to `.gitignore`.

## How to Run

From the project root, activate your virtual environment and execute:

```powershell
python -m streamlit run frontend_claude.py
```

Streamlit will start a local server. Open the URL printed in your terminal, usually:

http://localhost:8501

To stop the application, press `Ctrl+C` in the terminal.

### Open in Visual Studio Code

From the project directory:

```powershell
code .
```

In VS Code:

1. Press `Ctrl+Shift+P`.
2. Select `Python: Select Interpreter`.
3. Choose `.venv\Scripts\python.exe`.
4. Open the integrated terminal.
5. Run the Streamlit command.

## How It Works

### 1. User input

The user enters a message through the Streamlit frontend.

### 2. Thread configuration

The frontend passes the message and a conversation-specific `thread_id` to LangGraph.

### 3. State management

LangGraph retrieves the existing checkpoint for that thread when available.

The state can be defined as:

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
```

The `add_messages` reducer manages message updates in the graph state.

### 4. LLM execution

The chat node sends the conversation messages to the configured language model.

### 5. Tool execution

When the model requests a supported tool, LangGraph routes execution through `tools_condition` and `ToolNode`.

After the tool result is added to the conversation, the graph can call the model again.

### 6. Response generation

The model produces its response, which is displayed in the Streamlit interface.

### 7. Persistence

The SQLite checkpointer stores graph state associated with the thread ID, allowing conversations to continue across reruns and application restarts when persistent storage is configured correctly.

## Database and Memory

The project uses SQLite for local data storage.

### Checkpoints

LangGraph checkpoints store the graph's state, including conversation messages, associated with a thread.

### Chat metadata

The application-level database helpers may store information such as:

- Thread ID
- Chat title
- Creation time
- Last-updated time

The actual fields depend on the current `database.py` schema.

### Important considerations

- Reuse the same thread ID to continue an existing conversation.
- A new thread ID normally starts a separate conversation.
- Deleting chat metadata does not automatically guarantee that related checkpoints are deleted.
- Back up your database before changing its schema.
- Keep private conversations and database files out of public repositories.

SQLite is suitable for local development. A production deployment with multiple workers may require shared persistent storage and a production-appropriate checkpointer.

## Troubleshooting

### 1. No valid messages to send to the model

If the API returns:

```text
400 - No valid messages to send to the model
```

Check that the graph receives a non-empty list of valid messages.

Example frontend input:

```python
from langchain_core.messages import HumanMessage

inputs = {
    "messages": [
        HumanMessage(content=user_input)
    ]
}
```

Inspect the messages before the model invocation:

```python
def chat_node(state):
    messages = state.get("messages", [])

    if not messages:
        raise ValueError("The chatbot received no messages.")

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}
```

Also verify that the selected thread ID and checkpoint state are correct.

### 2. ModuleNotFoundError

Activate the correct virtual environment and reinstall the dependencies:

```powershell
python -m pip install -r requirements.txt
```

### 3. SQLite missing-column error

For an error such as:

```text
sqlite3.OperationalError: no such column: title
```

The existing database may use an older schema. `CREATE TABLE IF NOT EXISTS` does not automatically add missing columns.

Back up the database and perform a suitable schema migration before retrying.

### 4. API authentication errors

Verify:

- The API key is configured.
- The key is valid and has sufficient access.
- The model name and endpoint are correct.
- The environment variables are available in the terminal running Streamlit.

Do not print credentials in logs.

### 5. Chat history is not persistent

Check that:

- A persistent SQLite checkpointer is used.
- The graph is compiled with that checkpointer.
- The same thread ID is reused.
- The database path is stable.
- Checkpoints are not deleted by chat-management operations.

## Security

- Never upload `.env` or API credentials.
- Exclude `.venv`, local database files, and Python cache files from Git.
- Rotate any API key accidentally published.
- Treat external search results and tool output as untrusted input.
- Add authentication and authorization before exposing the application to other users.
- Protect stored conversations because checkpoint data may contain private messages.

## Future Improvements

Potential enhancements include:

- Better error handling and retries.
- Automated tests for tools and graph routing.
- Database migrations.
- Authentication and user-specific conversation access.
- Improved streaming and loading indicators.
- Production-grade database and checkpoint storage.
- Logging and monitoring that do not expose sensitive information.

## References

- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangGraph SQLite Checkpointer](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint-sqlite)
- [LangChain OpenAI Integration](https://python.langchain.com/docs/integrations/chat/openai/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

No license is specified. Add a `LICENSE` file if you want to define permissions for others to use, modify, or distribute this project.
