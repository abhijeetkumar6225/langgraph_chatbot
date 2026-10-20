from __future__ import annotations

import ipaddress
import os
import sqlite3
import socket
import tempfile
from typing import Annotated, Dict, Optional, TypedDict
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("ASHNA_API_KEY"),
    base_url="https://api.ashna.ai/v1/api",
    temperature=0,
)
embeddings = OpenAIEmbeddings(
    model="voyage-4-lite",
    api_key=os.getenv("ASHNA_API_KEY"),
    base_url="https://api.ashna.ai/v1/api",
    check_embedding_ctx_length=False,
)

# -------------------
# 2. Document retriever store (per thread)
# -------------------
_THREAD_VECTOR_STORES: Dict[str, InMemoryVectorStore] = {}
_THREAD_METADATA: Dict[str, list[dict]] = {}


def _get_retriever(thread_id: Optional[str]):
    """Create a retriever for the indexed sources in a thread."""
    if thread_id and thread_id in _THREAD_VECTOR_STORES:
        return _THREAD_VECTOR_STORES[thread_id].as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
    return None


def _index_documents(
    documents: list[Document], thread_id: str, metadata: dict
) -> dict:
    if not thread_id.strip():
        raise ValueError("A chat thread ID is required for document ingestion.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = [
        chunk
        for chunk in splitter.split_documents(documents)
        if chunk.page_content.strip()
    ]
    if not chunks:
        raise ValueError("The source contains no non-empty text to index.")

    for chunk in chunks:
        chunk.metadata.update(
            source=metadata["filename"],
            source_type=metadata["source_type"],
        )

    thread_key = str(thread_id)
    vector_store = _THREAD_VECTOR_STORES.get(thread_key)
    if vector_store is None:
        _THREAD_VECTOR_STORES[thread_key] = InMemoryVectorStore.from_documents(
            chunks, embeddings
        )
    else:
        vector_store.add_documents(chunks)

    _THREAD_METADATA.setdefault(thread_key, []).append({
        **metadata,
        "documents": len(documents),
        "chunks": len(chunks),
    })
    return {
        **metadata,
        "documents": len(documents),
        "chunks": len(chunks),
    }


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Add an uploaded PDF to the retriever for this thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        indexable_docs = [
            document for document in docs if document.page_content.strip()
        ]
        source_name = filename or os.path.basename(temp_path)
        for document in indexable_docs:
            document.metadata["source"] = source_name

        return _index_documents(
            indexable_docs,
            thread_id,
            {"filename": source_name, "source_type": "PDF"},
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _validate_public_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Enter a complete public http:// or https:// URL.")
    if parsed.username or parsed.password:
        raise ValueError("URLs containing login credentials are not supported.")
    if parsed.port not in (None, 80, 443):
        raise ValueError("Only standard HTTP and HTTPS ports are supported.")

    hostname = parsed.hostname.rstrip(".")
    if hostname.lower() == "localhost":
        raise ValueError("Local and private network URLs cannot be indexed.")
    try:
        addresses = {ipaddress.ip_address(hostname)}
    except ValueError:
        try:
            addresses = {
                ipaddress.ip_address(result[4][0])
                for result in socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
            }
        except OSError as exc:
            raise ValueError(f"Could not resolve the URL host: {hostname}") from exc

    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Local and private network URLs cannot be indexed.")


def _fetch_web_page(url: str) -> tuple[str, str]:
    """Fetch a public HTML page, checking every redirect and limiting its size."""
    current_url = url.strip()
    max_bytes = 5 * 1024 * 1024

    for redirect_count in range(6):
        _validate_public_url(current_url)
        response = requests.get(
            current_url,
            headers={"User-Agent": "NeuralCoreRAG/1.0"},
            timeout=(5, 15),
            allow_redirects=False,
            stream=True,
        )
        try:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("Location")
                if not location:
                    raise ValueError(
                        "The website returned a redirect without a destination."
                    )
                if redirect_count == 5:
                    raise ValueError("The website redirected too many times.")
                current_url = urljoin(current_url, location)
                continue

            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if (
                "text/html" not in content_type
                and "application/xhtml+xml" not in content_type
            ):
                raise ValueError("The URL must point to an HTML web page.")

            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                raise ValueError("The web page is larger than the 5 MB indexing limit.")

            body = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                body.extend(chunk)
                if len(body) > max_bytes:
                    raise ValueError("The web page is larger than the 5 MB indexing limit.")
            return (
                bytes(body).decode(response.encoding or "utf-8", errors="replace"),
                current_url,
            )
        finally:
            response.close()

    raise ValueError("The website redirected too many times.")


def ingest_url(url: str, thread_id: str) -> dict:
    """Fetch a web page, split its readable text, and add it to this thread's index."""
    if not url.strip():
        raise ValueError("Enter a URL to index.")
    if not thread_id.strip():
        raise ValueError("A chat thread ID is required for document ingestion.")

    html_content, final_url = _fetch_web_page(url)
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
        element.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else final_url
    page_text = soup.get_text("\n", strip=True)
    if not page_text:
        raise ValueError("The web page contains no readable text to index.")

    document = Document(
        page_content=f"Title: {title}\nURL: {final_url}\n\n{page_text}",
        metadata={"source": final_url, "title": title},
    )
    return _index_documents(
        [document],
        thread_id,
        {"filename": final_url, "source_type": "Web page", "title": title},
    )


def delete_thread_document(thread_id: str) -> None:
    """Remove the in-memory document index associated with a chat thread."""
    _THREAD_VECTOR_STORES.pop(str(thread_id), None)
    _THREAD_METADATA.pop(str(thread_id), None)


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    if operation == "add":
        result = first_num + second_num
    elif operation == "sub":
        result = first_num - second_num
    elif operation == "mul":
        result = first_num * second_num
    elif operation == "div":
        if second_num == 0:
            return {"error": "Division by zero is not allowed"}
        result = first_num / second_num
    else:
        return {"error": f"Unsupported operation '{operation}'"}

    return {
        "first_num": first_num,
        "second_num": second_num,
        "operation": operation,
        "result": result,
    }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    response = requests.get(
        "https://www.alphavantage.co/query",
        params={
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": os.getenv("ALPHA_VANTAGE_API_KEY"),
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json()


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Retrieve relevant information from documents and web pages indexed in the active chat.
    """
    thread_id = config.get("configurable", {}).get("thread_id")
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": (
                "No source is indexed for this chat. Upload a PDF or add a web page "
                "URL first."
            ),
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "sources": list(
            dict.fromkeys(
                item.get("source", "")
                for item in metadata
                if item.get("source")
            )
        ),
    }


tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def sanitize_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop assistant tool calls that never received a tool result.

    Interrupted runs (e.g. a failed API call mid tool loop) can leave an
    AIMessage with tool_calls but no matching ToolMessage, which the API
    rejects on every later turn. Replace such dangling calls with a plain
    text message so the conversation stays valid.
    """
    answered_ids = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage)
    }
    sanitized = []
    for message in messages:
        if isinstance(message, AIMessage) and message.tool_calls:
            dangling = [
                c for c in message.tool_calls if c.get("id") not in answered_ids
            ]
            if dangling:
                text = message.text or "Sorry, that lookup was interrupted."
                sanitized.append(AIMessage(content=text))
                continue
        sanitized.append(message)
    return sanitized


def _indexed_sources_summary(thread_id: Optional[str]) -> Optional[str]:
    """Describe the sources indexed for a thread, for the system prompt."""
    sources = _THREAD_METADATA.get(str(thread_id), [])
    if not sources:
        return None
    return "\n".join(
        f"- {item.get('source_type')}: {item.get('title') or item.get('filename')} "
        f"({item.get('chunks')} chunks)"
        for item in sources
    )


def chat_node(state: ChatState, config: RunnableConfig):
    """LLM node that may answer or request a tool call."""
    thread_id = config.get("configurable", {}).get("thread_id")
    sources = _indexed_sources_summary(thread_id)

    source_instructions = (
        "The user has attached the following sources to this chat:\n"
        f"{sources}\n"
        "When the user refers to 'the link', 'the page', 'the URL', or 'the PDF', "
        "they mean these attached sources. Use the `rag_tool` to answer questions "
        "about their content.\n"
        if sources
        else ""
    )

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. "
            f"{source_instructions}"
            "For questions about an uploaded PDF or an indexed web page, use the `rag_tool` and answer from its retrieved context. "
            "If the context does "
            "not contain the answer, say so instead of guessing. The active chat is "
            "provided to the tool automatically. You can also use web search, stock "
            "price, and calculator tools when helpful. If no document is available, "
            "ask the user to upload a PDF or add a web page URL."
        )
    )

    messages = [system_message, *sanitize_messages(state["messages"])]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
database_path = os.path.join(os.path.dirname(__file__), "chatbot.db")
conn = sqlite3.connect(database=database_path, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
checkpointer.setup()

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------

def delete_thread_checkpoints(thread_id):
    """Delete a conversation and its associated in-memory PDF index."""
    checkpointer.delete_thread(thread_id)
    delete_thread_document(thread_id)


def delete_thread_history(thread_id):
    """Delete one conversation from the LangGraph SQLite checkpointer."""
    delete_thread_checkpoints(thread_id)

    # Verify that the thread no longer has saved graph state.
    config = {"configurable": {"thread_id": thread_id}}
    state = chatbot.get_state(config)

    if state.values:
        raise RuntimeError(
            f"Conversation {thread_id} still has saved LangGraph state."
        )


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_VECTOR_STORES


def thread_document_metadata(thread_id: str) -> dict:
    sources = _THREAD_METADATA.get(str(thread_id), [])
    return sources[-1] if sources else {}