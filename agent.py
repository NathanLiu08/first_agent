from pathlib import Path
import gzip
from html import unescape
from html.parser import HTMLParser
import re
import zlib
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen
from agents import Agent, Runner, function_tool


CONCERT_FILE = Path(__file__).parent / "concert.txt"
KNOWLEDGE_FILE = Path(__file__).parent / "school_knowledge.txt"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "high",
    "i",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "saratoga",
    "school",
    "shs",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
}


class DuckDuckGoParser(HTMLParser):
    """
    Extract result titles, URLs, and snippets from DuckDuckGo's HTML results page.
    """

    def __init__(self):
        super().__init__()
        self.results = []
        self._current = None
        self._last_result = None
        self._capture = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        css_class = attrs.get("class", "")

        if tag == "a" and "result__a" in css_class:
            self._current = {"title": "", "url": normalize_result_url(attrs.get("href", "")), "snippet": ""}
            self._capture = "title"
        elif tag in {"a", "div"} and "result__snippet" in css_class:
            self._capture = "snippet"

    def handle_data(self, data):
        if self._current and self._capture == "title":
            self._current[self._capture] += data
        elif self._last_result and self._capture == "snippet":
            self._last_result["snippet"] += data

    def handle_endtag(self, tag):
        if tag == "a" and self._current and self._capture == "title":
            if self._current["title"].strip() and self._current["url"]:
                self._last_result = {
                    "title": clean_text(self._current["title"]),
                    "url": self._current["url"],
                    "snippet": "",
                }
                self.results.append(self._last_result)
            self._current = None
            self._capture = None
        elif self._capture == "snippet" and tag in {"a", "div"}:
            if self._last_result:
                self._last_result["snippet"] = clean_text(self._last_result["snippet"])
            self._capture = None


class TextExtractor(HTMLParser):
    """
    Convert basic HTML content into readable text for retrieval.
    """

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "title"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag in {"p", "li", "h1", "h2", "h3", "h4"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        lines = [clean_text(line) for line in "".join(self.parts).splitlines()]
        return "\n\n".join(line for line in lines if line)


def clean_text(text: str) -> str:
    """
    Normalize whitespace and HTML entities.
    """
    return re.sub(r"\s+", " ", unescape(text)).strip()


def normalize_result_url(url: str) -> str:
    """
    Convert DuckDuckGo redirect URLs into direct result URLs when possible.
    """
    if not url:
        return ""

    parsed = urlparse(unescape(url))
    query = parse_qs(parsed.query)

    if "uddg" in query:
        return unquote(query["uddg"][0])

    if parsed.scheme and parsed.netloc:
        return unescape(url)

    return ""


def is_search_ad_url(url: str) -> bool:
    """
    Detect sponsored result redirect URLs.
    """
    parsed = urlparse(url)
    return (
        parsed.netloc.endswith("duckduckgo.com")
        and parsed.path.endswith("/y.js")
    ) or "ad_domain=" in parsed.query


def tokenize(text: str) -> set[str]:
    """
    Convert text into searchable keyword tokens.
    """
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in STOP_WORDS and len(token) > 1
    }


def load_concert_info() -> str:
    """
    Read concert information from concert.txt.
    """
    if not CONCERT_FILE.exists():
        return (
            "ERROR: concert.txt was not found. "
            "Please put concert.txt in the same folder as agent.py."
        )

    return CONCERT_FILE.read_text(encoding="utf-8")


def load_school_knowledge() -> str:
    """
    Read local Saratoga High School knowledge used by the RAG search tool.
    """
    knowledge_parts = []

    if KNOWLEDGE_FILE.exists():
        knowledge_parts.append(KNOWLEDGE_FILE.read_text(encoding="utf-8"))
    else:
        knowledge_parts.append(
            "ERROR: school_knowledge.txt was not found. "
            "Please put school_knowledge.txt in the same folder as agent.py."
        )

    if CONCERT_FILE.exists():
        knowledge_parts.append(
            "Title: Saratoga High School concert information\n"
            f"{CONCERT_FILE.read_text(encoding='utf-8')}"
        )

    return "\n\n".join(knowledge_parts)


def retrieve_relevant_chunks(query: str, knowledge: str, limit: int = 3) -> list[str]:
    """
    Return the most relevant text chunks for the query using simple lexical scoring.
    """
    query_tokens = tokenize(query)
    chunks = split_into_chunks(knowledge)

    if not query_tokens:
        return chunks[:limit]

    scored_chunks = []
    for index, chunk in enumerate(chunks):
        chunk_tokens = tokenize(chunk)
        score = len(query_tokens & chunk_tokens)
        if score:
            scored_chunks.append((score, -index, chunk))

    if scored_chunks:
        scored_chunks.sort(reverse=True)
        return [chunk for _, _, chunk in scored_chunks[:limit]]

    return chunks[:limit]


def split_into_chunks(text: str, max_chars: int = 1200) -> list[str]:
    """
    Split retrieved text into chunks small enough to rank usefully.
    """
    chunks = []
    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]

    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            chunks.append(paragraph)
            continue

        current = []
        current_length = 0
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if current and current_length + len(sentence) + 1 > max_chars:
                chunks.append(" ".join(current))
                current = []
                current_length = 0

            current.append(sentence)
            current_length += len(sentence) + 1

        if current:
            chunks.append(" ".join(current))

    return chunks


def fetch_url(url: str, timeout: int = 8, max_chars: int = 200_000) -> str:
    """
    Fetch a URL and return decoded text. Raises network exceptions to the caller.
    """
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ""

        raw_body = response.read(max_chars)
        content_encoding = response.headers.get("Content-Encoding", "").lower()
        if content_encoding == "gzip":
            raw_body = gzip.decompress(raw_body)
        elif content_encoding == "deflate":
            raw_body = zlib.decompress(raw_body)

        charset = response.headers.get_content_charset() or "utf-8"
        return raw_body.decode(charset, errors="replace")


def extract_page_text(html: str) -> str:
    """
    Extract readable text from HTML.
    """
    parser = TextExtractor()
    parser.feed(html)
    return parser.text()


def search_web_results(query: str, limit: int = 5) -> list[dict[str, str]]:
    """
    Search the web and return result metadata.
    """
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    html = fetch_url(url)
    parser = DuckDuckGoParser()
    parser.feed(html)
    organic_results = [
        result
        for result in parser.results
        if result["url"] and not is_search_ad_url(result["url"])
    ]
    return organic_results[:limit]


@function_tool
def search_school_knowledge(query: str) -> str:
    """
    Search local Saratoga High School knowledge.
    Use this tool whenever the user asks about Saratoga High School facts,
    including the address, campus, contact details, concerts, music events,
    dates, times, locations, tickets, performers, or schedules.
    """
    knowledge = load_school_knowledge()
    chunks = retrieve_relevant_chunks(query, knowledge)

    if not chunks:
        return (
            "No relevant local knowledge was found for that question. "
            "Do not answer from memory."
        )

    return "\n\n---\n\n".join(chunks)


def internet_rag_search(query: str) -> str:
    """
    Search the internet and retrieve source snippets for a query.
    """
    try:
        results = search_web_results(query)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        return f"Internet search failed: {exc}. Do not answer from memory."

    if not results:
        return "No internet search results were found. Do not answer from memory."

    retrieved = []
    for result in results[:3]:
        page_text = ""
        try:
            page_html = fetch_url(result["url"])
            page_text = extract_page_text(page_html)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError):
            page_text = result["snippet"]

        chunks = retrieve_relevant_chunks(query, page_text, limit=2) if page_text else []
        source_text = "\n".join(chunks) if chunks else result["snippet"]

        retrieved.append(
            "\n".join(
                [
                    f"Title: {result['title']}",
                    f"URL: {result['url']}",
                    f"Search result snippet: {result['snippet']}",
                    f"Retrieved text: {source_text[:2500]}",
                ]
            )
        )

    return "\n\n---\n\n".join(retrieved)


@function_tool
def search_internet(query: str) -> str:
    """
    Search the internet and retrieve source snippets for general questions.
    Use this tool for questions that require information outside the local
    Saratoga High School knowledge base or for current information.
    """
    return internet_rag_search(query)


agent = Agent(
    name="RAG Assistant",
    instructions=(
        "You are a helpful RAG assistant. "
        "For Saratoga High School questions, first call search_school_knowledge with "
        "the user's question. "
        "For general questions, random questions, or anything that may need current "
        "or external information, call search_internet with the user's question. "
        "Answer using only retrieved knowledge from the tools. "
        "When using internet results, include the source URLs you used. "
        "If the retrieved knowledge does not contain the answer, say you could not "
        "find that information in the retrieved sources. Do not make up details."
    ),
    tools=[search_school_knowledge, search_internet],
)


def main():
    print("RAG Assistant")
    print("Ask me about Saratoga High School or general internet-searchable information.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ")

        if question.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        result = Runner.run_sync(agent, question)
        print("\nAgent:", result.final_output)
        print()


if __name__ == "__main__":
    main()
