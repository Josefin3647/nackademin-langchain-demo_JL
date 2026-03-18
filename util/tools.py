import math
from datetime import datetime, timezone
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain.tools import tool
from pathlib import Path

import re
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

from collections import deque

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression and return the result.

    Args:
        expression: A mathematical expression to evaluate.
                    Supports +, -, *, /, **, sqrt(), abs(), etc.
    """
    # Safe math evaluation with limited builtins
    allowed_names = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time in UTC."""
    now = datetime.now(timezone.utc)
    return f"Current UTC time: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}"

def get_web_search_tool(): 
    toolkit = RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(headers={}),
        allow_dangerous_requests=True,
    )
    return toolkit.get_tools()

ALLOWED_BASE_DIR = Path.home() / "OneDrive" / "Dokument" / "MLops_documents"

@tool
def read_local_file(file_path: str) -> str:
    """Read a local text document and return its contents.

    Args:
        file_path: Relative path to a file inside the documents folder.
    """
    try:
        path = (ALLOWED_BASE_DIR / file_path).resolve()

        # Stoppa sökvägar som försöker gå utanför tillåten katalog
        if not str(path).startswith(str(ALLOWED_BASE_DIR)):
            return "Error: Access denied. File must be inside the documents folder."

        if not path.exists():
            return f"Error: File not found: {file_path}"

        if not path.is_file():
            return f"Error: Not a file: {file_path}"

        # Försök läsa med flera encodings
        encodings = ["utf-8", "cp1252", "latin-1"]

        content = None
        for enc in encodings:
            try:
                content = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue

        if content is None:
            return "Error: Could not decode file with common encodings."

        # Undvik att skicka extremt stora filer direkt till modellen
        max_chars = 12000
        if len(content) > max_chars:
            return content[:max_chars] + "\n\n[File truncated]"

        return content

    except Exception as e:
        return f"Error reading file '{file_path}': {e}"
    
# tools.py
from __future__ import annotations

import ipaddress
import re
import socket
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urldefrag
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree as ET

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib3.util.retry import Retry


# =========================
# Konfiguration
# =========================

BASE_URL = "https://start.stockholm/"
ALLOWED_DOMAINS = {"start.stockholm"}  # lägg ev. till fler om ni vill
USER_AGENT = "StockholmsstadAgent/1.0 (+internal-use, respectful-crawler)"
REQUEST_TIMEOUT = 10
MAX_RESPONSE_BYTES = 2_000_000  # 2 MB
MIN_SECONDS_BETWEEN_REQUESTS = 1.5
MAX_PAGES_DEFAULT = 30
MAX_DEPTH_DEFAULT = 2
MAX_CHUNKS_RETURNED = 5
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


# =========================
# Hjälpklasser
# =========================

class RateLimiter:
    """Enkel global rate limiter: minst X sek mellan requests."""
    def __init__(self, min_interval_seconds: float):
        self.min_interval = min_interval_seconds
        self.lock = threading.Lock()
        self.last_request_ts = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_ts
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request_ts = time.time()


@dataclass
class DocumentChunk:
    url: str
    title: str
    chunk_id: int
    text: str


# =========================
# Säkerhets- och URL-hjälpare
# =========================

def normalize_url(url: str) -> Optional[str]:
    """Tar bort fragment, normaliserar, accepterar bara http/https."""
    if not url:
        return None

    url, _ = urldefrag(url.strip())
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return None

    # normalisera bort tom path
    normalized = parsed._replace(fragment="").geturl()
    return normalized


def is_private_or_local_host(hostname: str) -> bool:
    """Skydd mot SSRF via interna/private IP:n."""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror:
        return True  # kan inte resolvas => behandla som osäker

    for info in infos:
        ip = info[4][0]
        try:
            addr = ipaddress.ip_address(ip)
            if (
                addr.is_private
                or addr.is_loopback
                or addr.is_link_local
                or addr.is_reserved
                or addr.is_multicast
            ):
                return True
        except ValueError:
            return True

    return False


def is_allowed_domain(url: str, allowed_domains: Set[str]) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        return False

    # exakt domän eller subdomän till tillåten domän
    ok = any(host == d or host.endswith("." + d) for d in allowed_domains)
    if not ok:
        return False

    # extra SSRF-skydd
    if is_private_or_local_host(host):
        return False

    return True


def is_internal_link(url: str, allowed_domains: Set[str]) -> bool:
    return is_allowed_domain(url, allowed_domains)


def clean_text(html: str) -> Tuple[str, str]:
    """Plockar ut titel och brödtext från HTML."""
    soup = BeautifulSoup(html, "html.parser")

    # ta bort script/style/nav/footer om möjligt
    for tag in soup(["script", "style", "noscript", "svg", "img"]):
        tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # försök hitta main/article först
    main = soup.find("main") or soup.find("article") or soup.body or soup

    text = main.get_text(separator="\n", strip=True)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return title, text


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap

    return chunks


# =========================
# Kärnklass: crawl + RAG
# =========================

class StockholmSiteRAG:
    def __init__(
        self,
        base_url: str = BASE_URL,
        allowed_domains: Optional[Set[str]] = None,
        user_agent: str = USER_AGENT,
    ):
        self.base_url = base_url
        self.allowed_domains = allowed_domains or set(ALLOWED_DOMAINS)
        self.user_agent = user_agent

        self.session = self._build_session()
        self.rate_limiter = RateLimiter(MIN_SECONDS_BETWEEN_REQUESTS)
        self.robots = self._load_robots()

        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=5)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({"User-Agent": self.user_agent})
        return session

    def _load_robots(self) -> RobotFileParser:
        robots_url = urljoin(self.base_url, "/robots.txt")
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
        except Exception:
            # Om robots inte kan läsas väljer vi konservativt beteende:
            # vi tillåter bara sitemap-start och låg crawl-rate.
            pass
        return rp

    def _allowed_by_robots(self, url: str) -> bool:
        try:
            return self.robots.can_fetch(self.user_agent, url)
        except Exception:
            return False

    def _safe_get(self, url: str) -> Optional[requests.Response]:
        if not is_allowed_domain(url, self.allowed_domains):
            return None
        if not self._allowed_by_robots(url):
            return None

        self.rate_limiter.wait()

        try:
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT, stream=True, allow_redirects=True)
        except requests.RequestException:
            return None

        final_url = normalize_url(resp.url)
        if not final_url or not is_allowed_domain(final_url, self.allowed_domains):
            resp.close()
            return None

        content_type = resp.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xml" not in content_type:
            resp.close()
            return None

        content_length = resp.headers.get("Content-Length")
        if content_length:
            try:
                if int(content_length) > MAX_RESPONSE_BYTES:
                    resp.close()
                    return None
            except ValueError:
                pass

        # Läs max X bytes
        raw = b""
        try:
            for chunk in resp.iter_content(chunk_size=8192):
                raw += chunk
                if len(raw) > MAX_RESPONSE_BYTES:
                    resp.close()
                    return None
        except requests.RequestException:
            resp.close()
            return None

        resp._content = raw
        return resp

    def fetch_sitemap_urls(self, sitemap_url: Optional[str] = None, max_urls: int = 100) -> List[str]:
        sitemap_url = sitemap_url or urljoin(self.base_url, "/sitemap.xml")
        resp = self._safe_get(sitemap_url)
        if not resp:
            return []

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError:
            return []

        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        urls: List[str] = []

        # Vanlig urlset
        for loc in root.findall(".//sm:url/sm:loc", ns):
            if loc.text:
                url = normalize_url(loc.text)
                if url and is_allowed_domain(url, self.allowed_domains):
                    urls.append(url)

        # Om sitemap-index: följ undersitemaps försiktigt
        if not urls:
            for loc in root.findall(".//sm:sitemap/sm:loc", ns):
                if len(urls) >= max_urls:
                    break
                if not loc.text:
                    continue
                child_urls = self.fetch_sitemap_urls(loc.text, max_urls=max_urls - len(urls))
                urls.extend(child_urls)

        # unika, begränsade
        seen = set()
        result = []
        for u in urls:
            if u not in seen:
                seen.add(u)
                result.append(u)
            if len(result) >= max_urls:
                break

        return result

    def extract_internal_links(self, html: str, page_url: str) -> List[str]:
        soup = BeautifulSoup(html, "html.parser")
        links = []
        for a in soup.find_all("a", href=True):
            absolute = urljoin(page_url, a["href"])
            absolute = normalize_url(absolute)
            if not absolute:
                continue
            if is_internal_link(absolute, self.allowed_domains):
                links.append(absolute)

        # unika, behåll ordning
        seen = set()
        out = []
        for link in links:
            if link not in seen:
                seen.add(link)
                out.append(link)
        return out

    def crawl(
        self,
        start_urls: Optional[List[str]] = None,
        max_pages: int = MAX_PAGES_DEFAULT,
        max_depth: int = MAX_DEPTH_DEFAULT,
    ) -> int:
        if not start_urls:
            start_urls = self.fetch_sitemap_urls(max_urls=min(max_pages * 3, 100))
            if not start_urls:
                start_urls = [self.base_url]

        queue: List[Tuple[str, int]] = [(u, 0) for u in start_urls]
        visited: Set[str] = set()
        pages_indexed = 0
        collected_chunks: List[DocumentChunk] = []

        while queue and pages_indexed < max_pages:
            url, depth = queue.pop(0)

            if url in visited:
                continue
            visited.add(url)

            if depth > max_depth:
                continue

            resp = self._safe_get(url)
            if not resp:
                continue

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" not in content_type:
                continue

            html = resp.text
            title, text = clean_text(html)
            if len(text.strip()) < 200:
                continue

            page_chunks = split_text(text)
            for i, chunk in enumerate(page_chunks):
                collected_chunks.append(
                    DocumentChunk(
                        url=url,
                        title=title or url,
                        chunk_id=i,
                        text=chunk,
                    )
                )

            pages_indexed += 1

            if depth < max_depth:
                for link in self.extract_internal_links(html, url):
                    if link not in visited:
                        queue.append((link, depth + 1))

        self.chunks = collected_chunks
        self._build_index()
        return pages_indexed

    def _build_index(self) -> None:
        if not self.chunks:
            self.vectorizer = None
            self.matrix = None
            return

        docs = [
            f"{c.title}\n{c.url}\n{c.text}"
            for c in self.chunks
        ]
        self.vectorizer = TfidfVectorizer(
            stop_words=None,  # svenska stödjs inte bra som built-in; låt den vara
            max_features=20000,
            ngram_range=(1, 2),
        )
        self.matrix = self.vectorizer.fit_transform(docs)

    def search(self, query: str, top_k: int = MAX_CHUNKS_RETURNED) -> List[DocumentChunk]:
        if not query.strip():
            return []

        if not self.chunks or self.vectorizer is None or self.matrix is None:
            self.crawl()

        if not self.chunks or self.vectorizer is None or self.matrix is None:
            return []

        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix).flatten()
        ranked_idx = sims.argsort()[::-1]

        results = []
        seen_pairs = set()
        for idx in ranked_idx:
            chunk = self.chunks[idx]
            key = (chunk.url, chunk.chunk_id)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            results.append(chunk)
            if len(results) >= top_k:
                break

        return results


# =========================
# Tool-funktion för agenten
# =========================

_rag_instance: Optional[StockholmSiteRAG] = None


def get_stockholmsstad_rag() -> StockholmSiteRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = StockholmSiteRAG()
        # Första indexeringen kan göras lazy vid första sökningen.
    return _rag_instance


def search_stockholms_stad(query: str, top_k: int = 5, refresh: bool = False) -> Dict:
    """
    Tool som agenten kan anropa.

    Args:
        query: Vad som ska sökas efter.
        top_k: Antal chunks att returnera.
        refresh: Om True, bygg om index från webbplatsen.

    Returns:
        dict med relevanta chunks och källor.
    """
    rag = get_stockholmsstad_rag()

    if refresh or not rag.chunks:
        indexed_pages = rag.crawl(max_pages=30, max_depth=2)
    else:
        indexed_pages = None

    results = rag.search(query=query, top_k=top_k)

    return {
        "indexed_pages": indexed_pages,
        "query": query,
        "results": [
            {
                "title": r.title,
                "url": r.url,
                "chunk_id": r.chunk_id,
                "text": r.text,
            }
            for r in results
        ],
        "instructions_for_llm": (
            "Svara endast utifrån dessa källutdrag. "
            "Om underlaget inte räcker, säg att mer information behöver hämtas. "
            "Referera gärna till URL:erna i svaret."
        ),
    }