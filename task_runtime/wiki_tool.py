"""wiki_tool.py — a minimal, cached Wikipedia lookup tool.

Three operations, all cached on disk so repeated calls are free and the test
corpus is reproducible across runs:

    search(term, limit=10)         -> list of {title, url, description}
    fetch_page(title)              -> plain-text article body (paragraph per line)
    grep(title, pattern, ...)      -> matching context snippets within the page

The two caches live side-by-side under WIKI_CACHE_DIR (default: wiki_cache/):

    wiki_cache/
        searches.json              -> {normalized term: [result, ...]}
        pages/<slugified title>.txt -> plain-text article extract

Both are append-only in spirit: once a search or page is cached, the cached
copy is returned forever unless you delete the file. That's deliberate — it
keeps experiments reproducible and avoids hammering Wikipedia.

Designed to be importable from an agent (`import wiki_tool; wiki_tool.search(...)`)
or used standalone via the CLI for manual exploration:

    python wiki_tool.py search "ashikaga yoshimitsu"
    python wiki_tool.py fetch "Kinkaku-ji"
    python wiki_tool.py grep "Kinkaku-ji" "phoenix"
    python wiki_tool.py stats
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser


_DEFAULT_WIKI_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "wiki_cache"
)
WIKI_CACHE_DIR = os.environ.get("WIKI_CACHE_DIR") or _DEFAULT_WIKI_CACHE_DIR
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks for a descriptive User-Agent. Identify the project + version;
# no contact URL since this is a local research tool.
USER_AGENT = "decompose-graph-agent/0.1 (research tool; local use)"

# Politeness: minimum seconds between external (non-cached) requests, even when
# we have plenty of headroom against Wikipedia's published limits — bursty
# clients from a single IP do still get throttled.
MIN_REQUEST_INTERVAL = float(os.environ.get("WIKI_MIN_INTERVAL", "0.25"))
# Retries on transient 429 / 5xx. Backoff starts at this many seconds and
# doubles each attempt.
MAX_RETRIES = int(os.environ.get("WIKI_MAX_RETRIES", "3"))
INITIAL_BACKOFF = float(os.environ.get("WIKI_INITIAL_BACKOFF", "1.0"))

# Tracking counters for cost visibility from callers.
search_call_count = [0]   # external (non-cached) search requests
fetch_call_count = [0]    # external (non-cached) page fetches
_last_request_time = [0.0]

_searches_cache: dict | None = None


# ---- helpers ---------------------------------------------------------------


def _ensure_cache_dirs() -> None:
    os.makedirs(WIKI_CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.join(WIKI_CACHE_DIR, "pages"), exist_ok=True)


def _searches_path() -> str:
    return os.path.join(WIKI_CACHE_DIR, "searches.json")


def _load_searches() -> dict:
    """Load (and memoize) the search cache."""
    global _searches_cache
    if _searches_cache is not None:
        return _searches_cache
    if os.path.exists(_searches_path()):
        with open(_searches_path(), encoding="utf-8") as f:
            _searches_cache = json.load(f)
    else:
        _searches_cache = {}
    return _searches_cache


def _save_searches() -> None:
    _ensure_cache_dirs()
    with open(_searches_path(), "w", encoding="utf-8") as f:
        json.dump(_searches_cache, f, ensure_ascii=False, indent=2, sort_keys=True)


def _slug(title: str) -> str:
    """Filesystem-safe slug from an article title. Stable across runs."""
    s = re.sub(r"[^\w\-.()' ]", "_", title.strip())
    s = s.replace(" ", "_")
    return s[:200] or "_"


def _page_path(title: str) -> str:
    return os.path.join(WIKI_CACHE_DIR, "pages", _slug(title) + ".txt")


def _http_get_json(url: str) -> dict:
    """GET a JSON resource. Politely paces requests and retries on 429 / 5xx
    with exponential backoff. Other HTTPErrors propagate immediately."""
    elapsed = time.monotonic() - _last_request_time[0]
    if elapsed < MIN_REQUEST_INTERVAL:
        time.sleep(MIN_REQUEST_INTERVAL - elapsed)

    backoff = INITIAL_BACKOFF
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                _last_request_time[0] = time.monotonic()
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            # Retry on rate limit + transient server errors
            if e.code in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                # Honor Retry-After if Wikipedia sends one; else exponential backoff.
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after and retry_after.isdigit() else backoff
                time.sleep(wait)
                backoff *= 2
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    # Should never reach here, but just in case:
    raise last_err if last_err else RuntimeError("wiki request failed")


def safe_print(s: str) -> None:
    """Windows-safe print: falls back to direct UTF-8 bytes when the console
    is cp1252 (which can't represent CJK / many Wikipedia titles)."""
    try:
        print(s)
    except UnicodeEncodeError:
        sys.stdout.buffer.write(s.encode("utf-8", errors="replace"))
        sys.stdout.buffer.write(b"\n")


# ---- public API ------------------------------------------------------------


def search(term: str, limit: int = 10) -> list[dict]:
    """Search Wikipedia for `term` via full-text search (action=query&list=search).

    Unlike opensearch — which is a title-prefix matcher and returns nothing for
    descriptive multi-word queries — this hits Wikipedia's full-text search index
    and accepts natural-language queries like "Trump appointed Supreme Court
    justices". Results are articles ranked by relevance to the query, each as
    {"title": str, "url": str, "description": str}. First call for a term hits
    the API; subsequent calls return the cached list (keyed on lowercased term).
    """
    if not term or not term.strip():
        return []
    cache = _load_searches()
    key = term.strip().lower()
    if key in cache:
        return cache[key]
    params = {
        "action": "query",
        "list": "search",
        "srsearch": term,
        "srlimit": str(limit),
        "srnamespace": "0",
        "srprop": "snippet",
        "format": "json",
    }
    url = WIKI_API_URL + "?" + urllib.parse.urlencode(params)
    data = _http_get_json(url)
    search_call_count[0] += 1
    hits = data.get("query", {}).get("search", []) or []
    results = []
    for h in hits:
        title = h.get("title", "")
        # Strip HTML markup that snippet uses for highlighting.
        snippet = re.sub(r"<[^>]+>", "", h.get("snippet", "")).strip()
        results.append({
            "title": title,
            "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
            "description": snippet,
        })
    cache[key] = results
    _save_searches()
    return results


# Cache-format marker. Cached page files written by v2 (and later) start with
# this exact line; cached files lacking it are treated as cache misses and
# transparently re-fetched. v2 adds table-row extraction on top of the v1
# plain-text extract — list articles (Wikipedia's tables are the actual list)
# now produce usable bodies, where v1 returned only the stripped prose intro.
_CACHE_MARKER = "<!-- wiki_tool v2 -->\n"


class _TableExtractor(HTMLParser):
    """Walk rendered Wikipedia HTML and pull out content from `wikitable` data
    tables (the class Wikipedia uses for actual data tables — sortable lists,
    statistics, rosters). Everything else — sidebar/series tables, navboxes,
    layout tables — is skipped because it's chrome rather than article data.

    Each kept `<table>` becomes a list of rows; each row is a list of cell
    strings. Also strips reference superscripts, edit-section links, and
    style/script blocks within cells. Nested tables inside a kept table are
    extracted as separate entries (and their rendered text also appears in
    the parent cell, which is harmless for substring search)."""

    def __init__(self):
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._cur_row: list[str] | None = None
        self._cur_cell: list[str] | None = None
        # One entry per currently-open <table>; True iff content should be captured.
        # A nested table inherits "not kept" from any non-kept ancestor.
        self._kept_stack: list[bool] = []
        self._n_skipped_tables_above = 0
        self._skip_depth = 0  # inline non-table skip subtree (e.g. <sup class=reference>)

    def _in_kept_table(self) -> bool:
        return bool(self._kept_stack) and self._kept_stack[-1]

    @staticmethod
    def _is_wikitable(attrs_d: dict) -> bool:
        cls = (attrs_d.get("class") or "").split()
        return "wikitable" in cls

    def handle_starttag(self, tag, attrs):
        if self._skip_depth > 0:
            self._skip_depth += 1
            return
        attrs_d = dict(attrs)
        if tag == "table":
            kept = self._is_wikitable(attrs_d) and self._n_skipped_tables_above == 0
            self._kept_stack.append(kept)
            if kept:
                self.tables.append([])
            else:
                self._n_skipped_tables_above += 1
            return
        if not self._in_kept_table():
            return
        cls = attrs_d.get("class") or ""
        if tag in ("style", "script") \
                or (tag == "sup" and "reference" in cls) \
                or (tag == "span" and "mw-editsection" in cls):
            self._skip_depth = 1
            return
        if tag == "tr":
            self._cur_row = []
        elif tag in ("td", "th"):
            self._cur_cell = []
        elif tag == "br" and self._cur_cell is not None:
            self._cur_cell.append(" ")

    def handle_startendtag(self, tag, attrs):
        if self._skip_depth > 0:
            return
        if tag == "br" and self._cur_cell is not None:
            self._cur_cell.append(" ")

    def handle_endtag(self, tag):
        if self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "table" and self._kept_stack:
            was_kept = self._kept_stack.pop()
            if not was_kept:
                self._n_skipped_tables_above -= 1
            return
        if not self._in_kept_table():
            return
        if tag in ("td", "th") and self._cur_cell is not None and self._cur_row is not None:
            cell = " ".join("".join(self._cur_cell).split())
            self._cur_row.append(cell)
            self._cur_cell = None
        elif tag == "tr" and self._cur_row is not None:
            if any(c for c in self._cur_row):
                self.tables[-1].append(self._cur_row)
            self._cur_row = None

    def handle_data(self, data):
        if self._skip_depth > 0:
            return
        if self._cur_cell is not None:
            self._cur_cell.append(data)


def _format_tables(tables: list[list[list[str]]]) -> str:
    """Render the extracted tables as plain text, one row per line, cells
    separated by ' | '. Skips empty tables and single-cell rows that are
    pure formatting noise."""
    chunks: list[str] = []
    for table in tables:
        useful = [r for r in table if len(r) >= 2 or (len(r) == 1 and len(r[0]) > 40)]
        if not useful:
            continue
        chunks.append("\n".join(" | ".join(row) for row in useful))
    return "\n\n".join(chunks)


def _fetch_parsed_tables(title: str) -> str:
    """Fetch the rendered HTML for `title` via action=parse and return the
    formatted table content. Empty string if no tables (or fetch fails). This
    is the v2-only addition that captures content the explaintext extract API
    drops on the floor (Wikipedia list articles live in tables)."""
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "redirects": "1",
    }
    url = WIKI_API_URL + "?" + urllib.parse.urlencode(params)
    try:
        data = _http_get_json(url)
    except urllib.error.HTTPError:
        return ""
    html = (data.get("parse") or {}).get("text", {}).get("*", "") or ""
    if not html:
        return ""
    extractor = _TableExtractor()
    try:
        extractor.feed(html)
    except Exception:
        return ""
    return _format_tables(extractor.tables)


def fetch_page(title: str) -> str:
    """Fetch the article body for `title` and return it as plain text.

    The body is the explaintext extract followed (if present) by a
    `== TABLES ==` section containing pipe-separated rows extracted from
    the rendered HTML. Tables are included because list articles store
    their actual list as tables, which the explaintext API drops.

    First call hits the API (extract + tables — two requests); subsequent
    calls read from the disk cache. Cached files written before v2 are
    transparently re-fetched. Empty result on missing article is NOT
    cached so transient failures can be retried.
    """
    if not title or not title.strip():
        return ""
    _ensure_cache_dirs()
    path = _page_path(title)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cached = f.read()
        if cached.startswith(_CACHE_MARKER):
            return cached[len(_CACHE_MARKER):]
        # else: pre-v2 cache; fall through and re-fetch
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "format": "json",
        "titles": title,
        "redirects": "1",
    }
    url = WIKI_API_URL + "?" + urllib.parse.urlencode(params)
    data = _http_get_json(url)
    fetch_call_count[0] += 1
    pages = data.get("query", {}).get("pages", {})
    if not pages:
        return ""
    page = next(iter(pages.values()))
    if page.get("missing"):
        return ""
    extract = page.get("extract", "") or ""
    tables = _fetch_parsed_tables(title)
    body = extract
    if tables:
        body = (extract.rstrip() + "\n\n== TABLES ==\n\n" + tables) if extract else tables
    if body:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_CACHE_MARKER + body)
    return body


def grep(title: str, pattern: str, line_context: int = 1,
         max_hits: int = 20, case_insensitive: bool = True) -> list[dict]:
    """Find `pattern` (regex or literal) in the cached page for `title`.

    Returns up to `max_hits` matches. Each hit:
        {
            "line": int,                # 0-indexed line number of the match
            "match": str,               # the literal text the regex matched
            "context_before": str,      # `line_context` lines before the hit (may be empty)
            "context_match": str,       # the full matching line, untruncated
            "context_after": str,       # `line_context` lines after the hit (may be empty)
        }

    Returning whole-line context avoids the mid-token truncation that
    character-windowed snippets produced (e.g., "...el Coney..." from
    "Michael Coney"). `line_context=1` gives the matching line plus one line
    on each side. `pattern` is tried as a regex first; if it doesn't compile,
    the literal escaped form is used.
    """
    text = fetch_page(title)
    if not text:
        return []
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error:
        regex = re.compile(re.escape(pattern), flags)
    lines = text.splitlines()
    hits: list[dict] = []
    seen_lines: set[int] = set()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        if i in seen_lines:
            continue
        m = regex.search(line)
        if not m:
            continue
        seen_lines.add(i)
        before = "\n".join(lines[max(0, i - line_context):i])
        after = "\n".join(lines[i + 1:i + 1 + line_context])
        hits.append({
            "line": i,
            "match": m.group(0),
            "context_before": before,
            "context_match": line,
            "context_after": after,
        })
        if len(hits) >= max_hits:
            break
    return hits


# ---- inspection ------------------------------------------------------------


def cache_stats() -> dict:
    """Return a dict describing what's in the cache (no API calls)."""
    _ensure_cache_dirs()
    searches = _load_searches()
    pages_dir = os.path.join(WIKI_CACHE_DIR, "pages")
    page_files = [f for f in os.listdir(pages_dir) if f.endswith(".txt")] if os.path.isdir(pages_dir) else []
    total_bytes = 0
    for f in page_files:
        try:
            total_bytes += os.path.getsize(os.path.join(pages_dir, f))
        except OSError:
            pass
    return {
        "cache_dir": WIKI_CACHE_DIR,
        "searches_cached": len(searches),
        "pages_cached": len(page_files),
        "pages_total_kb": round(total_bytes / 1024, 1),
        "external_searches_this_session": search_call_count[0],
        "external_fetches_this_session": fetch_call_count[0],
    }


# ---- CLI -------------------------------------------------------------------


def _print_results(results: list[dict]) -> None:
    for r in results:
        safe_print(f"  {r['title']}")
        if r.get("description"):
            safe_print(f"    {r['description']}")
        if r.get("url"):
            safe_print(f"    {r['url']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cached Wikipedia lookup tool.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_search = sub.add_parser("search", help="Search Wikipedia for a term.")
    p_search.add_argument("term", nargs="+", help="The search term.")
    p_search.add_argument("--limit", type=int, default=10)

    p_fetch = sub.add_parser("fetch", help="Fetch the plain-text body of a page.")
    p_fetch.add_argument("title", nargs="+", help="The article title (e.g. \"Kinkaku-ji\").")
    p_fetch.add_argument("--head", type=int, default=None,
                         help="Print only the first N characters of the article.")

    p_grep = sub.add_parser("grep", help="Find a pattern in a cached page.")
    p_grep.add_argument("title", help="The article title.")
    p_grep.add_argument("pattern", help="Regex or literal string to search for.")
    p_grep.add_argument("--max-hits", type=int, default=20)
    p_grep.add_argument("--context", type=int, default=1,
                        help="Number of full lines of context above and below each match (default 1).")

    sub.add_parser("stats", help="Show what's currently cached.")

    args = parser.parse_args()

    if args.cmd == "search":
        term = " ".join(args.term)
        results = search(term, limit=args.limit)
        safe_print(f"{len(results)} result(s) for {term!r}:")
        _print_results(results)
        safe_print(f"\n  ({'cached' if search_call_count[0] == 0 else 'API call made'})")
    elif args.cmd == "fetch":
        title = " ".join(args.title)
        text = fetch_page(title)
        if not text:
            sys.exit(f"no article body returned for {title!r}")
        if args.head is not None:
            safe_print(text[:args.head])
            if len(text) > args.head:
                safe_print(f"\n... [truncated; full article is {len(text)} chars]")
        else:
            safe_print(text)
        sys.stderr.write(f"\n[{len(text)} chars; {'cached' if fetch_call_count[0] == 0 else 'API call made'}]\n")
    elif args.cmd == "grep":
        hits = grep(args.title, args.pattern, line_context=args.context, max_hits=args.max_hits)
        if not hits:
            safe_print(f"no matches for {args.pattern!r} in {args.title!r}")
            return
        for h in hits:
            safe_print(f"  line {h['line']:4d}  {h['match']!r}")
            if h.get("context_before"):
                safe_print(f"      | {h['context_before']}")
            safe_print(f"      > {h['context_match']}")
            if h.get("context_after"):
                safe_print(f"      | {h['context_after']}")
    elif args.cmd == "stats":
        stats = cache_stats()
        for k, v in stats.items():
            safe_print(f"  {k:36s} {v}")


if __name__ == "__main__":
    main()
