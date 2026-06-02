import math
import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[3] / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


@dataclass
class Section:
    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    # TODO: Parse one Markdown file into section-level records.
    #
    # Design decision: The retrieval unit is a heading section, not a whole file.
    #
    # Hints:
    # 1. Use HEADING_RE to detect Markdown headings.
    # 2. Track heading_path so citations include parent context.
    # 3. Each Section id should look like "refund_policy.md#refund-timeline".
    # 4. Tokens should include both headings and content.
    sections = []
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return []

    lines = content.splitlines()
    filename = path.name
    
    current_heading = filename
    path_stack = [filename]
    current_content = []
    
    def save_section():
        text = "\n".join(current_content).strip()
        if text:
            slug = slugify(current_heading)
            section_id = f"{filename}#{slug}"
            tokens = tokenize(current_heading + " " + text)
            sections.append(Section(
                id=section_id,
                file=filename,
                heading=current_heading,
                heading_path=list(path_stack),
                content=text,
                tokens=tokens
            ))

    for line in lines:
        match = HEADING_RE.match(line)
        if match:
            save_section()
            
            level = len(match.group(1))
            heading_text = match.group(2).strip()
            
            # Level 1 means keep index 0 (filename), so path_stack[:1] -> [filename]
            path_stack = path_stack[:level]
            if not path_stack:
                path_stack = [filename]
            path_stack.append(heading_text)
            
            current_heading = heading_text
            current_content = [line]
        else:
            current_content.append(line)
            
    save_section()
    return sections


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    # TODO: Persist the section index to .kb/index.json so it is inspectable.
    #
    # Hints:
    # 1. Create index_path.parent if it does not exist.
    # 2. Write {"sections": [...], "stats": {...}} as pretty JSON.
    # 3. Use section.to_dict() for each Section.
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": [s.to_dict() for s in sections],
        "stats": {
            "avg_doc_len": avg_doc_len,
            "files_indexed": files_indexed,
            "doc_freq": dict(doc_freq)
        }
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def rebuild_stats() -> None:
    # TODO: Rebuild doc_freq, avg_doc_len, and files_indexed from sections.
    #
    # Hints:
    # 1. files_indexed can be derived from the unique section.file values.
    # 2. doc_freq counts how many sections contain each token.
    # 3. avg_doc_len is the average token count across sections.
    global doc_freq, avg_doc_len, files_indexed
    
    doc_freq = Counter()
    total_tokens = 0
    unique_files = set()
    
    for section in sections:
        unique_files.add(section.file)
        total_tokens += len(section.tokens)
        
        unique_tokens = set(section.tokens)
        for token in unique_tokens:
            doc_freq[token] += 1
            
    files_indexed = len(unique_files)
    if sections:
        avg_doc_len = total_tokens / len(sections)
    else:
        avg_doc_len = 0.0


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    # TODO: Load .kb/index.json into the in-memory sections list.
    #
    # Hints:
    # 1. If index_path does not exist, return (0, 0).
    # 2. Read payload["sections"] and convert each item back to Section.
    # 3. Call rebuild_stats() after assigning sections.
    # 4. Return (files_indexed, sections_indexed).
    global sections
    if not index_path.exists():
        return 0, 0
    
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
            
        sections = [Section(**s) for s in payload.get("sections", [])]
        rebuild_stats()
        return files_indexed, len(sections)
    except Exception:
        return 0, 0


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    # TODO: Build an in-memory section index from docs/*.md.
    #
    # Hints:
    # 1. Read all Markdown files from docs_dir.
    # 2. Call parse_markdown() for each file.
    # 3. Call rebuild_stats() to compute BM25 metadata.
    # 4. Persist .kb/index.json with write_index_json().
    # 5. Call write_index_json() so students can inspect the generated index.
    # 6. Return (files_indexed, sections_indexed).
    global sections

    sections = []
    
    if docs_dir.exists() and docs_dir.is_dir():
        for file_path in docs_dir.glob("*.md"):
            file_sections = parse_markdown(file_path)
            sections.extend(file_sections)
            
    rebuild_stats()
    write_index_json()
    
    return files_indexed, len(sections)


def bm25_score(query_tokens: list[str], section: Section, k1: float = 1.5, b: float = 0.75) -> float:
    # TODO: Score one section for the query using BM25.
    #
    # Hints:
    # 1. Count term frequency in the section.
    # 2. Use doc_freq to give rare terms higher weight.
    # 3. Normalize by section length using avg_doc_len.
    # 4. Add a small boost when query terms appear in heading_path.
    score = 0.0
    N = len(sections)
    if N == 0 or avg_doc_len == 0:
        return 0.0
        
    term_counts = Counter(section.tokens)
    section_len = len(section.tokens)
    
    heading_tokens = set(tokenize(" ".join(section.heading_path)))
    
    for token in query_tokens:
        if token not in term_counts:
            continue
            
        tf = term_counts[token]
        df = doc_freq.get(token, 0)
        
        idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
        
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * (section_len / avg_doc_len))
        
        term_score = idf * (numerator / denominator)
        
        if token in heading_tokens:
            term_score *= 1.2
            
        score += term_score
        
    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
