from difflib import SequenceMatcher
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for exact-content comparisons."""
    return " ".join(text.casefold().split())


def unique_documents(documents: list[Document]) -> tuple[list[Document], list[str]]:
    """Remove exact duplicate chunks and assign stable Qdrant-compatible IDs."""
    unique: list[Document] = []
    ids: list[str] = []
    seen: set[str] = set()

    for document in documents:
        normalized = normalize_text(document.page_content)
        if not normalized or normalized in seen:
            continue

        seen.add(normalized)
        unique.append(document)
        ids.append(str(uuid5(NAMESPACE_URL, f"German_Basics:{normalized}")))

    return unique, ids


def unique_search_results(
    documents: list[Document], limit: int = 4, overlap_threshold: float = 0.3
) -> list[Document]:
    """Filter exact duplicates and overlapping chunks from one PDF page."""
    unique: list[Document] = []
    normalized_texts: list[str] = []

    for document in documents:
        normalized = normalize_text(document.page_content)
        if not normalized:
            continue

        is_duplicate = False
        for selected, selected_text in zip(unique, normalized_texts):
            if normalized == selected_text:
                is_duplicate = True
                break

            same_page = (
                document.metadata.get("source") == selected.metadata.get("source")
                and document.metadata.get("page") == selected.metadata.get("page")
                and document.metadata.get("page") is not None
            )
            if not same_page:
                continue

            shared_length = SequenceMatcher(
                None, normalized, selected_text, autojunk=False
            ).find_longest_match().size
            shorter_length = min(len(normalized), len(selected_text))
            if shorter_length and shared_length / shorter_length >= overlap_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(document)
            normalized_texts.append(normalized)
            if len(unique) >= limit:
                break

    return unique
