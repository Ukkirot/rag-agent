import re
import sys
from pathlib import Path
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama
from pypdf import PdfReader


def load_and_split_document(
    file_path: str, splitter: RecursiveCharacterTextSplitter
) -> tuple[list[str], list[dict]]:
    """
    Wczytuje dokument (PDF lub tekst) i dzieli go na chunki.
    Zwraca listę fragmentów oraz powiązane metadane (m.in. numer strony).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Plik '{file_path}' nie istnieje.")

    chunks = []
    metadatas = []

    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        total_pages = len(reader.pages)
        print(f"Przetwarzanie PDF ({total_pages} stron)...")

        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue

            page_chunks = splitter.split_text(text)
            for chunk in page_chunks:
                annotated_chunk = f"[Strona {idx}]\n{chunk}"
                chunks.append(annotated_chunk)
                metadatas.append({"source": path.name, "page": idx})
    else:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        raw_chunks = splitter.split_text(text)
        for chunk in raw_chunks:
            chunks.append(chunk)
            metadatas.append({"source": path.name, "page": 1})

    return chunks, metadatas


def get_embedding(text: str) -> list[float]:
    """Generuje embedding dla pojedynczego tekstu za pomocą nomic-embed-text."""
    response = ollama.embeddings(model="nomic-embed-text", prompt=text)
    return response["embedding"]


def retrieve_context(collection, query: str) -> tuple[list[str], list[dict]]:
    """
    Wykrywa intencję zapytania:
    - Pytanie o stronę -> filtr metadanych w ChromaDB
    - Pytanie tematyczne -> wyszukiwanie semantyczne (embeddingi)
    """
    page_match = re.search(
        r'(?:stron(?:a|ie|y)?\s*(\d+)|\b(\d+)\s*stron)', query, re.IGNORECASE
    )

    if page_match:
        page_num = int(page_match.group(1) or page_match.group(2))
        print(f"-> Wykryto pytanie o konkretną stronę: {page_num} (filtr metadanych)")
        results = collection.get(
            where={"page": page_num},
            limit=6
        )
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        return docs, metas

    # Standardowe wyszukiwanie wektorowe
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return docs, metas


def main():
    if len(sys.argv) < 2:
        print("Użycie: python test_rag.py <ścieżka_do_pliku> [zapytanie]")
        return

    doc_path = sys.argv[1]
    query = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "Jakie są najważniejsze informacje w tym dokumencie?"
    )

    # 1. Połączenie z trwałą bazą na dysku
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="test_rag_collection")

    existing_count = collection.count()

    # 2. Indeksowanie dokumentu tylko jeśli baza jest pusta
    if existing_count == 0:
        print(f"--- 1. Wczytywanie i dzielenie dokumentu: {doc_path} ---")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=80,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks, metadatas = load_and_split_document(doc_path, text_splitter)
        print(f"Wygenerowano {len(chunks)} fragmentów z metadanymi stron.")

        print("--- 2. Indeksacja i zapisywanie chunków do ChromaDB ---")
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_meta = metadatas[i : i + batch_size]
            batch_ids = [f"doc_chunk_{j}" for j in range(i, i + len(batch_chunks))]

            batch_embeddings = [get_embedding(chunk) for chunk in batch_chunks]

            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_chunks,
                metadatas=batch_meta,
            )
            print(f"Zaindeksowano fragmenty {min(i + batch_size, len(chunks))} / {len(chunks)}")
        print("Indeksacja zakończona pomyślnie i utrwalona na dysku!")
    else:
        print(f"--- Baza wektorowa zawiera {existing_count} fragmentów. Pomijanie indeksowania. ---")

    # 3. Pobranie pasujących fragmentów (hybrydowe pobieranie)
    print(f"\n--- 3. Zapytanie: '{query}' ---")
    retrieved_docs, retrieved_meta = retrieve_context(collection, query)

    if not retrieved_docs:
        print("Nie znaleziono fragmentów pasujących do podanego zapytania.")
        return

    print("\n--- Znalezione fragmenty ---")
    for doc, meta in zip(retrieved_docs, retrieved_meta):
        page_val = meta.get("page", "?")
        preview = doc.replace("\n", " ")[:120].strip()
        print(f"-> [Strona {page_val}] {preview}...")

    context = "\n\n---\n\n".join(retrieved_docs)

    prompt = f"""Odpowiedz na poniższe pytanie wyłącznie na podstawie dostarczonego kontekstu.
Jeśli odpowiedź nie wynika bezpośrednio z kontekstu, poinformuj, że nie posiadasz takich informacji.
Zawsze na końcu odpowiedzi dodaj numer strony źródłowej w formacie: [Źródło: str. X].

Kontekst:
{context}

Pytanie: {query}
"""

    # 4. Generowanie odpowiedzi przez Ollama
    print("\n--- 4. Generowanie odpowiedzi (qwen2.5:7b) ---")
    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}],
    )

    print("\n=== Odpowiedź modelu ===")
    print(response["message"]["content"])


if __name__ == "__main__":
    main()