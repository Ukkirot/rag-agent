import chromadb
import ollama

# 1. Połączenie z lokalną bazą Chroma (zapis w folderze ./chroma_db)
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="project_knowledge")

# 2. Przykładowe dane techniczne (symulacja dokumentacji / reguł)
documents = [
    "Endpoint POST /api/v1/auth/token przyjmuje JSON z polami 'client_id' i 'client_secret'. Token wygasa po 3600 sekundach.",
    "Błąd ERR_CONN_404 oznacza brak połączenia z węzłem nadrzędnym. Rozwiązaniem jest restart usługi daemon_sync.",
    "Limit zapytań dla użytkowników darmowych wynosi 60 zapytań na minutę. Przekroczenie limitu zwraca kod HTTP 429 Too Many Requests.",
    "W systemie reguł test sporny rzucany jest dwiema kośćmi k10. Wynik powyżej 15 oznacza natychmiastowy sukces krytyczny."
]

print("[1/3] Generowanie embeddingów i zapis do ChromaDB...")
for i, doc in enumerate(documents):
    # Generujemy wektor za pomocą nomic-embed-text
    emb_res = ollama.embeddings(model="nomic-embed-text", prompt=doc)
    embedding = emb_res["embedding"]
    
    collection.upsert(
        ids=[f"doc_{i}"],
        documents=[doc],
        embeddings=[embedding],
        metadatas=[{"source": "technical_manual.md", "id": i}]
    )

print("[2/3] Baza gotowa. Testujemy wyszukiwanie...")
# Pytanie testowe
question = "Co zrobić, gdy wystąpi błąd ERR_CONN_404 i ile wynosi limit dla darmowych kont?"

# Zamieniamy pytanie na wektor i szukamy 2 najbardziej pasujących fragmentów
q_emb = ollama.embeddings(model="nomic-embed-text", prompt=question)["embedding"]
results = collection.query(
    query_embeddings=[q_emb],
    n_results=2
)

retrieved_docs = results["documents"][0]
context = "\n---\n".join(retrieved_docs)

print(f"\n--- ZNALEZIONY KONTEKST ---\n{context}\n---------------------------\n")

print("[3/3] Przekazywanie kontekstu do LLM (Ollama)...")
system_prompt = (
    "Jesteś precyzyjnym asystentem technicznym. "
    "Odpowiadaj WYŁĄCZNIE na podstawie dostarczonego poniżej kontekstu. "
    "Jeśli w kontekście nie ma odpowiedzi, napisz wprost: 'Brak informacji w dokumentacji'.\n\n"
    f"KONTEKST:\n{context}"
)

response = ollama.chat(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
)

print("\n=== ODPOWIEDŹ ASYSTENTA ===")
print(response["message"]["content"])