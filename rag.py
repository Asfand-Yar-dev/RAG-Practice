import faiss
import numpy as np
import openai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ----------------------------
# 1. Connect to LM Studio
# ----------------------------
MODEL_NAME = "mistral-7b-instruct"

# ----------------------------
# 2. Read PDF
# ----------------------------
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

pdf_text = read_pdf("02_pdc_handouts.pdf")

# ----------------------------
# 3. Chunk text
# ----------------------------
def chunk_text(text, chunk_size=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

chunks = chunk_text(pdf_text)

print(f"Loaded {len(chunks)} text chunks from PDF")

# ----------------------------
# 4. Create embeddings (LOCAL)
# ----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)

# ----------------------------
# 5. Store embeddings in FAISS
# ----------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("Vector database created")

# ----------------------------
# 6. Ask a question
# ----------------------------
question = "What is Parallel Distributing Computing?"

question_embedding = embedding_model.encode([question])

# ----------------------------
# 7. Retrieve relevant chunks
# ----------------------------
k = 3
D, I = index.search(question_embedding, k)

retrieved_chunks = [chunks[i] for i in I[0]]
context = "\n\n".join(retrieved_chunks)

print("\nRetrieved Context:\n")
print(context)


prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

client = openai.OpenAI(api_key="lm-studio", base_url="http://localhost:1234/v1")

response = client.chat.completions.create(
    model="mistralai/mistral-7b-instruct-v0.3",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print("\nAnswer:\n")
print(response.choices[0].message.content)