import faiss
import numpy as np
import openai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

MODEL_NAME = "mistral-7b-instruct"

# Read text from PDF
def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

pdf_text = read_pdf("02_pdc_handouts.pdf")

# Split text into chunks
def chunk_text(text, chunk_size=200):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]

chunks = chunk_text(pdf_text)
print(f"Loaded {len(chunks)} text chunks from PDF")

# Create embeddings locally
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)

# Store embeddings in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

print("Vector database created")

question = "What is Parallel Distributed Computing?"
question_embedding = embedding_model.encode([question])

# Retrieve relevant chunks
k = 3
_, I = index.search(question_embedding, k)
context = "\n\n".join([chunks[i] for i in I[0]])

print("\nRetrieved Context:\n")
print(context)

# RAG prompt
prompt = f"""
You are a helpful assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

# Query local LLM via LM Studio
client = openai.OpenAI(
    api_key="lm-studio",
    base_url="http://localhost:1234/v1"
)

response = client.chat.completions.create(
    model="mistralai/mistral-7b-instruct-v0.3",
    messages=[{"role": "user", "content": prompt}],
    temperature=0
)

print("\nAnswer:\n")
print(response.choices[0].message.content)
