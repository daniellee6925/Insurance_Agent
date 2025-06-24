import re
import json

# === Load the TXT file ===
with open("docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

# === Regex to extract each Q&A block ===
pattern = r"(###\s?.+?\n(?:[^\n#].*?\n)+)"
chunks = re.findall(pattern, text)

# === Print each chunk ===
for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---\n{chunk.strip()}")

# === Optional: Save as JSON ===
chunk_data = [{"text": chunk.strip()} for chunk in chunks]

with open("chunked_output.json", "w", encoding="utf-8") as f:
    json.dump(chunk_data, f, indent=2)
