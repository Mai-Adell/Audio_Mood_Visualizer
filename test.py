import json
from pathlib import Path

# If chunker.py is in the same folder:
from java_code_chunker import chunker  # or: from java_code_chunker import chunker

def chunk_file(java_path: Path):
    codelines = chunker.get_code_lines(java_path)
    tree = chunker.parse_code(str(java_path), codelines)

    # Collect chunks from all member types
    chunks = []
    chunks += chunker.chunk_methods(tree, codelines)
    chunks += chunker.chunk_fields(tree, codelines)
    chunks += chunker.chunk_constructors(tree, codelines)
    # constants: for enums
    try:
        chunks += chunker.chunk_constants(tree)
    except chunker.ChunkingError:
        pass

    return {
        "file": str(java_path),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

def main(root: str):
    root_path = Path(root).expanduser().resolve()
    files = chunker.get_file_list(str(root_path), "*.java")
    results = []
    for f in files:
        try:
            results.append(chunk_file(f))
        except chunker.ParseError as e:
            print(f"[SKIP] Parse failed: {f} -> {e}")
        except Exception as e:
            print(f"[SKIP] Error: {f} -> {e}")

    # Pretty print to console
    for item in results:
        print(f"\n=== {item['file']} ({item['chunk_count']} chunks) ===")
        for i, c in enumerate(item["chunks"], 1):
            print(f"\n--- Chunk {i} ---")
            meta = f"{c['package']} | {c['type']} {c['typename']} | {c['member']} {c['membername']}"
            print(meta)
            print(c["code"])

    # Also dump JSON if you want to index later
    out = Path("chunks.json")
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved JSON -> {out.resolve()}")

if __name__ == "__main__":
    # change this to your project root
    main(r"Examples\commons-io-master")
