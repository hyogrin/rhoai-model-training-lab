#!/usr/bin/env bash
# build_index.sh — Build ChromaDB vector index and BM25 index from bundle
# Usage: bash scripts/build_index.sh --bundle data/prepared/tau-knowledge-v1
set -euo pipefail

# ─── Defaults ──────────────────────────────────────────────────────────────────
BUNDLE_PATH=""
RAG_CONFIG="configs/rag.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ─── Colors ────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# ─── Usage ─────────────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") --bundle PATH [OPTIONS]

Build ChromaDB vector index and BM25 index from prepared bundle KB documents.

Required:
  --bundle PATH       Path to the prepared dataset bundle

Options:
  --config PATH       RAG config file (default: configs/rag.yaml)
  -h, --help          Show this help

EOF
    exit "${1:-0}"
}

# ─── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bundle)
            BUNDLE_PATH="$2"; shift 2 ;;
        --config)
            RAG_CONFIG="$2"; shift 2 ;;
        -h|--help)
            usage 0 ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            usage 1 ;;
    esac
done

if [[ -z "${BUNDLE_PATH}" ]]; then
    echo -e "${RED}Error: --bundle is required.${NC}" >&2
    usage 1
fi

# Resolve paths
BUNDLE_PATH="$(cd "${PROJECT_ROOT}" && realpath "${BUNDLE_PATH}" 2>/dev/null || echo "${PROJECT_ROOT}/${BUNDLE_PATH}")"

# ─── Validate bundle ──────────────────────────────────────────────────────────
echo -e "${CYAN}Validating bundle...${NC}"

if [[ ! -d "${BUNDLE_PATH}" ]]; then
    echo -e "${RED}Error: Bundle directory not found: ${BUNDLE_PATH}${NC}" >&2
    exit 1
fi

KB_DOCUMENTS="${BUNDLE_PATH}/kb/documents.jsonl"
if [[ ! -f "${KB_DOCUMENTS}" ]]; then
    echo -e "${RED}Error: KB documents not found: ${KB_DOCUMENTS}${NC}" >&2
    exit 1
fi

KB_DOC_COUNT=$(grep -c '.' "${KB_DOCUMENTS}" 2>/dev/null || echo "0")
echo -e "  ✅  Found ${KB_DOC_COUNT} KB documents"

EXAMPLES_FILE="${BUNDLE_PATH}/examples/train_examples.jsonl"
EXAMPLES_COUNT=0
if [[ -f "${EXAMPLES_FILE}" ]]; then
    EXAMPLES_COUNT=$(grep -c '.' "${EXAMPLES_FILE}" 2>/dev/null || echo "0")
    echo -e "  ✅  Found ${EXAMPLES_COUNT} training examples"
else
    echo -e "  ⚠️  No training examples file found (optional)"
fi

# ─── Load .env ─────────────────────────────────────────────────────────────────
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PROJECT_ROOT}/.env"
    set +a
fi

# ─── Build indexes ─────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}Building indexes...${NC}"

cd "${PROJECT_ROOT}"

python3 -c "
import json
import sys
from pathlib import Path

# Load config
from rhoai_model_training_lab.config import load_env, load_yaml_config
load_env()

try:
    rag_config = load_yaml_config('${RAG_CONFIG}')
except FileNotFoundError:
    print('Warning: RAG config not found, using defaults', file=sys.stderr)
    rag_config = {}

retrieval_cfg = rag_config.get('retrieval', {})
vector_cfg = retrieval_cfg.get('vector_store', {})
kb_cfg = retrieval_cfg.get('kb_index', {})
example_cfg = retrieval_cfg.get('example_index', {})
embedding_cfg = retrieval_cfg.get('embedding', {})
bm25_cfg = retrieval_cfg.get('bm25', {})

persist_dir = vector_cfg.get('persist_directory', 'data/indexes/chromadb')
kb_collection = kb_cfg.get('collection_name', 'kb_documents')
chunk_size = kb_cfg.get('chunk_size', 512)
chunk_overlap = kb_cfg.get('chunk_overlap', 64)
embedding_model = embedding_cfg.get('model_id', 'sentence-transformers/all-MiniLM-L6-v2')

bundle_path = Path('${BUNDLE_PATH}')
kb_path = bundle_path / 'kb' / 'documents.jsonl'
examples_path = bundle_path / 'examples' / 'train_examples.jsonl'

# Load KB documents
print(f'Loading KB documents from {kb_path}...')
documents = []
with open(kb_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
            text = doc.get('text', '')
            doc_id = doc.get('doc_id', f'doc-{len(documents)}')
            if text:
                documents.append({'doc_id': doc_id, 'text': text, 'metadata': doc})
        except json.JSONDecodeError:
            pass

print(f'Loaded {len(documents)} documents')

# Chunk documents
print(f'Chunking (size={chunk_size}, overlap={chunk_overlap})...')
chunks = []
for doc in documents:
    text = doc['text']
    doc_id = doc['doc_id']
    start = 0
    chunk_idx = 0
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunks.append({
            'chunk_id': f'{doc_id}_chunk_{chunk_idx}',
            'doc_id': doc_id,
            'text': chunk_text,
        })
        start += chunk_size - chunk_overlap
        chunk_idx += 1

print(f'Created {len(chunks)} chunks')

# Build ChromaDB index
print(f'Building ChromaDB index ({kb_collection})...')
try:
    import chromadb
    from chromadb.config import Settings

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))

    # Delete existing collection if it exists
    try:
        client.delete_collection(kb_collection)
    except Exception:
        pass

    collection = client.create_collection(
        name=kb_collection,
        metadata={'hnsw:space': 'cosine'},
    )

    if chunks:
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            collection.add(
                ids=[c['chunk_id'] for c in batch],
                documents=[c['text'] for c in batch],
                metadatas=[{'doc_id': c['doc_id']} for c in batch],
            )
        print(f'ChromaDB: indexed {collection.count()} chunks into {kb_collection}')
    else:
        print('ChromaDB: no chunks to index')

    # Build examples collection if available
    if examples_path.exists():
        example_collection_name = example_cfg.get('collection_name', 'training_examples')
        try:
            client.delete_collection(example_collection_name)
        except Exception:
            pass

        examples = []
        with open(examples_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

        if examples:
            ex_collection = client.create_collection(
                name=example_collection_name,
                metadata={'hnsw:space': 'cosine'},
            )
            ex_texts = [json.dumps(e.get('messages', [])[:2]) for e in examples]
            ex_ids = [e.get('sample_id', f'ex-{i}') for i, e in enumerate(examples)]
            for i in range(0, len(examples), batch_size):
                ex_collection.add(
                    ids=ex_ids[i:i + batch_size],
                    documents=ex_texts[i:i + batch_size],
                )
            print(f'ChromaDB: indexed {ex_collection.count()} examples into {example_collection_name}')

except ImportError:
    print('ERROR: chromadb not installed. Install with: pip install -e \".[backend]\"', file=sys.stderr)
    sys.exit(1)
except Exception as exc:
    print(f'ERROR: ChromaDB indexing failed: {exc}', file=sys.stderr)
    sys.exit(1)

# Build BM25 index
if bm25_cfg.get('enabled', True):
    print('Building BM25 index...')
    try:
        import pickle
        from rank_bm25 import BM25Okapi

        tokenized = [c['text'].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)

        bm25_dir = Path(persist_dir).parent / 'bm25'
        bm25_dir.mkdir(parents=True, exist_ok=True)
        bm25_path = bm25_dir / 'kb_index.pkl'

        with open(bm25_path, 'wb') as f:
            pickle.dump({'bm25': bm25, 'chunks': chunks}, f)

        print(f'BM25: indexed {len(chunks)} chunks → {bm25_path}')
    except ImportError:
        print('WARNING: rank_bm25 not installed — BM25 index skipped', file=sys.stderr)
    except Exception as exc:
        print(f'WARNING: BM25 indexing failed: {exc}', file=sys.stderr)
else:
    print('BM25: disabled in config')

print()
print('Index build complete.')
"

echo ""
echo -e "${CYAN}═══ Index Build Summary ═══${NC}"
echo -e "  Bundle:     ${BUNDLE_PATH}"
echo -e "  KB docs:    ${KB_DOC_COUNT}"
echo -e "  Examples:   ${EXAMPLES_COUNT}"
echo -e "${GREEN}✅  Index build complete${NC}"
