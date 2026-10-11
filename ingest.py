"""Explicit ingestion: creates/validates Pinecone index and publishes a new namespace."""
import argparse
import json

from rag_app.config import ROOT, Settings, embedding_signature, models
from rag_app.pinecone_store import PineconeStore, load_chunks
from rag_app.store import Store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs', default=str(ROOT / 'docs'))
    args = parser.parse_args()
    settings = Settings.from_env()
    chunks = load_chunks(args.docs)
    _, embeddings = models()
    vectors = PineconeStore(settings, embeddings, embedding_signature())
    try:
        result = vectors.ingest(chunks, Store(settings.database_path))
        print(json.dumps({key: result[key] for key in ('index_name', 'namespace', 'chunk_count', 'sources', 'dimension')}, indent=2))
    finally:
        vectors.close()


if __name__ == '__main__':
    main()

