

"""Pinecone dense-vector retrieval and explicit, versioned corpus ingestion."""  
import hashlib  
import math  
import os  
import threading  
import time  
import uuid  
from pathlib import Path  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langsmith import traceable  
from rag_app.config import required  
from pinecone import Pinecone, ServerlessSpec
import hashlib

def _chunk_id(text, source=''):
    h = hashlib.sha256(f'{source}::{text}'.encode('utf-8')).hexdigest()
    return h[:32]

class DependencyError(Exception):  
    pass  
  
def as_dict(value):  
    return value if isinstance(value, dict) else value.__dict__  
  
def load_chunks(directory):  
    root = Path(directory).resolve()  
    if not root.is_dir():  
        raise ValueError('The document directory does not exist.')  
    splitter = RecursiveCharacterTextSplitter(chunk_size=850, chunk_overlap=150)  
    chunks = []  
    for path in sorted(root.rglob('*')):  
        if not path.is_file() or path.suffix.lower() not in {'.md', '.txt'}:  
            continue  
        if not path.resolve().is_relative_to(root):  
            raise ValueError('Documents must stay inside the selected directory.')  
        if path.stat().st_size > 2_000_000:  
            raise ValueError('Each input file must be at most 2 MB.')  
        source = path.relative_to(root).as_posix()  
        text = path.read_text(encoding='utf-8-sig')  
        for number, piece in enumerate(splitter.split_text(text)):  
            if not piece.strip():  
                continue  
            chunk_id = hashlib.sha256(f'{source}:{number}:{piece}'.encode()).hexdigest()  
            chunks.append({'id': chunk_id, 'source': source, 'text': piece, 'chunk_number': number})  
            if len(chunks) > 1000:  
                raise ValueError('This teaching application supports at most 1,000 chunks per corpus.')  
    if not chunks:  
        raise ValueError('No nonempty UTF-8 .md or .txt documents were found.')  
    return chunks  
  
def validate_vectors(vectors, expected_count, dimension=None):  
    if len(vectors) != expected_count or not vectors:  
        raise ValueError('Embedding count does not match the text batch.')  
    dim = dimension or len(vectors[0])  
    if not dim or any(len(v) != dim or not all(math.isfinite(x) for x in v) for v in vectors):  
        raise ValueError('Embedding vectors have inconsistent dimensions or non-finite values.')  
    return dim  
  
class PineconeStore:  
    def __init__(self, settings, embeddings, signature, client=None):  
        self.settings, self.embeddings, self.signature = settings, embeddings, signature  
        self.client = client  
        self.guard, self.indexes = threading.Lock(), {}  
  

    def control(self):  
        with self.guard:  
            if self.client is None:  
                # Create an instance of the Pinecone client using the new API  
                self.client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))  
            return self.client 
  
    def index(self, index_name):  
        control = self.control()  
        with self.guard:  
            if index_name not in self.indexes:  
                self.indexes[index_name] = control.Index(index_name)  
            return self.indexes[index_name]  
  
    def close(self):  
        # Pinecone Python SDK doesn't require explicit close calls for indexes.  
        pass  
  
    def validate_manifest(self, manifest):  
        if (manifest['index_name'] != self.settings.index_name or manifest['signature'] != self.signature  
            or not manifest['namespace'].startswith(self.settings.namespace_prefix + '-')):  
            raise ValueError('Corpus configuration changed. Re-ingest into the configured index before querying.')  
  
    @traceable(name='pinecone_retrieval', run_type='retriever')  
    def search(self, query, manifest, source=None):  
        self.validate_manifest(manifest)  
        vector = self.embeddings.embed_query(query)  
        validate_vectors([vector], 1, manifest['dimension'])  
        metadata_filter = {'corpus_id': {'$eq': manifest['corpus_id']}}  
        if source:  
            metadata_filter = {'$and': [metadata_filter, {'source': {'$eq': source}}]}  
        try:  
            response = self.index(manifest['index_name']).query(vector=vector, top_k=self.settings.top_k,  
                namespace=manifest['namespace'], filter=metadata_filter, include_metadata=True,  
                include_values=False, _request_timeout=30)  
            response = as_dict(response)  
        except Exception as error:  
            raise DependencyError('Pinecone retrieval failed. Check the API key, index, region, and connectivity.') from error  
        rows = []  
        for match in response.get('matches', []):  
            metadata = match.get('metadata', {})  
            if metadata.get('corpus_id') != manifest['corpus_id'] or (source and metadata.get('source') != source):  
                continue  
            if not metadata.get('text') or not metadata.get('source'):  
                continue  
            rows.append({'id': match['id'], 'source': metadata['source'], 'text': metadata['text'],  
                         'score': match.get('score', 0)})  
        return rows  
  
    def ingest(self, chunks, store, visibility_attempts=20, pause=time.sleep):
    # --- 1. Compute a fingerprint of the corpus + embedding signature ---
        fingerprint = hashlib.sha256(
            (self.signature + '||' + '||'.join(
                f"{row['id']}:{row['text']}" for row in chunks
            )).encode('utf-8')
        ).hexdigest()

        # --- 2. Skip entirely if nothing changed (saves embedding quota) ---
        # previous = store.manifest() if hasattr(store, 'manifest') else None
        try:
            previous = store.manifest()
        except ValueError:
            previous = None
        if previous and previous.get('fingerprint') == fingerprint:
            print(f"No changes detected ({len(chunks)} chunks). Skipping ingest.")
            return previous

        probe = self.embeddings.embed_query('embedding dimension probe')
        dimension = validate_vectors([probe], 1)
        control = self.control()
        existing_indexes = [idx.name for idx in control.list_indexes()]

        if self.settings.index_name not in existing_indexes:
            control.create_index(
                name=self.settings.index_name,
                dimension=dimension,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        # Wait until the index is ready before using it
        while not control.describe_index(self.settings.index_name).status['ready']:
            time.sleep(1)
        index = self.index(self.settings.index_name)
        revision = uuid.uuid4().hex
        namespace = self.settings.namespace_prefix + '-' + revision
        for offset in range(0, len(chunks), 64):
            batch = chunks[offset:offset + 64]
            vectors = self.embeddings.embed_documents([row['text'] for row in batch])
            validate_vectors(vectors, len(batch), dimension)
            result = index.upsert(namespace=namespace, vectors=[
                {'id': row['id'], 'values': vector, 'metadata': {'text': row['text'], 'source': row['source'],
                                                                'chunk_number': row['chunk_number'], 'corpus_id': revision}}
                for row, vector in zip(batch, vectors)], _request_timeout=30)
            if result.upserted_count != len(batch):
                raise DependencyError('Incomplete Pinecone batch; the previous corpus remains active.')

        manifest = {'index_name': self.settings.index_name, 'namespace': namespace, 'host': self.settings.index_name,
                    'corpus_id': revision, 'dimension': dimension, 'signature': self.signature,
                    'sources': sorted({row['source'] for row in chunks}), 'chunk_count': len(chunks),
                    'fingerprint': fingerprint, 'created': time.time()}
        store.publish(manifest)

        # --- 3. Delete the previous namespace now that the new one is live ---
        if previous and previous.get('namespace') and previous['namespace'] != namespace:
            try:
                index.delete(namespace=previous['namespace'], delete_all=True)
                print(f"Removed stale namespace: {previous['namespace']}")
            except Exception as exc:
                print(f"Warning: could not delete old namespace {previous['namespace']}: {exc}")

        return manifest