import os
import threading
import time
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import merge_configs
from langchain_openai import ChatOpenAI
from rag_app.config import embedding_signature, models
from rag_app.engine import Engine
from rag_app.llm import Gateway
from rag_app.pinecone_store import PineconeStore


class Service:
    def __init__(self, settings, store, engine=None):
        self.settings, self.store, self.engine = settings, store, engine
        self.lock = threading.Lock()
        self.runnable = RunnableLambda(self.invoke, name='chaitech_strategy_assistant')

    def prepare(self):
        with self.lock:
            if self.engine is None:
                model, embeddings = models()
                retriever = PineconeStore(self.settings, embeddings, embedding_signature())
                self.engine = Engine(Gateway(), retriever, self.settings, self.store.manifest)

    def invoke(self, payload, config):
        with self.store.session_lock(payload['session_id']):
            previous = self.store.cached(payload)
            if previous is not None:
                return previous
            self.prepare()
            usage = UsageMetadataCallbackHandler()
            metadata = {'session_id': payload['session_id'], 'thread_id': payload['session_id'], 'request_id': payload['request_id']}
            started = time.perf_counter()
            result = self.engine.run(payload['question'], self.store.history(payload['session_id'], self.settings.history_turns),
                payload.get('source'), config=merge_configs(config, {'callbacks': [usage], 'metadata': metadata}),
                langsmith_extra={'metadata': metadata})
            result.update(request_id=payload['request_id'], session_id=payload['session_id'],
                          latency_ms=round((time.perf_counter() - started) * 1000), usage=usage.usage_metadata)
            self.store.complete(payload, result)
            return result

    def ask(self, payload):
        result = self.runnable.invoke(payload, config={'metadata': {
            'session_id': payload['session_id'], 'thread_id': payload['session_id'], 'request_id': payload['request_id']}})
        # A completed replay preserves the stored response rather than inventing fresh usage.
        return result
