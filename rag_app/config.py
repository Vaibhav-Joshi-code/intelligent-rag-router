import hashlib
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / '.env')
os.environ.setdefault('LANGSMITH_PROJECT', 'chaitech-rag-strategies')
os.environ.setdefault('LANGSMITH_HIDE_INPUTS', 'true')
os.environ.setdefault('LANGSMITH_HIDE_OUTPUTS', 'true')


def required(name):
    value = os.getenv(name, '').strip()
    if not value:
        raise ValueError(f'Configure {name} in .env.')
    return value


@dataclass(frozen=True)
class Settings:
    index_name: str = 'chaitech-rag-strategies'
    namespace_prefix: str = 'chaitech-strategies'
    database_path: str = str(ROOT / 'data/strategies.db')
    top_k: int = 4
    crag_attempts: int = 2
    multi_steps: int = 4
    history_turns: int = 6

    @classmethod
    def from_env(cls):
        value = cls(index_name=os.getenv('PINECONE_INDEX_NAME', cls.index_name),
            namespace_prefix=os.getenv('PINECONE_NAMESPACE_PREFIX', cls.namespace_prefix),
            database_path=str(ROOT / os.getenv('DATABASE_PATH', 'data/strategies.db')),
            top_k=int(os.getenv('TOP_K', '4')), crag_attempts=int(os.getenv('CRAG_MAX_ATTEMPTS', '2')),
            multi_steps=int(os.getenv('MULTI_MAX_STEPS', '4')), history_turns=int(os.getenv('HISTORY_TURNS', '6')))
        if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,43}[a-z0-9]', value.index_name):
            raise ValueError('Use a Pinecone index name of 2–45 lowercase letters, numbers, or hyphens.')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,50}', value.namespace_prefix):
            raise ValueError('Use a namespace prefix of at most 50 safe identifier characters.')
        if not (1 <= value.top_k <= 8 and 1 <= value.crag_attempts <= 3 and 2 <= value.multi_steps <= 4
                and 1 <= value.history_turns <= 12):
            raise ValueError('Limits: TOP_K 1–8, CRAG_MAX_ATTEMPTS 1–3, MULTI_MAX_STEPS 2–4, HISTORY_TURNS 1–12.')
        return value


def embedding_signature():
    identity = required('EURI_BASE_URL').rstrip('/') + '/' + required('EURI_CHAT_MODEL')
    return hashlib.sha256(identity.encode()).hexdigest()


@lru_cache
def models():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    base_url = required("EURI_BASE_URL").rstrip("/")
    chat_model = required("EURI_CHAT_MODEL")
    embedding_model = required("EURI_EMBEDDING_MODEL")
    api_key = required("EURI_API_KEY")

    model = ChatOpenAI(
        model=chat_model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        timeout=45,
        max_retries=1,
    )

    embeddings = OpenAIEmbeddings(
        model=embedding_model,
        api_key=api_key,
        base_url=base_url,
        timeout=45,
        max_retries=1,
    )

    return model, embeddings