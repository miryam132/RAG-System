"""Offline regression tests. Real LlamaIndex/Chroma; synthetic embeddings, no Gemini calls."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import config
import ingest
import rag
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, NodeWithScore
from llama_index.core.llms import MockLLM

@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, 'CHROMA_DIR', tmp_path / 'chroma')
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.llm = MockLLM()


def test_documents():
    docs = ingest.load_sections()
    assert len(docs) == 41
    assert len({d.metadata['game'] for d in docs}) == 5
    assert all(d.text.strip() and d.metadata['section'] and d.metadata['source_file'] for d in docs)

@pytest.mark.parametrize('label', ingest.CHUNK_CONFIGS)
def test_real_chunking(label):
    size, overlap = ingest.CHUNK_CONFIGS[label]
    nodes = SentenceSplitter(chunk_size=size, chunk_overlap=overlap).get_nodes_from_documents(ingest.load_sections())
    print(f'CHUNKS {label}: {len(nodes)}')
    assert len(nodes) >= 41
    assert all(n.text.strip() and n.metadata.get('game') and n.metadata.get('section') for n in nodes)
    assert len({(n.metadata['source_file'], n.metadata['section']) for n in nodes}) == 41


def test_persist_reload_and_filter():
    docs = ingest.load_sections()
    nodes = ingest.build_index(docs, 512, 50, config.COLLECTION_NAME + '_c512')
    assert rag.available_labels() == ['c512']
    assert len(rag.list_games('c512')) == 5
    index = rag.load_index('c512')
    results = rag.retrieve(index, 'כמה קלפים מקבלים?', game='אונו', top_k=3)
    assert len(results) == 3
    assert all(r.node.metadata['game'] == 'אונו' for r in results)
    assert rag._client().get_collection(config.COLLECTION_NAME + '_c512').count() == len(nodes)
    assert rag.retrieve(index, 'שאלה', game='משחק שאינו קיים') == []


def test_no_results_no_llm(monkeypatch):
    llm = Mock()
    monkeypatch.setattr(rag, 'Settings', SimpleNamespace(llm=llm))
    assert rag.generate_answer('שאלה', []) == rag.NO_ANSWER
    llm.chat.assert_not_called()


def test_prompt_history_limit_and_sources(monkeypatch):
    llm = Mock()
    llm.chat.return_value.message.content = 'תשובת בדיקה בלבד'
    monkeypatch.setattr(rag, 'Settings', SimpleNamespace(llm=llm))
    source = NodeWithScore(node=TextNode(text='טקסט מקור', metadata={'game':'אונו','section':'הכנה'}), score=.8)
    history = [{'role':'user' if i % 2 == 0 else 'assistant', 'content':str(i)} for i in range(10)]
    rag.generate_answer('שאלה', [source], history)
    messages = llm.chat.call_args.args[0]
    assert len(messages) == 8
    assert [m.content for m in messages[1:-1]] == [str(i) for i in range(4,10)]
    assert 'טקסט מקור' in messages[-1].content
    assert 'אונו' in messages[-1].content


def test_missing_key(monkeypatch):
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='GOOGLE_API_KEY'):
        config.setup_embeddings()


def test_followup_query_keeps_game_context(monkeypatch):
    """Desired behavior: follow-up retrieval must retain the prior game."""
    retrieve = Mock(return_value=[])
    monkeypatch.setattr(rag, 'retrieve', retrieve)
    rag.ask(None, 'וכמה צריך כדי לנצח?', history=[{'role':'user','content':'איך משחקים בקטאן?'}])
    args = retrieve.call_args.args
    assert 'קטאן' in args[1] or args[2] == 'קטאן', f'History was lost before retrieval: {args[1:]}'


def test_failed_rebuild_preserves_old_data(monkeypatch):
    """Desired behavior: an interrupted rebuild should retain the existing index."""
    name = config.COLLECTION_NAME + '_c512'
    ingest.build_index(ingest.load_sections(), 512, 50, name)
    before = rag._client().get_collection(name).count()
    monkeypatch.setattr(ingest, 'VectorStoreIndex', Mock(side_effect=RuntimeError('simulated embedding failure')))
    with pytest.raises(RuntimeError, match='simulated'):
        ingest.build_index(ingest.load_sections(), 512, 50, name)
    after = rag._client().get_collection(name).count()
    assert after == before, f'Rebuild lost existing nodes: before={before}, after={after}'


def test_ui_game_change_clears_history(monkeypatch):
    """Real Streamlit script; backend stubbed so the test makes no paid calls."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(config, 'setup_embeddings', lambda: None)
    monkeypatch.setattr(config, 'setup_llm', lambda: None)
    monkeypatch.setattr(rag, 'available_labels', lambda: ['c512'])
    monkeypatch.setattr(rag, 'list_games', lambda label: ['אונו','קטאן'])
    monkeypatch.setattr(rag, 'load_index', lambda label: object())
    monkeypatch.setattr(rag, 'ask', lambda *a: {'answer':'תשובת בדיקה', 'sources':[]})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'src/app.py')).run()
    assert not app.exception
    app.selectbox[1].select('קטאן').run()
    app.chat_input[0].set_value('איך משחקים בקטאן?').run()
    assert len(app.session_state['messages']) == 2
    app.selectbox[1].select('אונו').run()
    assert not app.exception
    assert app.session_state['messages'] == [], 'Old game history is still present after switching game'
