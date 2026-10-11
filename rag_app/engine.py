"""Router + three distinct LangGraph strategy subgraphs.

Simple = one retrieve/grade/write/check pass, with parent-level CRAG escalation.
CRAG = feedback-driven query rewrite and bounded re-retrieval.
Multi-step = decompose, retrieve/grade each part, then synthesize and verify.
"""
import operator
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langsmith import get_current_run_tree, traceable

from rag_app.citations import passages, render
from rag_app.schemas import AnswerCheck, DocumentGrade, Draft, Plan, Rewrite, RouteDecision


class State(TypedDict, total=False):
    question: str
    history: list
    source: str | None
    goal: str
    current_goal: str
    query: str
    initial_strategy: str
    final_strategy: str
    route_reason: str
    clarification: str
    snapshot: dict
    documents: list
    evidence: list
    sufficient: bool
    feedback: str
    draft: dict
    validated: bool
    answer: str
    sources: list
    status: str
    retrieval_rounds: int
    correction_attempts: int
    subquestions: list
    step_index: int
    step_repairs: int
    step_query: str
    collected: list
    gaps: list
    synthesis_attempts: int
    events: Annotated[list, operator.add]


def event(stage, reason, **details):
    return {'stage': stage, 'reason': reason, **details}


class Engine:
    def __init__(self, llm, retriever, settings, snapshot):
        self.llm, self.retriever, self.settings, self.read_snapshot = llm, retriever, settings, snapshot
        self.simple = self._simple_graph()
        self.crag = self._crag_graph()
        self.multi = self._multi_graph()
        root = StateGraph(State)
        root.add_node('router_agent', self.route)
        root.add_node('simple_rag', self.run_simple)
        root.add_node('corrective_rag', self.run_crag)
        root.add_node('multi_step_rag', self.run_multi)
        root.add_node('clarify', self.clarify)
        root.add_node('final_response', lambda s: {})
        root.add_edge(START, 'router_agent')
        root.add_conditional_edges('router_agent', lambda s: 'clarify' if s['clarification'] else s['initial_strategy'],
            {'simple': 'simple_rag', 'crag': 'corrective_rag', 'multi_step': 'multi_step_rag', 'clarify': 'clarify'})
        root.add_conditional_edges('simple_rag', lambda s: 'finish' if s['validated'] else 'correct',
            {'finish': 'final_response', 'correct': 'corrective_rag'})
        for node in ('corrective_rag', 'multi_step_rag', 'clarify'):
            root.add_edge(node, 'final_response')
        root.add_edge('final_response', END)
        self.graph = root.compile(name='IntelligentRAGRouter')

    def _simple_graph(self):
        graph = StateGraph(State)
        for name, function in [('retrieve', self.retrieve), ('grade', self.grade), ('write', self.write), ('check', self.check)]:
            graph.add_node(name, function)
        graph.add_edge(START, 'retrieve')
        graph.add_edge('retrieve', 'grade')
        graph.add_conditional_edges('grade', lambda s: 'write' if s['sufficient'] else 'escalate', {'write': 'write', 'escalate': END})
        graph.add_edge('write', 'check')
        graph.add_edge('check', END)
        return graph.compile(name='SimpleRAG')

    def _crag_graph(self):
        graph = StateGraph(State)
        for name, function in [('rewrite_query', self.correct), ('retrieve_again', self.retrieve),
            ('grade_documents', self.grade), ('write_answer', self.write), ('validate_answer', self.check), ('abstain', self.abstain)]:
            graph.add_node(name, function)
        graph.add_edge(START, 'rewrite_query')
        graph.add_edge('rewrite_query', 'retrieve_again')
        graph.add_edge('retrieve_again', 'grade_documents')
        graph.add_conditional_edges('grade_documents', lambda s: 'write' if s['sufficient'] else self.crag_next(s),
            {'write': 'write_answer', 'retry': 'rewrite_query', 'stop': 'abstain'})
        graph.add_edge('write_answer', 'validate_answer')
        graph.add_conditional_edges('validate_answer', lambda s: 'finish' if s['validated'] else self.crag_next(s),
            {'finish': END, 'retry': 'rewrite_query', 'stop': 'abstain'})
        graph.add_edge('abstain', END)
        return graph.compile(name='CorrectiveRAG')

    def _multi_graph(self):
        graph = StateGraph(State)
        for name, function in [('plan_subquestions', self.plan), ('retrieve_step', self.retrieve_step),
            ('grade_step', self.grade), ('repair_step', self.repair_step), ('collect_step', self.collect_step),
            ('synthesize', self.synthesize), ('validate_synthesis', self.check), ('abstain', self.abstain)]:
            graph.add_node(name, function)
        graph.add_edge(START, 'plan_subquestions')
        graph.add_edge('plan_subquestions', 'retrieve_step')
        graph.add_edge('retrieve_step', 'grade_step')
        graph.add_conditional_edges('grade_step', lambda s: 'collect' if s['sufficient'] or s['step_repairs'] >= 1 else 'repair',
            {'collect': 'collect_step', 'repair': 'repair_step'})
        graph.add_edge('repair_step', 'retrieve_step')
        graph.add_conditional_edges('collect_step', self.multi_next,
            {'next': 'retrieve_step', 'write': 'synthesize', 'stop': 'abstain'})
        graph.add_edge('synthesize', 'validate_synthesis')
        graph.add_conditional_edges('validate_synthesis',
            lambda s: 'finish' if s['validated'] else ('repair' if s['synthesis_attempts'] < 2 else 'stop'),
            {'finish': END, 'repair': 'synthesize', 'stop': 'abstain'})
        graph.add_edge('abstain', END)
        return graph.compile(name='MultiStepRAG')

    def route(self, state):
        decision = self.llm.call(RouteDecision, 'router', {'question': state['question'], 'history': state['history']})
        return {'initial_strategy': decision.strategy, 'final_strategy': decision.strategy,
            'goal': decision.standalone_question, 'current_goal': decision.standalone_question,
            'query': decision.standalone_question, 'route_reason': decision.reason, 'clarification': decision.clarification,
            'events': [event('router', decision.reason, strategy=decision.strategy)]}

    def run_simple(self, state, config: RunnableConfig):
        return self.simple.invoke({**state, 'events': []}, config=config)

    def run_crag(self, state, config: RunnableConfig):
        reason = state.get('feedback') or state['route_reason']
        result = self.crag.invoke({**state, 'final_strategy': 'crag', 'validated': False, 'events': []}, config=config)
        result['events'] = [event('strategy_switch' if state['initial_strategy'] == 'simple' else 'strategy_start',
                                  reason, strategy='crag'), *result['events']]
        return result

    def run_multi(self, state, config: RunnableConfig):
        return self.multi.invoke({**state, 'events': []}, config=config)

    def clarify(self, state):
        return {'answer': state['clarification'], 'sources': [], 'status': 'clarification_needed',
                'events': [event('clarify', 'No retrieval was performed; the request needs clarification.')]}

    def retrieve(self, state):
        snapshot = state.get('snapshot') or self.read_snapshot()
        rows = self.retriever.search(state['query'], snapshot, state.get('source'))
        return {'snapshot': snapshot, 'documents': rows, 'evidence': [], 'validated': False,
            'retrieval_rounds': state['retrieval_rounds'] + 1,
            'events': [event('retrieve', 'Searched the pinned Pinecone corpus.', query=state['query'],
                              sources=list(dict.fromkeys(row['source'] for row in rows)), chunks=len(rows))]}

    def grade(self, state):
        if not state['documents']:
            return {'evidence': [], 'sufficient': False, 'feedback': 'No matching evidence was retrieved.',
                    'events': [event('grade', 'No matching evidence was retrieved.', sufficient=False)]}
        result = self.llm.call(DocumentGrade, 'grade', {'goal': state['current_goal'], 'passages': passages(state['documents'])})
        mapping = {f'S{i}': row for i, row in enumerate(state['documents'], 1)}
        valid = all(label in mapping for label in result.relevant_ids)
        evidence = [mapping[label] for label in dict.fromkeys(result.relevant_ids)] if valid else []
        sufficient = result.sufficient and bool(evidence)
        reason = result.reason if valid else 'Document grader selected an unknown source ID.'
        return {'evidence': evidence, 'sufficient': sufficient, 'feedback': reason,
                'events': [event('grade', reason, sufficient=sufficient)]}

    def write(self, state):
        result = self.llm.call(Draft, 'write', {'original_goal': state['goal'], 'passages': passages(state['evidence']),
            'previous_validation_feedback': state.get('feedback', '')})
        return {'draft': result.model_dump(), 'events': [event('write', 'Generated a draft from selected evidence.')]}

    def check(self, state):
        draft = state['draft']
        if not draft['answerable']:
            return {'validated': False, 'feedback': 'The writer could not answer from the available evidence.',
                    'events': [event('check', 'Writer reported insufficient evidence.', passed=False)]}
        rendered = render(draft, state['evidence'])
        if rendered['issue']:
            return {'validated': False, 'feedback': rendered['issue'],
                    'events': [event('check', rendered['issue'], passed=False)]}
        result = self.llm.call(AnswerCheck, 'check', {'original_goal': state['goal'],
            'draft': rendered['answer'], 'cited_passages': rendered['sources']})
        valid = result.grounded and result.complete
        return {'validated': valid, 'feedback': result.reason, 'answer': rendered['answer'] if valid else '',
            'sources': rendered['sources'] if valid else [], 'status': 'answered' if valid else 'insufficient_evidence',
            'events': [event('check', result.reason, passed=valid, grounded=result.grounded, complete=result.complete)]}

    def correct(self, state):
        result = self.llm.call(Rewrite, 'rewrite', {'original_goal': state['goal'], 'previous_query': state['query'],
            'failure': state.get('feedback') or state['route_reason']})
        return {'query': result.query, 'correction_attempts': state['correction_attempts'] + 1,
                'events': [event('crag_rewrite', 'Reformulated the search using failure feedback.', query=result.query)]}

    def crag_next(self, state):
        return 'retry' if state['correction_attempts'] < self.settings.crag_attempts else 'stop'

    def plan(self, state):
        result = self.llm.call(Plan, 'plan', {'goal': state['goal'], 'maximum_subquestions': self.settings.multi_steps})
        if len(result.subquestions) > self.settings.multi_steps:
            raise ValueError('The planner exceeded the configured step budget.')
        return {'subquestions': result.subquestions, 'step_index': 0, 'step_repairs': 0, 'step_query': '',
            'collected': [], 'gaps': [], 'synthesis_attempts': 0,
            'events': [event('plan', 'Split the request into distinct evidence needs.', subquestions=result.subquestions)]}

    def retrieve_step(self, state):
        goal = state['subquestions'][state['step_index']]
        query = state.get('step_query') or goal
        result = self.retrieve({**state, 'query': query})
        return {**result, 'current_goal': goal, 'query': query}

    def repair_step(self, state):
        result = self.llm.call(Rewrite, 'rewrite', {'original_goal': state['current_goal'],
            'previous_query': state['query'], 'failure': state['feedback']})
        return {'step_query': result.query, 'step_repairs': state['step_repairs'] + 1,
            'correction_attempts': state['correction_attempts'] + 1,
            'events': [event('step_rewrite', 'Corrected this subquestion once before continuing.', query=result.query)]}

    def collect_step(self, state):
        combined = {row['id']: row for row in state['collected']}
        gaps = list(state['gaps'])
        if state['sufficient']:
            combined.update({row['id']: row for row in state['evidence']})
        else:
            gaps.append(state['current_goal'])
        return {'collected': list(combined.values()), 'gaps': gaps, 'step_index': state['step_index'] + 1,
            'step_repairs': 0, 'step_query': '',
            'events': [event('collect_step', 'Stored this step’s evidence; unresolved gaps remain explicit.',
                             question=state['current_goal'], sufficient=state['sufficient'])]}

    def multi_next(self, state):
        if state['step_index'] < len(state['subquestions']):
            return 'next'
        return 'stop' if state['gaps'] else 'write'

    def synthesize(self, state):
        result = self.write({**state, 'evidence': state['collected']})
        return {**result, 'evidence': state['collected'], 'synthesis_attempts': state['synthesis_attempts'] + 1}

    def abstain(self, state):
        reason = ('Missing evidence for: ' + '; '.join(state['gaps'])) if state.get('gaps') else state.get('feedback', 'Evidence gaps remain after the retry budget.')
        return {'answer': 'I could not verify a complete answer from the indexed ChaiTech documents. '
                'Please clarify the question or add the missing policy.', 'sources': [], 'status': 'insufficient_evidence',
                'events': [event('abstain', reason)]}

    @traceable(name='rag_strategy_request', run_type='chain')
    def run(self, question, history=None, source=None, config=None):
        result = self.graph.invoke({'question': question, 'history': history or [], 'source': source,
            'snapshot': {}, 'events': [], 'retrieval_rounds': 0, 'correction_attempts': 0,
            'validated': False, 'sources': [], 'subquestions': []}, config={**(config or {}), 'recursion_limit': 100})
        run = get_current_run_tree()
        return {key: result[key] for key in ('answer', 'status', 'initial_strategy', 'final_strategy', 'route_reason',
                    'sources', 'events', 'subquestions', 'retrieval_rounds', 'correction_attempts')} | {
            'standalone_question': result['goal'], 'trace_id': str(run.trace_id) if run else None}
