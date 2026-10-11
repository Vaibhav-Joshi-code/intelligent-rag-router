'use strict';

const $ = (id) => document.getElementById(id);

let session = crypto.randomUUID();
let pending = null;
let busy = false;
let ready = false;

const names = {
  simple: 'Simple RAG',
  crag: 'Corrective RAG',
  multi_step: 'Multi-step RAG',
};

$('session').textContent = `Session: ${session}`;

function status(id, text, error = false) {
  const element = $(id);
  if (!element) return;

  element.textContent = text;
  element.classList.toggle('error', error);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.headers || {}),
    },
  });

  let result;

  try {
    result = await response.json();
  } catch {
    throw new Error(`Invalid server response [${response.status}]`);
  }

  if (!response.ok) {
    const detail =
      typeof result?.detail === 'string'
        ? result.detail
        : 'Request failed.';

    throw new Error(`${detail} [${response.status}]`);
  }

  return result;
}

function clearPanels() {
  $('messages').replaceChildren();
  $('events').replaceChildren();
  $('sources').replaceChildren();

  $('route').textContent = 'Waiting for a question';
  $('reason').textContent = '';
  $('stats').textContent = '';
  $('trace').textContent = 'No request yet.';

  for (const key of Object.keys(names)) {
    $(`card-${key}`).classList.remove('active');
  }

  pending = null;
  $('retry').hidden = true;
  status('status', '');
}

async function connect() {
  try {
    const info = await api('/api/status');

    clearPanels();

    session = crypto.randomUUID();
    $('session').textContent = `Session: ${session}`;

    $('source').replaceChildren(
      new Option('All indexed documents', '')
    );

    for (const source of info.corpus?.sources || []) {
      $('source').add(new Option(source, source));
    }

    ready = true;
    $('send').disabled = false;

    status(
      'connection',
      info.corpus
        ? `Connected · ${info.corpus.chunk_count} indexed chunks`
        : 'Connected · run ingestion before policy questions'
    );
  } catch (error) {
    ready = false;
    $('send').disabled = true;

    status(
      'connection',
      error instanceof Error ? error.message : 'Connection failed.',
      true
    );
  }
}

function message(role, text) {
  const node = document.createElement('div');
  const label = document.createElement('strong');

  node.className = `message ${role}`;
  label.textContent = role;

  node.append(
    label,
    document.createTextNode(text ?? '')
  );

  $('messages').append(node);
  $('messages').scrollTop = $('messages').scrollHeight;
}

function show(result) {
  const finalStrategy = result.final_strategy;
  const initialStrategy = result.initial_strategy;

  for (const key of Object.keys(names)) {
    $(`card-${key}`).classList.toggle(
      'active',
      key === finalStrategy && result.status !== 'clarification_needed'
    );
  }

  if (initialStrategy === finalStrategy) {
    $('route').textContent =
      names[finalStrategy] || finalStrategy || 'Unknown strategy';
  } else {
    $('route').textContent =
      `${names[initialStrategy] || initialStrategy || 'Unknown'} → ` +
      `${names[finalStrategy] || finalStrategy || 'Unknown'}`;
  }

  $('reason').textContent = result.route_reason ?? '';

  $('stats').textContent =
    `${String(result.status || 'unknown').replaceAll('_', ' ')} · ` +
    `${result.retrieval_rounds ?? 0} searches · ` +
    `${result.correction_attempts ?? 0} corrections · ` +
    `${result.latency_ms ?? 0} ms`;

  $('events').replaceChildren();

  for (const event of result.events || []) {
    const item = document.createElement('li');
    const title = document.createElement('b');

    title.textContent =
      `${String(event.stage || 'event').replaceAll('_', ' ')}: `;

    item.append(
      title,
      document.createTextNode(event.reason ?? '')
    );

    $('events').append(item);
  }

  $('sources').replaceChildren();

  for (const source of result.sources || []) {
    const item = document.createElement('details');
    const title = document.createElement('summary');
    const text = document.createElement('p');

    title.textContent =
      `[${source.id ?? '?'}] ${source.source ?? 'Unknown source'}`;

    text.textContent = source.excerpt ?? '';

    item.append(title, text);
    $('sources').append(item);
  }

  $('trace').textContent = JSON.stringify(
    {
      standalone_question: result.standalone_question,
      subquestions: result.subquestions,
      trace_id: result.trace_id,
      request_id: result.request_id,
      usage: result.usage,
      events: result.events,
    },
    null,
    2
  );
}

async function send(retry = false) {
  if (busy || !ready) return;

  if (!retry) {
    const question = $('question').value.trim();

    if (!question) return;

    pending = {
      question,
      session_id: session,
      request_id: crypto.randomUUID(),
      source: $('source').value || null,
    };

    message('user', question);

    $('question').value = '';
  }

  if (!pending) return;

  busy = true;

  $('send').disabled = true;
  $('new').disabled = true;
  $('retry').hidden = true;

  status(
    'status',
    'Choosing a strategy and checking the evidence…'
  );

  try {
    const result = await api('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(pending),
    });

    message('assistant', result.answer ?? '');
    show(result);

    pending = null;
    status('status', 'Complete');
  } catch (error) {
    status(
      'status',
      error instanceof Error ? error.message : 'Request failed.',
      true
    );

    $('retry').hidden = false;
  } finally {
    busy = false;
    $('send').disabled = !ready;
    $('new').disabled = false;
  }
}

$('chat-form').addEventListener('submit', (event) => {
  event.preventDefault();
  send();
});

$('retry').addEventListener('click', () => {
  send(true);
});

$('new').addEventListener('click', () => {
  session = crypto.randomUUID();
  $('session').textContent = `Session: ${session}`;

  clearPanels();

  status(
    'status',
    'Started a new session. Older conversations remain stored until cleared or expired.'
  );
});

document
  .querySelectorAll('.examples button')
  .forEach((button) => {
    button.addEventListener('click', () => {
      $('question').value = button.textContent.trim();
      $('question').focus();
    });
  });

$('send').disabled = true;

connect();
