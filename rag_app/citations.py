import re


def passages(rows):
    return [{'id': f'S{i}', 'source': row['source'], 'text': row['text']} for i, row in enumerate(rows, 1)]


def render(draft, rows):
    mapping = {f'S{i}': row for i, row in enumerate(rows, 1)}
    def resolve(value):
        raw = value.strip().strip('[]').strip()
        if re.fullmatch(r'[sS]?\s*\d+', raw):
            label = 'S' + str(int(re.sub('[sS ]', '', raw)))
            return label if label in mapping else None
        candidates = [label for label, row in mapping.items() if raw in {row['id'], row['source']}]
        return candidates[0] if len(candidates) == 1 else None

    chosen, inline, errors = set(), set(), []
    for reference in draft['citations']:
        label = resolve(reference)
        if label is None:
            errors.append('Unknown or ambiguous citation.')
        else:
            chosen.add(label)

    def normalize(match):
        body = match.group(1).strip()
        if re.fullmatch(r'[sS]\s*\d+(?:\s*[,;]\s*[sS]?\s*\d+)*', body):
            values = re.split(r'\s*[,;]\s*', body)
        elif '.md' in body or '.txt' in body:
            values = [body]
        else:
            return match.group(0)
        labels = []
        for value in values:
            label = resolve(value)
            if label is None:
                errors.append('Unknown or ambiguous inline citation.')
            else:
                labels.append(f'[{label}]')
                inline.add(label)
        return ' '.join(labels)

    text = re.sub(r'\[([^\[\]\n]+)\]', normalize, draft['answer']).strip()
    if not draft['citations']:
        chosen = inline.copy()
    if inline - chosen or not chosen:
        errors.append('Missing or conflicting source references.')
    if not re.sub(r'\[S\d+\]', '', text).strip(' \n\t.,;:'):
        errors.append('Answer text is empty.')
    if errors:
        return {'answer': '', 'sources': [], 'issue': ' '.join(errors)}
    missing = [label for label in mapping if label in chosen - inline]
    if missing:
        text += '\n\nSources: ' + ' '.join(f'[{label}]' for label in missing)
    return {'answer': text, 'issue': '', 'sources': [
        {'id': label, 'source': row['source'], 'chunk_id': row['id'], 'excerpt': row['text']}
        for label, row in mapping.items() if label in chosen]}
