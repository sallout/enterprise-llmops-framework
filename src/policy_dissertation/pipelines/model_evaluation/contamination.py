"""Limited matching against the exact cached compression training inputs."""
import json
from collections import defaultdict, Counter
from .scoring import words


def grams(tokens, n=13):
    return {tuple(tokens[i:i+n]) for i in range(max(0, len(tokens)-n+1))}


def audit(cases, tokens_path, tokenizer, progress):
    targets = {}; index = defaultdict(set)
    for case in cases:
        # Prompt and answer checked separately; common instruction wrappers omitted.
        for kind, text in [('prompt', case.messages[-1]['content']), ('reference', case.reference)]:
            values = grams(words(text))
            if values:
                key = (case.id, kind); targets[key] = values
                for gram in values: index[gram].add(key)
    matches = {}; count = 0
    with tokens_path.open() as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line); count += 1
            text = tokenizer.decode(row['input_ids'], skip_special_tokens=True)
            hits = Counter(key for gram in grams(words(text)) for key in index.get(gram, ()))
            for key, shared in hits.items():
                ratio = shared / len(targets[key])
                if ratio >= .8:
                    matches[key[0]] = {'reason': 'suspected_training_overlap', 'field': key[1],
                                      'coverage_of_13_word_ngrams': ratio, 'training_record_id': row.get('record_id')}
            if count % 1000 == 0: progress(f'  Training-overlap audit: {count} cached rows')
    return {'status': 'LIMITED_CHECK_COMPLETED', 'training_rows_checked': count, 'excluded': matches,
            'method': '>=80% containment of 13-word prompt OR reference ngrams in a single cached training row',
            'limitations': 'Short passages, paraphrases and pretraining overlap remain unresolved; absence of matches is not proof of independence.'}
