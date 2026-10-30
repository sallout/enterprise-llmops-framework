"""Executable base-test pass rates and explicitly lexical enterprise proxies."""
import math
import re
from collections import Counter
from .execution import extract_code


def words(text):
    return re.findall(r'\w+', text.casefold())


def lexical(answer, reference):
    a, b = words(answer), words(reference)
    common = sum((Counter(a) & Counter(b)).values())
    f1 = 2 * common / (len(a) + len(b)) if a and b else 0.0
    return {'normalized_exact_match': bool(a and a == b), 'token_f1': f1,
            'semantic_correctness': 'NOT_ASSESSED', 'human_review_required': True}


def score(case, generated, executor):
    if generated['status'] != 'ok':
        return {'status': 'failed', 'reason': generated['status']}
    if case.suite == 'enterprise':
        return {'status': 'review_required', **lexical(generated['text'], case.reference)}
    if executor is None:
        return {'status': 'unverified', 'reason': 'execution_disabled'}
    code = extract_code(generated['text'])
    if code is None:
        return {'status': 'failed', 'reason': 'ambiguous_code_output'}
    return executor.run(code, case.tests)


def summarize(rows):
    result = {}
    for suite in sorted({r['suite'] for r in rows}):
        selected = [r for r in rows if r['suite'] == suite]
        n = len(selected)
        if suite == 'enterprise':
            assessed = [r['score'] for r in selected if 'token_f1' in r['score']]
            result[suite] = {'selected': n, 'answers_scored_lexically': len(assessed),
                'normalized_exact_match': sum(r['normalized_exact_match'] for r in assessed) / n,
                'mean_token_f1': sum(r['token_f1'] for r in assessed) / n,
                'semantic_correctness': 'NOT_ASSESSED', 'human_review_required': n}
        else:
            counts = Counter(r['score']['status'] for r in selected)
            passed = counts['passed']; unverified = counts['unverified']
            p = passed / n; z = 1.96
            center = (p + z*z/(2*n)) / (1+z*z/n)
            half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
            result[suite] = {'selected': n, 'passed': passed, 'failed': counts['failed'],
                'unverified': unverified, 'pass_at_1': None if unverified else p,
                'wilson_95_interval': None if unverified else [max(0, center-half), min(1, center+half)],
                'scope': 'one greedy answer; selected base Python tests only'}
    measured = [r['generation'] for r in rows if r['generation'].get('generation_seconds') is not None]
    seconds = sum(r['generation_seconds'] for r in measured)
    result['efficiency'] = {'measured_tasks': len(measured), 'generation_seconds': seconds,
        'output_tokens_per_second_including_prefill': sum(r['output_tokens'] for r in measured)/seconds if seconds else None,
        'peak_allocated_bytes': max((r.get('peak_allocated_bytes', 0) for r in measured), default=0),
        'peak_reserved_bytes': max((r.get('peak_reserved_bytes', 0) for r in measured), default=0),
        'note': 'Sequential batch 1, synchronized wall time including prefill; TTFT not measured; excludes model load and warmup.'}
    return result
