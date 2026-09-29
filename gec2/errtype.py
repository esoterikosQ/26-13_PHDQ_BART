# plan2.md §5 단계 6: 오류 유형별 점수 (논문 Table 7 / Table D.4 재구성).
# 논문은 계산 방식을 적지 않았다. Table D.4에서 편집이 없는 유형(WO 0개)이 P/R/F0.5 100, GLEU 0으로 나오는 것
# (m2scorer에 빈 집합을 넣은 결과)으로 보아, "그 유형의 정답 편집이 하나 이상 있는 테스트 문장만 골라 GLEU·M²를 계산"한
# 것으로 재구성한다. 재현 KoBART 출력으로 같은 계산을 해 논문 수치와 비교해 검증한다(--validate).
# 사용 (저장소 루트): python3 -m gec2.errtype [--validate] [--jobs 10] [--out logs/gec2_errtype]
import argparse
import json
import os
import re
import statistics as st
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

from gec2.evaluate import M2SCORER, REPO_ROOT, read_hyp, read_pairs, split_path
from metric.gleumodule import run_gleu

TYPES = [('INSERTION', 'INS'), ('DELETION', 'DEL'), ('SPELL', 'SPELL'), ('PUNCT', 'PUNCT'), ('SHORTEN', 'SHORT'),
         ('WS', 'WS'), ('WO', 'WO'), ('NOUN', 'NOUN'), ('VERB', 'VERB'), ('ADJECTIVE', 'ADJ'), ('CONJUGATION', 'CONJ'),
         ('PARTICLE', 'PART'), ('ENDING', 'END'), ('MODIFIER', 'MOD'), ('UNCLASSIFIED', 'UNK')]
DATAS = ('korean_learner', 'native', 'lang8', 'union')
MODELS = [  # (이름, 결과 루트, 시나리오 1 lr)
    ('KoBART(gec2)', 'outputs2/kobart-base-v1', '5e-5'), ('pko-t5-base', 'outputs2/pko-t5-base', '5e-4'),
    ('mBART-50', 'outputs2/mbart-large-50', '3e-5'), ('pko-t5-large', 'outputs2/pko-t5-large', '1e-4'),
    ('kanana-1.5-2.1b', 'neuron_outputs/kanana-1.5-2.1b', '1e-5')]
# 논문 Table D.4, KoBART (test) Kor-Union 행 (검증용): 유형 순서는 TYPES, 마지막은 TOTAL
PAPER_KOBART_UNION = {
    'gleu': [23.67, 18.22, 36.10, 20.29, 33.42, 34.83, 13.76, 25.09, 27.96, 29.27, 25.95, 27.15, 27.72, 23.68, 26.44, 33.70],
    'p': [40.58, 45.64, 52.16, 42.35, 58.52, 63.26, 37.89, 43.76, 46.45, 49.16, 41.30, 49.35, 47.51, 46.67, 42.69, 44.75],
    'r': [9.47, 12.05, 22.54, 10.06, 19.61, 21.48, 9.04, 12.46, 13.29, 13.92, 12.34, 14.30, 13.23, 12.70, 12.68, 14.64],
    'f05': [24.47, 29.28, 41.30, 25.78, 41.84, 45.48, 23.11, 29.11, 30.98, 32.61, 28.09, 33.10, 31.28, 30.38, 28.97, 31.70]}


def read_m2_blocks(path):
    with open(path, encoding='utf-8') as f:
        blocks = [b for b in f.read().strip().split('\n\n') if b.strip()]
    types = [{line.split('|||')[1] for line in b.split('\n') if line.startswith('A ')} for b in blocks]
    return blocks, types


def score_subset(args):
    """idx 문장만 모아 GLEU(원문 기준)와 M²를 계산."""
    key, idx, pairs, hyps, blocks = args
    if not idx:
        return key, {'n': 0}
    with tempfile.TemporaryDirectory() as d:
        paths = {k: os.path.join(d, f'{k}.txt') for k in ('source', 'reference', 'hypothesis')}
        for k, lines in (('source', [pairs[i][0] for i in idx]), ('reference', [pairs[i][1] for i in idx]),
                         ('hypothesis', [hyps[i] for i in idx])):
            with open(paths[k], 'w', encoding='utf-8') as f:
                f.write('\n'.join(x.replace('\n', ' ') for x in lines) + '\n')
        m2 = os.path.join(d, 'gold.m2')
        with open(m2, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(blocks[i] for i in idx) + '\n')
        g = float(run_gleu(reference=paths['reference'], source=paths['source'], hypothesis=paths['hypothesis'])) * 100
        out = subprocess.run([sys.executable, M2SCORER, paths['hypothesis'], m2], capture_output=True, text=True, check=True).stdout
        v = dict(re.findall(r'(Precision|Recall|F_0\.5)\s*:\s*([0-9.]+)', out))
    return key, {'n': len(idx), 'gleu': g, 'p': float(v['Precision']) * 100, 'r': float(v['Recall']) * 100,
                 'f05': float(v['F_0.5']) * 100}


def tasks_for(system, data, seed, hyp_path, data_root):
    pairs = read_pairs(split_path(data_root, data, 'test'))
    blocks, types = read_m2_blocks(os.path.join(data_root, data, f'{data}_test.m2'))
    hyps = read_hyp(hyp_path)
    assert len(pairs) == len(blocks) == len(hyps), (data, len(pairs), len(blocks), len(hyps))
    out = []
    for full, short in TYPES + [('TOTAL', 'TOTAL')]:
        idx = list(range(len(pairs))) if full == 'TOTAL' else [i for i, t in enumerate(types) if full in t]
        out.append(((system, data, seed, short), idx, pairs, hyps, blocks))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=os.path.join(REPO_ROOT, '..', 'data', 'Preprocessed'))
    ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'logs', 'gec2_errtype'))
    ap.add_argument('--jobs', type=int, default=10)
    ap.add_argument('--validate', action='store_true', help='재현 KoBART(union, 3 seed)만 계산해 논문 Table D.4와 비교')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tasks = []
    if a.validate:
        import glob
        for s in range(3):
            h = glob.glob(os.path.join(REPO_ROOT, f'outputs/generation/union/dropout_seed{s}/epoch*/test/hypothesis.txt'))[0]
            tasks += tasks_for('재현 KoBART', 'union', s, h, a.data_root)
    else:
        for name, root, lr in MODELS:
            for d in DATAS:
                for s in range(3):
                    tasks += tasks_for(name, d, s, os.path.join(REPO_ROOT, root, d, f'lr{lr}_seed{s}', 'test', 'hypothesis.txt'), a.data_root)
    with ProcessPoolExecutor(a.jobs) as ex:
        res = dict(ex.map(score_subset, tasks, chunksize=1))
    raw = [{'system': k[0], 'data': k[1], 'seed': k[2], 'type': k[3], **v} for k, v in res.items()]
    with open(os.path.join(a.out, 'validate.json' if a.validate else 'errtype.json'), 'w', encoding='utf-8') as f:
        json.dump(raw, f, ensure_ascii=False, indent=1)
    cols = [s for _, s in TYPES] + ['TOTAL']

    def mean(system, data, t, k):
        v = [r[k] for r in raw if r['system'] == system and r['data'] == data and r['type'] == t and r['n'] > 0]
        return st.mean(v) if v else None

    if a.validate:
        print('| 지표 | ' + ' | '.join(cols) + ' |')
        for k in ('gleu', 'p', 'r', 'f05'):
            ours = [mean('재현 KoBART', 'union', t, k) for t in cols]
            print(f'| {k} 재현 | ' + ' | '.join('—' if x is None else f'{x:.2f}' for x in ours) + ' |')
            print(f'| {k} 논문 | ' + ' | '.join(f'{x:.2f}' for x in PAPER_KOBART_UNION[k]) + ' |')
            diffs = [abs(x - y) for x, y in zip(ours, PAPER_KOBART_UNION[k]) if x is not None]
            print(f'| {k} 차이 평균/최대 | {st.mean(diffs):.2f} / {max(diffs):.2f} |')
        return
    for d in DATAS:
        n = {t: next(r['n'] for r in raw if r['data'] == d and r['type'] == t) for t in cols}
        for k, lab in (('gleu', 'GLEU'), ('f05', 'F0.5')):
            print(f'### {d} — 유형별 테스트 {lab} (시나리오 1, 3-seed 평균; 괄호는 해당 유형 편집이 있는 문장 수)\n')
            print('| 모델 | ' + ' | '.join(f'{t} ({n[t]})' for t in cols) + ' |')
            print('| --- | ' + ' | '.join('---' for _ in cols) + ' |')
            for name, _, _ in MODELS:
                vals = [mean(name, d, t, k) for t in cols]
                print(f'| {name} | ' + ' | '.join('—' if x is None else f'{x:.2f}' for x in vals) + ' |')
            print()


if __name__ == '__main__':
    main()
