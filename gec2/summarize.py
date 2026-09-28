# plan2.md §5 단계 6: 모델 × 데이터셋 × 시나리오 결과를 3-seed 평균·범위로 모은다 (markdown 표 출력).
# 사용 (저장소 루트, itcerdo에서 git pull로 neuron_outputs/까지 받은 뒤): python3 -m gec2.summarize [--json out.json]
import argparse
import json
import os
import statistics as st

from gec2.evaluate import REPO_ROOT

# (이름, 파라미터, 결과 루트, 시나리오 1 lr, 트랙)
MODELS = [
    ('KoBART(gec2)', '124M', 'outputs2/kobart-base-v1', '5e-5', 'seq2seq'),
    ('pko-t5-base', '276M', 'outputs2/pko-t5-base', '5e-4', 'seq2seq'),
    ('mBART-50', '611M', 'outputs2/mbart-large-50', '3e-5', 'seq2seq'),
    ('pko-t5-large', '821M', 'outputs2/pko-t5-large', '1e-4', 'seq2seq'),
    ('kanana-1.5-2.1b', '2.09B', 'neuron_outputs/kanana-1.5-2.1b', '1e-5', 'llm'),
]
DATAS = ('korean_learner', 'native', 'lang8', 'union')
SEEDS = (0, 1, 2)
# 재현 KoBART(논문 recipe, result_kobart.md 3-seed 평균)와 논문 Table 6: (GLEU, P, R, F0.5)
REPRO = {('s1', 'korean_learner'): (45.68, 43.31, 25.46, 37.98), ('s1', 'native'): (68.40, 80.34, 58.31, 74.69),
         ('s1', 'lang8'): (29.65, 39.46, 13.33, 28.33), ('s1', 'union'): (34.75, 44.38, 16.09, 32.80),
         ('s2', 'korean_learner'): (44.19, 52.15, 23.47, 41.88), ('s2', 'native'): (60.96, 86.28, 48.73, 74.70),
         ('s2', 'lang8'): (29.35, 37.84, 13.05, 27.41)}
PAPER = {('s1', 'korean_learner'): (45.06, 43.35, 24.54, 37.58), ('s1', 'native'): (67.24, 75.34, 55.95, 70.45),
         ('s1', 'lang8'): (28.48, 37.56, 11.62, 25.93), ('s1', 'union'): (33.70, 44.75, 14.64, 31.70),
         ('s2', 'korean_learner'): (42.66, 53.51, 21.18, 41.00), ('s2', 'native'): (59.71, 85.47, 47.38, 73.63),
         ('s2', 'lang8'): (28.65, 37.46, 12.00, 26.78)}
KEYS = (('gleu', 'GLEU'), ('p', 'P'), ('r', 'R'), ('f05', 'F0.5'))


def run_dir(root, lr, sc, data, seed):
    return os.path.join(REPO_ROOT, root, data, f'{"s2_" if sc == "s2" else ""}lr{lr}_seed{seed}')


def collect():
    res = {}
    for name, _, root, lr, _ in MODELS:
        for sc in ('s1', 's2'):
            for d in DATAS:
                if sc == 's2' and d == 'union':
                    continue
                runs = []
                for s in SEEDS:
                    f = os.path.join(run_dir(root, lr, sc, d, s), 'test', 'scores.json')
                    if os.path.exists(f):
                        x = json.load(open(f))
                        runs.append({'seed': s, **{k: x[k] for k, _ in KEYS}, 'best_epoch': x['best_epoch'],
                                     'best_val_gleu': x['best_val_gleu']})
                res[(name, sc, d)] = runs
    return res


def cell(runs, k):
    if not runs:
        return '—'
    v = [r[k] for r in runs]
    m = st.mean(v)
    if len(v) == 1:
        return f'{m:.2f} (1 seed)'
    return f'{m:.2f} ±{(max(v) - min(v)) / 2:.2f}' + ('' if len(v) == 3 else f' ({len(v)} seed)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    res = collect()
    for sc, title in (('s1', '시나리오 1 (개별 데이터셋)'), ('s2', '시나리오 2 (union → 개별)')):
        ds = [d for d in DATAS if not (sc == 's2' and d == 'union')]
        for k, lab in KEYS:
            print(f'### {title} — 테스트 {lab} (3-seed 평균 ±범위/2)\n')
            print('| 모델 | 파라미터 | ' + ' | '.join(ds) + ' |')
            print('| --- | --- | ' + ' | '.join('---' for _ in ds) + ' |')
            for name, size, _, _, _ in MODELS:
                print(f'| {name} | {size} | ' + ' | '.join(cell(res[(name, sc, d)], k) for d in ds) + ' |')
            i = [x for x, _ in KEYS].index(k)
            print('| (참고) 재현 KoBART, 논문 recipe | 124M | ' + ' | '.join(f'{REPRO[(sc, d)][i]:.2f}' for d in ds) + ' |')
            print('| (참고) 논문 Table 6 | 124M | ' + ' | '.join(f'{PAPER[(sc, d)][i]:.2f}' for d in ds) + ' |')
            print()
    missing = [f'{n} {sc} {d}: {len(r)}/3' for (n, sc, d), r in res.items() if len(r) < 3]
    print('누락 seed: ' + (', '.join(missing) if missing else '없음'))
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as f:
            json.dump({f'{n}|{sc}|{d}': r for (n, sc, d), r in res.items()}, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
