# logs/runs.log에서 3-seed 평균을 계산해 논문 Table 6과 비교 (markdown 표 출력)
import re, sys, statistics as st

res = {}
for line in open(sys.argv[1], encoding='utf-8'):
    m = re.search(r'TEST DONE (\S+) (\S+) .*gleu_out: ([0-9.]+)', line)
    if m:
        res.setdefault((m[1], m[2]), {})['g'] = float(m[3]) * 100
    m = re.search(r'M2 DONE (\S+) (\S+) Precision\s*: ([0-9.]+) Recall\s*: ([0-9.]+) F_0.5\s*: ([0-9.]+)', line)
    if m:
        res.setdefault((m[1], m[2]), {}).update(p=float(m[3]) * 100, r=float(m[4]) * 100, f=float(m[5]) * 100)

PAPER = {('korean_learner', 's1'): (45.06, 43.35, 24.54, 37.58), ('native', 's1'): (67.24, 75.34, 55.95, 70.45),
         ('lang8', 's1'): (28.48, 37.56, 11.62, 25.93), ('union', 's1'): (33.70, 44.75, 14.64, 31.70),
         ('korean_learner', 's2'): (42.66, 53.51, 21.18, 41.00), ('native', 's2'): (59.71, 85.47, 47.38, 73.63),
         ('lang8', 's2'): (28.65, 37.46, 12.00, 26.78)}
PREFIX = {'s1': (sys.argv[2] if len(sys.argv) > 2 else 'legacyfix') + '_seed', 's2': 'afterunion_' + (sys.argv[2] + '_' if len(sys.argv) > 2 else '') + 'seed'}
if len(sys.argv) <= 2: PREFIX['s2'] = 'afterunion_seed'
NAME = {'s1': '1 개별', 's2': '2 Union→개별'}

print('| 시나리오 | 데이터 | GLEU 재현 / 논문 | P 재현 / 논문 | R 재현 / 논문 | F0.5 재현 / 논문 | GLEU seed 범위 |')
print('| --- | --- | --- | --- | --- | --- | --- |')
for sc in ('s1', 's2'):
    for d in ('korean_learner', 'native', 'lang8', 'union'):
        if (d, sc) not in PAPER:
            continue
        rs = [res[(d, PREFIX[sc] + str(s))] for s in range(3) if 'f' in res.get((d, PREFIX[sc] + str(s)), {})]
        if not rs:
            continue
        mean = {k: st.mean(r[k] for r in rs) for k in 'gprf'}
        paper = dict(zip('gprf', PAPER[(d, sc)]))
        cells = [f'{mean[k]:.2f} / {paper[k]:.2f} ({mean[k] - paper[k]:+.2f})' for k in 'gprf']
        rng = max(r['g'] for r in rs) - min(r['g'] for r in rs)
        print(f'| {NAME[sc]} | {d} | ' + ' | '.join(cells) + f' | {rng:.2f} ({len(rs)} seed) |')
