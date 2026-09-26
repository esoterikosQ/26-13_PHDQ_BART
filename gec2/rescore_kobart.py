# plan2.md §5 단계 1: 재현된 KoBART 21 run의 기존 생성 결과를 원문 기준 평가(gec2/evaluate.py)로 다시 채점해
# 기존 수치(모델 토크나이저로 디코딩한 source·reference 기준)와 비교한다.
#
# 사용 (저장소 루트에서): python3 -m gec2.rescore_kobart [--out outputs2/kobart_repro]
import argparse
import glob
import json
import os
import re

from gec2.evaluate import REPO_ROOT, evaluate, read_hyp, read_pairs, split_path

DATAS = ('korean_learner', 'native', 'lang8', 'union')
RUNS = [(d, f'dropout_seed{s}') for d in DATAS for s in range(3)] + \
       [(d, f'afterunion_dropout_seed{s}') for d in DATAS[:3] for s in range(3)]


def old_gleu(path):
    return float(re.search(r'gleu_out: ([0-9.]+)', open(path, encoding='utf-8').read())[1]) * 100


def old_m2(path):
    vals = dict(re.findall(r'(Precision|Recall|F_0\.5)\s*:\s*([0-9.]+)', open(path, encoding='utf-8').read()))
    return {'p': float(vals['Precision']) * 100, 'r': float(vals['Recall']) * 100, 'f05': float(vals['F_0.5']) * 100}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=os.path.join(REPO_ROOT, '..', 'data', 'Preprocessed'))
    ap.add_argument('--gen_root', default=os.path.join(REPO_ROOT, 'outputs', 'generation'))
    ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'outputs2', 'kobart_repro'))
    a = ap.parse_args()

    rows, max_val_diff, text_diff = [], 0.0, {}
    for data, run in RUNS:
        raw = {sp: read_pairs(split_path(a.data_root, data, sp)) for sp in ('val', 'test')}
        val = {}
        for d in sorted(glob.glob(f'{a.gen_root}/{data}/{run}/epoch*/val')):
            ep = int(d.split('/epoch')[-1].split('/')[0])
            new = evaluate(a.data_root, data, 'val', read_hyp(f'{d}/hypothesis.txt'), f'{a.out}/{data}/{run}/epoch{ep:02d}/val')
            val[ep] = (old_gleu(f'{d}/gleu.txt'), new['gleu'])
            max_val_diff = max(max_val_diff, abs(val[ep][0] - val[ep][1]))
        # 기존 파일의 source·reference(모델 디코딩)와 원문 비교
        for sp, d in [('val', sorted(glob.glob(f'{a.gen_root}/{data}/{run}/epoch*/val'))[0])] + \
                     [('test', glob.glob(f'{a.gen_root}/{data}/{run}/epoch*/test')[0])]:
            src, ref = read_hyp(f'{d}/source.txt'), read_hyp(f'{d}/reference.txt')
            n = sum(s != x or t != y for s, t, (x, y) in zip(src, ref, raw[sp]))
            text_diff[(data, sp)] = (n + abs(len(src) - len(raw[sp])), len(raw[sp]))
        tdir = glob.glob(f'{a.gen_root}/{data}/{run}/epoch*/test')[0]
        ep_t = int(tdir.split('/epoch')[-1].split('/')[0])
        new = evaluate(a.data_root, data, 'test', read_hyp(f'{tdir}/hypothesis.txt'),
                       f'{a.out}/{data}/{run}/epoch{ep_t:02d}/test', m2=True)
        old = {'gleu': old_gleu(f'{tdir}/gleu.txt'), **old_m2(f'{tdir}/m2score.txt')}
        best_old = max(val, key=lambda e: val[e][0])
        best_new = max(val, key=lambda e: val[e][1])
        rows.append((data, run, ep_t, best_old, best_new, old, new))

    lines = ['| 데이터 | run | test epoch | best val epoch 기존→원문 | 테스트 GLEU 기존 → 원문 | P | R | F0.5 |',
             '| --- | --- | --- | --- | --- | --- | --- | --- |']
    for data, run, ep_t, bo, bn, old, new in rows:
        cells = [f"{old[k]:.2f} → {new[k]:.2f}" for k in ('gleu', 'p', 'r', 'f05')]
        lines.append(f'| {data} | {run} | {ep_t} | {bo}→{bn} | ' + ' | '.join(cells) + ' |')
    lines.append('')
    lines.append(f'검증 GLEU(전 epoch) 기존 대비 최대 절대차: {max_val_diff:.4f}')
    lines.append('기존 source/reference(모델 디코딩)가 원문과 다른 줄 수: ' +
                 ', '.join(f'{d} {sp} {n}/{tot}' for (d, sp), (n, tot) in text_diff.items()))
    lines.append('테스트 최대 절대차: ' + ', '.join(
        f"{k} {max(abs(r[5][k] - r[6][k]) for r in rows):.4f}" for k in ('gleu', 'p', 'r', 'f05')))
    report = '\n'.join(lines)
    os.makedirs(a.out, exist_ok=True)
    with open(f'{a.out}/rescore.md', 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    with open(f'{a.out}/rescore.json', 'w', encoding='utf-8') as f:
        json.dump([{'data': r[0], 'run': r[1], 'test_epoch': r[2], 'best_val_epoch_old': r[3],
                    'best_val_epoch_new': r[4], 'old': r[5], 'new': r[6]} for r in rows], f, ensure_ascii=False, indent=1)
    print(report)


if __name__ == '__main__':
    main()
