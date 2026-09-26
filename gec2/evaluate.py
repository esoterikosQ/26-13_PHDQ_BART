# 원문 기준 평가 (plan2.md 원칙 2): source·reference는 데이터 파일의 원문, hypothesis만 모델 출력.
# GLEU는 저자 저장소의 metric/gleumodule.py, M²는 metric/m2scorer와 공식 <data>_<split>.m2를 그대로 쓴다.
#
# 사용 (저장소 루트에서):
#   python3 -m gec2.evaluate --data_root ../data/Preprocessed --data korean_learner --split test \
#       --hyp <hypothesis.txt> --out_dir <dir> [--m2]
import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from metric.gleumodule import run_gleu  # noqa: E402

M2SCORER = os.path.join(REPO_ROOT, 'metric', 'm2scorer', 'scripts', 'm2scorer.py')


def read_pairs(path):
    """KoBARTGecDataset.read_docs와 같은 규칙: 빈 줄·탭 2칸이 아닌 줄·빈 칸이 있는 줄은 버린다."""
    with open(path, encoding='utf-8') as f:
        rows = [x.split('\t') for x in f.read().split('\n') if x != '']
    return [(x[0], x[1]) for x in rows if len(x) == 2 and x[0] != '' and x[1] != '']


def split_path(data_root, data, split):
    return os.path.join(data_root, data, f'{data}_{split}.txt')


def read_hyp(path):
    """'\n'.join으로 저장된 기존 출력(마지막 줄바꿈 없음)과 줄마다 줄바꿈이 있는 새 출력을 모두 읽는다."""
    with open(path, encoding='utf-8') as f:
        text = f.read()
    if text.endswith('\n'):
        text = text[:-1]
    return text.split('\n')


def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        for x in lines:
            f.write(x.replace('\n', ' ') + '\n')


def score_gleu(sources, references, hypotheses, out_dir):
    assert len(sources) == len(references) == len(hypotheses), \
        f'줄 수 불일치: src {len(sources)} ref {len(references)} hyp {len(hypotheses)}'
    os.makedirs(out_dir, exist_ok=True)
    files = {k: os.path.join(out_dir, f'{k}.txt') for k in ('source', 'reference', 'hypothesis')}
    write_lines(files['source'], sources)
    write_lines(files['reference'], references)
    write_lines(files['hypothesis'], hypotheses)
    return float(run_gleu(reference=files['reference'], source=files['source'], hypothesis=files['hypothesis'])) * 100


def score_m2(hyp_file, m2_file, out_file, python=sys.executable):
    out = subprocess.run([python, M2SCORER, hyp_file, m2_file], capture_output=True, text=True, check=True).stdout
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(out)
    vals = dict(re.findall(r'(Precision|Recall|F_0\.5)\s*:\s*([0-9.]+)', out))
    return {'p': float(vals['Precision']) * 100, 'r': float(vals['Recall']) * 100, 'f05': float(vals['F_0.5']) * 100}


def evaluate(data_root, data, split, hypotheses, out_dir, m2=False, extra=None, limit=None):
    """hypotheses: 데이터 파일(필터 후) 순서의 출력 리스트. out_dir에 파일과 scores.json을 남기고 점수 dict를 돌려준다.
    limit: 스모크용. 앞 N쌍만 채점하고 M²도 앞 N블록만 쓴다(본 실험에서는 None)."""
    pairs = read_pairs(split_path(data_root, data, split))
    if limit:
        pairs = pairs[:limit]
    scores = {'data': data, 'split': split, 'n': len(pairs), 'limit': limit,
              'gleu': score_gleu([s for s, _ in pairs], [t for _, t in pairs], hypotheses, out_dir)}
    with open(os.path.join(out_dir, 'gleu.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{scores['gleu'] / 100:.6f}\n")
    if m2:
        m2_file = os.path.join(data_root, data, f'{data}_{split}.m2')
        if limit:
            with open(m2_file, encoding='utf-8') as f:
                blocks = f.read().strip().split('\n\n')[:limit]
            m2_file = os.path.join(out_dir, f'gold_first{limit}.m2')
            with open(m2_file, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(blocks) + '\n')
        scores.update(score_m2(os.path.join(out_dir, 'hypothesis.txt'), m2_file, os.path.join(out_dir, 'm2score.txt')))
    if extra:
        scores.update(extra)
    with open(os.path.join(out_dir, 'scores.json'), 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=1)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default=os.path.join(REPO_ROOT, '..', 'data', 'Preprocessed'))
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--hyp', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--m2', action='store_true')
    a = ap.parse_args()
    s = evaluate(a.data_root, a.data, a.split, read_hyp(a.hyp), a.out_dir, m2=a.m2)
    print(json.dumps(s, ensure_ascii=False))


if __name__ == '__main__':
    main()
