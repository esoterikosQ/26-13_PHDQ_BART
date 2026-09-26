# 데이터 무결성 점검: data/Preprocessed/<ds>/<ds>_{train,val,test}.txt
import os, sys
sys.path.insert(0, os.getcwd())
from dataset import KoBARTGecDataset

ROOT = os.path.abspath('../data/Preprocessed')
DATASETS = ['korean_learner', 'native', 'lang8', 'union']
SPLITS = ['train', 'val', 'test']
PAPER = {'korean_learner': 28426, 'native': 17559, 'lang8': 109559}

def raw_lines(path):
    with open(path, encoding='utf-8') as f:
        return f.read().split('\n')

def kept_rows(path):
    # dataset.py read_docs와 동일한 필터
    return KoBARTGecDataset.read_docs(None, path)

def m2_sources(path):
    with open(path, encoding='utf-8') as f:
        return [l[2:] for l in f.read().split('\n') if l.startswith('S ')]

rows = {}
for ds in DATASETS:
    print(f'\n##### {ds}')
    total = 0
    for sp in SPLITS:
        p = f'{ROOT}/{ds}/{ds}_{sp}.txt'
        lines = raw_lines(p)
        kept = kept_rows(p)
        rows[(ds, sp)] = kept
        total += len(kept)
        empty = sum(1 for l in lines if l == '')
        bad_tab = [i for i, l in enumerate(lines) if l != '' and l.count('\t') != 1]
        same = sum(1 for s, t in kept if s == t)
        print(f'[{sp}] 전체줄 {len(lines)} / 빈줄 {empty} / 탭≠1 {len(bad_tab)} / read_docs 유지 {len(kept)} / 소스=타깃 {same}')
        if bad_tab[:3]:
            for i in bad_tab[:3]:
                print(f'    탭 이상 {i}: {lines[i][:120]!r}')
        for suffix, col in (('original', 0), ('corrected', 1)):
            q = f'{ROOT}/{ds}/{ds}_{sp}_{suffix}.txt'
            if os.path.exists(q):
                ref = [l for l in raw_lines(q)]
                ref = ref[:-1] if ref and ref[-1] == '' else ref
                match = len(ref) == len(kept) and all(r == k[col] for r, k in zip(ref, kept))
                print(f'    _{suffix}.txt {len(ref)}줄, .txt 컬럼{col}과 일치: {match}')
        m = f'{ROOT}/{ds}/{ds}_{sp}.m2'
        if os.path.exists(m):
            srcs = m2_sources(m)
            eq = len(srcs) == len(kept) and all(a == k[0] for a, k in zip(srcs, kept))
            eq_ws = len(srcs) == len(kept) and all(a.split() == k[0].split() for a, k in zip(srcs, kept))
            print(f'    .m2 S줄 {len(srcs)}개, 소스와 완전일치 {eq}, 공백정규화 일치 {eq_ws}')
    paper = PAPER.get(ds)
    print(f'합계 {total}' + (f' (논문 {paper}, 차이 {total - paper})' if paper else ''))
    tr, va, te = (len(rows[(ds, s)]) for s in SPLITS)
    print(f'비율 train/val/test = {tr/total:.3f}/{va/total:.3f}/{te/total:.3f}')

print('\n##### union 구성 (각 split이 세 데이터셋의 합집합인지)')
for sp in SPLITS:
    u = rows[('union', sp)]
    parts = {ds: rows[(ds, sp)] for ds in ['korean_learner', 'native', 'lang8']}
    n = sum(len(v) for v in parts.values())
    from collections import Counter
    same_multiset = Counter(map(tuple, u)) == Counter(t for v in parts.values() for t in map(tuple, v))
    print(f'[{sp}] union {len(u)} / 3개 합 {n} / 멀티셋 동일 {same_multiset}')

print('\n##### split 간 중복 (소스-타깃 쌍 기준)')
for ds in DATASETS:
    s = {sp: set(map(tuple, rows[(ds, sp)])) for sp in SPLITS}
    print(f'{ds}: train∩val {len(s["train"] & s["val"])} / train∩test {len(s["train"] & s["test"])} / val∩test {len(s["val"] & s["test"])}')
