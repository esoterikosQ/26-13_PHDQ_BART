# 분할 재현(random_state=1)과 NFKC 적용 상태 점검
import os, glob, unicodedata
from sklearn.model_selection import train_test_split

R = os.path.abspath('../data')

def lines(p):
    return [d for d in open(p, encoding='utf-8').read().split('\n') if d != '']

print('== 분할 재현 (저장소 train_test_val_split, random_state=1)')
for ds in ['korean_learner', 'native', 'lang8']:
    data = lines(f'{R}/Preprocessed/{ds}/{ds}.txt')
    tr, vt = train_test_split(data, test_size=0.3, random_state=1)
    va, te = train_test_split(vt, test_size=0.5, random_state=1)
    cur = {sp: lines(f'{R}/Preprocessed/{ds}/{ds}_{sp}.txt') for sp in ['train', 'val', 'test']}
    print(f'  {ds}: 전체 {len(data)} | 순서까지 동일 train {tr == cur["train"]} val {va == cur["val"]} test {te == cur["test"]}'
          f' | 집합 동일 train {sorted(tr) == sorted(cur["train"])} val {sorted(va) == sorted(cur["val"])} test {sorted(te) == sorted(cur["test"])}')

print('== NFKC 적용 여부 (NFKC로 바뀌는 줄 수)')
files = sorted(glob.glob(f'{R}/Preprocessed/*/*.txt')) + sorted(glob.glob(f'{R}/Raw/*.txt'))
for p in files:
    if 'hanspell' in p:
        continue
    L = lines(p)
    n = sum(unicodedata.normalize('NFKC', x) != x for x in L)
    print(f'  {p.replace(R + "/", ""):58s} {len(L):7d}줄  NFKC 변경 {n}')

print('== Raw vs Preprocessed 전체 파일 비교')
for ds in ['korean_learner', 'native']:
    a = lines(f'{R}/Raw/{ds}.txt')
    b = lines(f'{R}/Preprocessed/{ds}/{ds}.txt')
    nf = [unicodedata.normalize('NFKC', x) for x in a]
    print(f'  {ds}: Raw {len(a)} / Pre {len(b)} / 동일 {a == b} / NFKC(Raw)==Pre {nf == b}')
