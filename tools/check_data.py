# 데이터 무결성 점검: data/Preprocessed/<ds>/<ds>_{train,val,test}.txt
import os, sys  # 경로 처리, 모듈 검색 경로 수정용
# 현재 작업 디렉터리를 검색 경로 맨 앞에 추가 (repo/dataset.py import용)
sys.path.insert(0, os.getcwd())
# 학습에 쓰는 데이터셋 클래스 (read_docs 필터를 재사용하기 위함)
from dataset import KoBARTGecDataset

# 전처리 데이터 루트 경로 (실행 위치 기준 상대경로)
ROOT = os.path.abspath('../data/Preprocessed')
# 점검 대상 데이터셋 이름
DATASETS = ['korean_learner', 'native', 'lang8', 'union']
# 각 데이터셋의 분할 종류
SPLITS = ['train', 'val', 'test']
# 논문에 보고된 데이터셋별 전체 샘플 수 (union은 보고값 없음)
PAPER = {'korean_learner': 28426, 'native': 17559, 'lang8': 109559}

def raw_lines(path):
    # UTF-8로 파일 열기
    with open(path, encoding='utf-8') as f:
        # 줄 단위 분리 (마지막 빈 문자열 포함)
        return f.read().split('\n')

def kept_rows(path):
    # dataset.py read_docs와 동일한 필터
    # self 없이 호출하기 위해 None을 전달 → 유효한 [소스, 타깃] 쌍만 반환
    return KoBARTGecDataset.read_docs(None, path)

def m2_sources(path):
    # M2 파일 열기
    with open(path, encoding='utf-8') as f:
        # 'S '로 시작하는 소스 문장 줄만 접두사 제거 후 추출
        return [l[2:] for l in f.read().split('\n') if l.startswith('S ')]

# (데이터셋, 분할) -> 유지된 쌍 목록 저장소
rows = {}
# 데이터셋별로 점검
for ds in DATASETS:
    print(f'\n##### {ds}')
    # 데이터셋 전체 유지 샘플 수 합계
    total = 0
    # 분할별로 점검
    for sp in SPLITS:
        # 해당 분할의 TSV 파일 경로
        p = f'{ROOT}/{ds}/{ds}_{sp}.txt'
        # 원본 줄 목록
        lines = raw_lines(p)
        # read_docs 필터를 통과한 쌍 목록
        kept = kept_rows(p)
        # 나중에 union 비교/중복 검사에 쓰도록 보관
        rows[(ds, sp)] = kept
        # 합계 누적
        total += len(kept)
        # 빈 줄 개수
        empty = sum(1 for l in lines if l == '')
        # 빈 줄이 아닌데 탭이 정확히 1개가 아닌 줄의 인덱스
        bad_tab = [i for i, l in enumerate(lines) if l != '' and l.count('\t') != 1]
        # 소스와 타깃이 같은(수정 없는) 쌍 개수
        same = sum(1 for s, t in kept if s == t)
        # 분할별 요약 출력
        print(f'[{sp}] 전체줄 {len(lines)} / 빈줄 {empty} / 탭≠1 {len(bad_tab)} / read_docs 유지 {len(kept)} / 소스=타깃 {same}')
        # 탭 이상 줄이 있으면
        if bad_tab[:3]:
            # 처음 3개만 예시로 출력 (120자까지)
            for i in bad_tab[:3]:
                print(f'    탭 이상 {i}: {lines[i][:120]!r}')
        # 별도 저장된 원문(컬럼 0)/교정문(컬럼 1) 파일과 일치 여부 확인
        for suffix, col in (('original', 0), ('corrected', 1)):
            # 해당 파일 경로
            q = f'{ROOT}/{ds}/{ds}_{sp}_{suffix}.txt'
            # 파일이 있는 경우만
            if os.path.exists(q):
                # 모든 줄 복사
                ref = [l for l in raw_lines(q)]
                # 마지막 빈 줄(파일 끝 개행) 제거
                ref = ref[:-1] if ref and ref[-1] == '' else ref
                # 줄 수가 같고 모든 줄이 해당 컬럼과 일치하는지
                match = len(ref) == len(kept) and all(r == k[col] for r, k in zip(ref, kept))
                print(f'    _{suffix}.txt {len(ref)}줄, .txt 컬럼{col}과 일치: {match}')
        # 이 분할의 M2 파일 경로
        m = f'{ROOT}/{ds}/{ds}_{sp}.m2'
        # M2 파일이 있는 경우만
        if os.path.exists(m):
            # M2의 소스 문장 목록
            srcs = m2_sources(m)
            # TSV 소스와 완전 일치 여부
            eq = len(srcs) == len(kept) and all(a == k[0] for a, k in zip(srcs, kept))
            # 공백 정규화(어절 분리) 후 일치 여부
            eq_ws = len(srcs) == len(kept) and all(a.split() == k[0].split() for a, k in zip(srcs, kept))
            print(f'    .m2 S줄 {len(srcs)}개, 소스와 완전일치 {eq}, 공백정규화 일치 {eq_ws}')
    # 논문 보고 샘플 수 (없으면 None)
    paper = PAPER.get(ds)
    # 합계와 논문값 대비 차이 출력
    print(f'합계 {total}' + (f' (논문 {paper}, 차이 {total - paper})' if paper else ''))
    # train/val/test 유지 샘플 수
    tr, va, te = (len(rows[(ds, s)]) for s in SPLITS)
    # 분할 비율 출력
    print(f'비율 train/val/test = {tr/total:.3f}/{va/total:.3f}/{te/total:.3f}')

print('\n##### union 구성 (각 split이 세 데이터셋의 합집합인지)')
# 각 분할별로 union이 세 데이터셋의 합과 같은지 비교
for sp in SPLITS:
    # union의 해당 분할 쌍 목록
    u = rows[('union', sp)]
    # 개별 데이터셋 3개의 해당 분할 쌍 목록
    parts = {ds: rows[(ds, sp)] for ds in ['korean_learner', 'native', 'lang8']}
    # 3개 데이터셋의 총 샘플 수
    n = sum(len(v) for v in parts.values())
    # 루프 안에서 Counter import (중복 집계용)
    from collections import Counter
    # 중복 개수까지 포함해 두 집합이 동일한지 (쌍을 튜플로 변환해 비교)
    same_multiset = Counter(map(tuple, u)) == Counter(t for v in parts.values() for t in map(tuple, v))
    print(f'[{sp}] union {len(u)} / 3개 합 {n} / 멀티셋 동일 {same_multiset}')

print('\n##### split 간 중복 (소스-타깃 쌍 기준)')
# 데이터셋별로 분할 간 중복(데이터 누수) 검사
for ds in DATASETS:
    # 각 분할의 쌍을 집합으로 변환
    s = {sp: set(map(tuple, rows[(ds, sp)])) for sp in SPLITS}
    # 분할 쌍별 교집합 크기 출력
    print(f'{ds}: train∩val {len(s["train"] & s["val"])} / train∩test {len(s["train"] & s["test"])} / val∩test {len(s["val"] & s["test"])}')
