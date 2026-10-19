'''
https://github.com/soyoung97/Standard_Korean_GEC
Modified MIT License

Software Copyright (c) 2022 Soyoung Yoon

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.
The above copyright notice and this permission notice need not be included
with content created by the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.
'''
'''
Description: This file preprocesses and make Lang-8 dataset.

Usage: python filter.py -d [file path] -o [file path]
-d: pickled result file path after running extract_err-cor-pair.py
-o: file path to save the result of this script
'''
from tqdm import tqdm  # 진행률 표시줄
import pandas as pd  # 데이터프레임 처리
import re  # 정규표현식
import argparse  # 커맨드라인 인자 파싱
import string  # string.punctuation(구두점 목록) 사용
import math  # 수학 함수 (현재는 주석 처리된 코드에서만 사용)
from gluonnlp.data import SentencepieceTokenizer  # SentencePiece 토크나이저 래퍼
from kobert.utils import get_tokenizer  # KoBERT 토크나이저 모델 경로 다운로드/반환
from collections import Counter  # 중복 개수 집계
from hangul_util import *  # 한글 처리 유틸 함수 전부 가져오기
from soynlp.hangle import jamo_levenshtein  # 자모 단위 편집 거리



def parse_args():
    # 인자 파서 생성
    parser = argparse.ArgumentParser()
    # 입력 파일 경로 (extract 단계에서 만든 원본 오류-교정 쌍)
    parser.add_argument('-d', '--datafile', type=str, default='lang8_raw.txt')
    # 필터링 결과 저장 경로
    parser.add_argument('-o', '--output', type=str, default='lang8_filtered.txt')
    # 정제 완료 결과 저장 경로
    parser.add_argument('-c', '--cleaned_output', type=str, default='lang8.txt')
    # 인자 파싱 실행
    args = parser.parse_args()
    # 파싱 결과 반환
    return args

def load_data(file_path):
    # 오류 문장 리스트
    original = []
    # 교정 문장 리스트
    corrected = []
    # 파일 열기 (읽기 모드)
    with open(file_path, 'r') as f:
        # 전체를 줄 단위로 나누고 빈 줄 제거
        data = [x for x in f.read().split('\n') if x != '']
        # 진행률 표시하며 각 줄 순회
        for line in tqdm(data):
            try:
                # 탭으로 오류문/교정문 분리
                o, c = line.split('\t')
            except:
                # 탭 개수가 맞지 않는 줄은 건너뜀
                continue
            # 오류 문장 추가
            original.append(o)
            # 교정 문장 추가
            corrected.append(c)
    # 두 리스트 반환
    return original, corrected

def get_df(data):
    # 각 쌍의 첫 번째 원소(오류 문장) 추출
    original = [x[0] for x in data]
    # 각 쌍의 두 번째 원소(교정 문장) 추출
    corrected = [x[1] for x in data]
    # 두 열을 가진 DataFrame 생성
    df = pd.DataFrame({'original': original, 'corrected': corrected})
    # DataFrame 반환
    return df

def drop_other_lang(data):
    # 허용 문자(한글/숫자/영문/일부 기호) 이외의 문자가 하나라도 있으면 매칭되는 패턴
    hangul = re.compile('[^ ㄱ-ㅎㅏ-ㅣ가-힣0-9a-zA-Z@#$%&\(\){|}\[\]\/\-><=_:;.,?!~\^"（）\'‘’]+')
    # 한글 자모/음절이 포함되어 있는지 확인하는 패턴
    only_hangul = re.compile('[ㄱ-ㅎㅏ-ㅣ가-힣]+')
    # If there is data that contains other characters other than hangul, the findall returns a list of that.
    # Example: डीटीसी की जो बसें सु, or ờ á Đấ à ộ ự ệ ố ế
    # 오류문/교정문 모두 허용 외 문자가 없는 쌍만 남김
    data = [x for x in data if len(hangul.findall(x[0])) == 0 and len(hangul.findall(x[1])) == 0]
    # 오류문/교정문 모두 한글이 최소 1자 이상 있는 쌍만 남김
    data = [x for x in data if len(only_hangul.findall(x[0])) != 0 and len(only_hangul.findall(x[1])) != 0]
    # 필터링된 데이터 반환
    return data

def token_too_long(org, cor):
    # 오류문의 어절별 글자 수 배열 (주의: np import가 파일에 없음)
    orgs = np.array([len(x) for x in org.split(" ")])
    # 교정문의 어절별 글자 수 배열
    cor = np.array([len(x) for x in cor.split(" ")])
    # 두 문장 통틀어 20자를 넘는 어절이 있으면
    if max(max(orgs), max(cor)) > 20:
        # 너무 긴 어절이 있음
        return True
    # 그렇지 않으면 정상
    return False


def calc_vars(data):
    # KoBERT SentencePiece 모델 파일 경로 가져오기
    tok_path = get_tokenizer()
    # 토크나이저 객체 생성
    sp = SentencepieceTokenizer(tok_path)
    # 각 쌍마다 토큰 수/글자 수/원문을 담은 딕셔너리 생성
    txt = [{
        # 오류문·교정문의 토큰 수와 글자 수
        'len_token_o': len(sp(x[0])), 'len_token_c': len(sp(x[1])), 'len_char_o': len(x[0]), 'len_char_c': len(x[1]),
        #'jamo_levenshtein': jamo_levenshtein(x[0], x[1]), 
        # 원문 보관 및 이후 계산될 값의 자리 확보
        'o': x[0], 'c': x[1], 'token_div': None, 'char_div': None, 'token_min': None
        } for x in data]
    # 각 샘플의 비율 값 계산
    for info in txt:
        # 토큰 수 비율 (교정/오류)
        info['token_div'] = info['len_token_c']/info['len_token_o']
        # 글자 수 비율 (교정/오류)
        info['char_div'] = info['len_char_c']/info['len_char_o']
        # 두 문장 중 짧은 쪽의 토큰 수
        info['token_min'] = min(info['len_token_c'], info['len_token_o'])
        #info['token_log'] = math.log(info['token_min'], 20)
        #info['ratio'] = info['jamo_levenshtein'] / info['token_min'] * info['token_log']
    # apply cond
    try:
        # 불리언 인덱싱을 위해 numpy 객체 배열로 변환
        txt = np.array(txt, dtype=object)
    except:
        # 변환 실패 시 디버거 진입
        import pdb; pdb.set_trace()
    # 필터 조건: 토큰비 0.25~4, 글자비 0.5~1.25, 짧은 쪽 토큰 5 초과,
    # 교정문에 'or'/'good' 없음, 20자 초과 어절 없음 -> 모두 만족하면 True
    const = np.array([(0.25 < x['token_div'] < 4) & (0.5 < x['char_div'] < 1.25) & (x['token_min'] > 5)
        & ('or' not in x['c']) & ('good' not in x['c']) & (not token_too_long(x['o'], x['c'])) for x in txt])
    # 조건을 통과한 샘플만 앞뒤 공백 제거 후 [오류, 교정] 쌍으로 반환
    data = [[x['o'].strip(), x['c'].strip()] for x in txt[const]]
    return data

def cleanup_corpus(data):
    # 오류문: 구두점 앞뒤에 공백 삽입
    cleaned_src = [re.sub(r'([{}])'.format(string.punctuation),r' \1 ', x[0]) for x in data]
    # 연속 공백을 하나로 축소
    cleaned_src = [re.sub(' +', ' ', x) for x in cleaned_src]
    # 교정문: 구두점 앞뒤에 공백 삽입
    cleaned_tgt = [re.sub(r'([{}])'.format(string.punctuation), r' \1 ', x[1]) for x in data]
    # 연속 공백을 하나로 축소
    cleaned_tgt = [re.sub(' +', ' ', x) for x in cleaned_tgt]
    # '/'가 포함된 샘플 제거 후 (오류, 교정) 튜플로 묶기
    cleaned = [(s, t) for s, t in zip(cleaned_src, cleaned_tgt) if '/' not in s and '/' not in t]
    # remove samples with \
    # 쌍별 등장 횟수 집계
    lang8_count = Counter(cleaned)
    # 2회 이상 등장한 중복 쌍 목록
    first_dups = [pair for pair,cnt  in lang8_count.items() if cnt > 1]
    # 중복된 쌍의 종류 수 출력
    print(len(first_dups))
    # 고유 쌍 수 출력
    print('lang8 counter:', len(lang8_count.keys()))
    # set으로 중복 제거
    uniq = list(set(cleaned))
    # 중복 제거 후 개수 출력
    print('unique lang8 data:', len(uniq))
    # 중복 제거된 코퍼스 반환
    return uniq

def write(output, cleaned_unique_corpus):
    # 저장한 샘플 수 카운터
    cnt = 0
    # UTF-8로 출력 파일 열기
    with open(output, 'w', encoding='utf8') as f:
        # 헤더 행 기록
        f.write('src' + '\t' + 'tgt' + '\n')
        # 각 쌍 순회
        for pair in cleaned_unique_corpus:
            # 오류문과 교정문이 다른 경우만
            if pair[0] != pair[1]:
                # 탭으로 이어 한 줄로 기록
                f.write('\t'.join(pair).strip() + '\n')
                # 카운터 증가
                cnt += 1

    # 저장된 샘플 수 출력
    print('Saved dataset:', cnt)


# flatten datasets that have only one lines per each data sample.
def split_sentences(orig, cor):
    # 결과 리스트
    res = []
    # 오류/교정 문장 쌍 순회
    for o, c in zip(orig, cor):
        # 마침표가 2개 이상이면 여러 문장으로 구성된 샘플
        if o.count('.') > 1:
            # 마침표 기준으로 문장 분리
            orig_sents, cor_sents = o.split('.'), c.split('.')
            # 문장 수가 같을 때만 정렬 가능하므로 처리
            if len(orig_sents) == len(cor_sents):
                # 마지막 빈 조각을 제외하고 문장별로 순회
                for o_sub, c_sub in zip(orig_sents[:-1], cor_sents[:-1]):
                    # 수정이 있는 문장만
                    if o_sub != c_sub:
                        # 두 문장 모두 4자 초과인 경우만
                        if min(len(o_sub), len(c_sub)) > 4:
                            # 분리한 마침표를 다시 붙여 추가
                            res.append([o_sub + ' .', c_sub + ' .'])
        else:
            # 한 문장짜리는 그대로 추가
            res.append([o, c])
    # 문장 단위로 펼쳐진 결과 반환
    return res

def remove_misfunctioned_cases(data):
    # 원소가 정확히 2개(오류, 교정)인 항목만 유지
    data = ([d for d in data if len(d) == 2])
    # 오류문이 2자 초과인 항목만 유지
    data = [d for d in data if len(d[0]) > 2]
    # 교정문이 2자 초과인 항목만 유지하여 반환
    return [d for d in data if len(d[1]) > 2]


def only_leave_unique(data):
    # 오류문과 교정문이 동일한(수정 없는) 쌍 제거
    data = [x for x in data if x[0] != x[1]]
    return data

def cleanup_punct_and_brackets(cleaned_list):
    # 오류문에서 '^' 문자 제거
    orig = [x[0].replace('^', '') for x in cleaned_list]
    # 교정문에서 '^' 문자 제거
    cor = [x[1].replace('^', '') for x in cleaned_list]
    # remove (in order)
    # 1 ) 2 ) ....
    # (Inside paranthesis)
    # <Inside brackets>
    # Duplicates of ㄱ > ; ㅠ ㅜ ? ! . ~
    # 오류문에 위 정제 규칙을 안쪽(마지막 항목)부터 바깥쪽 순서로 적용 후 strip
    orig = [re.sub(r'( )(?=\1)', '', re.sub(r'\[.*?\]', '', re.sub(r'\<.*?\>', '', re.sub(r'(\d+ )\)', '', re.sub(r'\([^)]*\)', '', re.sub(r'(~)(?=\1)', '', re.sub(r'(ㄱ)(?=\1)', '', re.sub(r'( >)(?=\1)', '', re.sub(r'( ;)(?=\1)', '', re.sub(r'(ㅠ)(?=\1)', '', re.sub(r'(ㅜ)(?=\1)', '', re.sub(r'( ?)(?=\1)', '', re.sub(r'( !)(?=\1)', '', re.sub(r'( .)(?=\1)', '', x)))))))))))))).strip() for x in orig]
    # 교정문에도 동일한 정제 규칙 적용
    cor = [re.sub(r'( )(?=\1)', '', re.sub(r'\[.*?\]', '', re.sub(r'\<.*?\>', '', re.sub(r'(\d+ )\)', '', re.sub(r'\([^)]*\)', '', re.sub(r'(~)(?=\1)', '', re.sub(r'(ㄱ)(?=\1)', '', re.sub(r'( >)(?=\1)', '', re.sub(r'( ;)(?=\1)', '', re.sub(r'(ㅠ)(?=\1)', '', re.sub(r'(ㅜ)(?=\1)', '', re.sub(r'( ?)(?=\1)', '', re.sub(r'( !)(?=\1)', '', re.sub(r'( .)(?=\1)', '', x)))))))))))))).strip() for x in cor]
    # 정제된 오류문/교정문 리스트 반환
    return orig, cor

def final_cleanup_and_write(filename, res):
    # 원본 보관 (이후 사용되지 않음)
    orig = res
    # 탭이 없고 2자 이상인 쌍만 유지 (TSV 형식 보호)
    res = [x for x in res if ('\t' not in x[0] and '\t' not in x[1] and len(x[0]) >= 2 and len(x[1]) >= 2)]
    # 편집 거리 필터 전 개수 출력
    print("Length before levenshtein: ", len(res))
    # 자모 편집 거리가 10 초과인 쌍만 유지
    res = [x for x in res if jamo_levenshtein(x[0], x[1]) > 10]
    # 편집 거리 필터 후 개수 출력
    print("Length after levenshtein: ", len(res))
    # 각 쌍을 탭으로, 쌍 사이는 개행으로 연결
    data = '\n'.join(['\t'.join(x) for x in res])
    # 고정 파일명 kor_lang8.txt에 저장 (filename 인자는 사용되지 않음)
    with open('kor_lang8.txt', 'w') as f:
        f.write(data)




def main():
    # 커맨드라인 인자 파싱
    args = parse_args()

    print('1. Load data')
    # 입력 파일에서 오류문/교정문 로드
    org, cor = load_data(args.datafile)
    # 로드된 샘플 수 출력
    print('dataset length', len(org))

    print('2. Remove white spaces, misfunctioned cases, and other languages')
    # remove other languages
    # 오류문에서 일본어/중국어/전각 문자, 이어서 영문 알파벳 제거
    org = [re.sub('[a-zA-Z]+', '', re.sub(u'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]', '', x)) for x in org]
    # 교정문에도 동일하게 적용
    cor = [re.sub('[a-zA-Z]+', '', re.sub(u'[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf\u3400-\u4dbf]', '', x)) for x in cor]

    # 양끝 공백 제거
    org, cor = [x.strip() for x in org], [x.strip() for x in cor]
    # 너무 짧거나 형식이 잘못된 쌍 제거
    res = remove_misfunctioned_cases(zip(org, cor))
    # 남은 샘플 수 출력
    print("dataset length", len(res))

    print('3. Drop rows of other languages')
    # 허용되지 않은 문자가 있거나 한글이 없는 쌍 제거
    res = drop_other_lang(res)

    print('4. Calculate variables and apply conditions')
    # 토큰/글자 비율 등 조건으로 필터링
    res = calc_vars(res)
    # 남은 샘플 수 출력
    print("dataset length", len(res))

    print('5. Save cleaned up data')
    # 구두점 띄어쓰기 정규화 및 중복 제거
    cleaned_list = cleanup_corpus(res)
    # changed to list
    # 괄호/중복 기호 등 정리
    orig, cor = cleanup_punct_and_brackets(cleaned_list)
    # 다문장 샘플을 문장 단위로 분리
    res = split_sentences(orig, cor)
    # 수정 없는 쌍 제거
    res = only_leave_unique(res)
    # 편집 거리 필터링 후 파일 저장
    res = final_cleanup_and_write('lang8.txt', res)
    # 종료 메시지
    print('Done!')

# 스크립트로 직접 실행될 때만 main 호출
if __name__ == '__main__':
    main()

