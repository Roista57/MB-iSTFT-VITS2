import argparse
import text
from tqdm import tqdm
from utils import load_filepaths_and_text
from multiprocessing import Pool, cpu_count


def clean_text(args):
    idx, original_text, text_cleaners = args
    cleaned_text = text._clean_text(original_text, text_cleaners)
    return idx, cleaned_text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_extension", default="cleaned")
    parser.add_argument("--text_index", default=1, type=int)
    parser.add_argument("--filelists", nargs="+",
                        default=["filelists/filelist_train.txt", "filelists/filelist_val.txt"])
    parser.add_argument("--text_cleaners", nargs="+", default=["korean_cleaners"])
    args = parser.parse_args()

    # CPU 코어의 수만큼 프로세스를 사용합니다.
    num_workers = cpu_count()
    pool = Pool(num_workers)

    for filelist in args.filelists:
        print("START:", filelist)
        filepaths_and_text = load_filepaths_and_text(filelist)

        # 최초 1회 단일 쓰레드로 clean
        first_idx, first_text, first_cleaners = 0, filepaths_and_text[0][args.text_index], args.text_cleaners
        filepaths_and_text[0][args.text_index] = text._clean_text(first_text, first_cleaners)

        # 그 다음부터는 멀티프로세싱 사용
        arguments = [(i, filepaths_and_text[i][args.text_index], args.text_cleaners) for i in
                     range(1, len(filepaths_and_text))]  # 첫 번째 항목은 제외하고 시작

        cleaned_texts = {idx: text for idx, text in
                         tqdm(pool.imap_unordered(clean_text, arguments), total=len(arguments))}

        for i in range(1, len(filepaths_and_text)):  # 첫 번째 항목은 이미 처리했으므로 제외하고 시작
            filepaths_and_text[i][args.text_index] = cleaned_texts[i-1]  # 첫 번째 항목은 이미 처리했으므로 인덱스 조정

        # 나머지 코드는 동일합니다.
        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

    # Pool 닫기
    pool.close()
    pool.join()
