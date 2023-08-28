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

        # 멀티프로세싱을 위해 인자를 준비합니다.
        arguments = [(i, filepaths_and_text[i][args.text_index], args.text_cleaners) for i in
                     range(len(filepaths_and_text))]

        # tqdm을 사용하여 병렬 처리 진행 상황을 표시합니다.
        cleaned_texts = {idx: text for idx, text in
                         tqdm(pool.imap_unordered(clean_text, arguments), total=len(arguments))}

        for i in range(len(filepaths_and_text)):
            filepaths_and_text[i][args.text_index] = cleaned_texts[i]

        # 나머지 코드는 동일합니다.
        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

    # Pool 닫기
    pool.close()
    pool.join()
