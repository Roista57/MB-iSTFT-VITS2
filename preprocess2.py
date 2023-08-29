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

    num_workers = cpu_count()
    pool = Pool(num_workers)

    for filelist in args.filelists:
        print("START:", filelist)
        filepaths_and_text = load_filepaths_and_text(filelist)

        first_idx, first_text, first_cleaners = 0, filepaths_and_text[0][args.text_index], args.text_cleaners
        filepaths_and_text[0][args.text_index] = text._clean_text(first_text, first_cleaners)

        arguments = [(i, filepaths_and_text[i][args.text_index], args.text_cleaners) for i in
                     range(1, len(filepaths_and_text))]

        cleaned_texts = {idx: text for idx, text in
                         tqdm(pool.imap_unordered(clean_text, arguments), total=len(arguments))}

        for i in range(1, len(filepaths_and_text)):
            filepaths_and_text[i][args.text_index] = cleaned_texts[i]

        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

    pool.close()
    pool.join()
