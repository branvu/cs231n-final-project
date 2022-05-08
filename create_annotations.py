'''
THIS FILE IS FOR CREATING ANNOTATIONS
- takes one argument for language to create annotations for (can be "all" or specific language)
- will generate and save annotions
- Format: image_name, language_id, letter_label
'''

import os
import pandas as pd
import argparse

'''
UPDATE THESE CONSTANTS AS NEEDED
'''
ANNOTATIONS_DIR = "annotations/"
ANNOTATIONS_FILE = "annotations.csv"
LANGUAGE_INFO = {"asl": ("data/asl", ANNOTATIONS_DIR + "asl_annotations.csv", "0"), "jsl": ("data/jsl", ANNOTATIONS_DIR + "jsl_annotations.csv", "1"), "isl": (
    "data/isl", ANNOTATIONS_DIR + "isl_annotations.csv", "2"), "arsl": (
    "data/arsl", ANNOTATIONS_DIR + "arsl_annotations.csv", "3")}

available_langs = [lang for lang in LANGUAGE_INFO]

parser = argparse.ArgumentParser()
parser.add_argument(
    "lang", help=f"Specified language to create annotations for, can be 'all' or specific language. Available languages: {available_langs}")
args = parser.parse_args()


def generate(language):
    path, filename, lang_id = LANGUAGE_INFO[language]

    images = os.listdir(path)
    data = [[path + "/" + x, lang_id, x.split('-')[1]] for x in images]

    df = pd.DataFrame(data,
                      columns=["image", "language", "letter"])

    df.to_csv(ANNOTATIONS_DIR + ANNOTATIONS_FILE,
              mode="a", index=False, header=False)
    print(f"Added annotions for {language} to {ANNOTATIONS_FILE}")

    df.to_csv(filename, index=False)

    print(f"Created annotations for {language} in {filename}")


def main():
    if args.lang == "all":
        df = pd.DataFrame(columns=["image", "language", "letter"])
        df.to_csv(ANNOTATIONS_DIR + ANNOTATIONS_FILE, index=False)
        for lang in LANGUAGE_INFO:
            generate(lang)
    elif args.lang in available_langs:
        if ANNOTATIONS_FILE not in os.listdir(ANNOTATIONS_DIR):
            print("Created annotations file")
            df = pd.DataFrame(columns=["image", "language", "letter"])
            df.to_csv(ANNOTATIONS_DIR + ANNOTATIONS_FILE, index=False)
        generate(args.lang)
    else:
        raise Exception("Invalid language")


if __name__ == "__main__":
    main()
