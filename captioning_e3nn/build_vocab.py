import pickle
import argparse
import json
import sys
import os
from collections import Counter

MAX_Length = 245

args = str(sys.argv[1])
print(args)

with open(args) as json_file:
    config = json.load(json_file)


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx["<unk>"]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def build_vocab(config):
    dir_path = config["preprocessing"]["path_proteins"]
    files_pr = os.listdir(dir_path)
    files_pr.remove(".DS_Store")
    max = 0
    counter = Counter()
    for file in files_pr:
        print(file)

        # if file in files_exceptions:
        #     shutil.rmtree(os.path.join(dir_path, file))
        path_to_smile = os.path.join(dir_path, file, file + "_ligand.smi")
        #     path_to_smile_csv = os.path.join(dir_path, file,file + "_ligand.txt")

        with open(path_to_smile, "r") as file:
            data = file.read()
        tokens = [token for token in data]
        counter.update(tokens)
    words = [word for word, cnt in counter.items()]
    vocab = Vocabulary()
    vocab.add_word("<pad>")
    vocab.add_word("<start>")
    vocab.add_word("<end>")

    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


def main(args):
    vocab = build_vocab(config)
    vocab_path = config["preprocessing"]["vocab_path"]
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print("Total vocabulary size: {}".format(len(vocab)))
    print("Saved the vocabulary wrapper to '{}'".format(vocab_path))


if __name__ == "__main__":
    main(args)
