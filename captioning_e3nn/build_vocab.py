import argparse
import json
import os
import pickle
import sys
from collections import Counter
import argparse
import config

MAX_Length = 245

# args = str(sys.argv[1])
# print(args)

# with open(args) as json_file:
#     config = json.load(json_file)
# Arguments
# parser = argparse.ArgumentParser(
#     description='Train a 3D reconstruction model.'
# )
# parser.add_argument('config', type=str, help='Path to config file.')

# args = parser.parse_args()
# cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')

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

import re

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens

def build_vocab(cfg):
    # dir_path = config["preprocessing"]["path_proteins"]
    dir_path = cfg['data']['path_refined']
    files_pr = os.listdir(dir_path)
    # files_pr.remove(".DS_Store") #for my mac
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
            print(data)
        
        # tokens = [token for token in data]
        tokens = smi_tokenizer(data)
        counter.update(tokens)
    print("counter", counter)
    words = [word for word, cnt in counter.items()]
    vocab = Vocabulary()
    # vocab.add_word("pad>")
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
    parser = argparse.ArgumentParser(
    description='Train a 3D reconstruction model.'
)
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()
    
     
    cfg = config.load_config(args.config, 'configurations/config_lab/default.yaml')
    # with open(vocab_path, "r") as f:
    #     v = pickle.load(f) 
    # v = pickle.load( open( vocab_path, "rb" ) )   
    # print("vocab", v)
    # build_vocab(cfg)
    vocab = build_vocab(cfg)
    vocab_path = cfg['preprocessing']['vocab_path']
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)

