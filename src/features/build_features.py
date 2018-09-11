import pkg_resources

ROOT_DIR = pkg_resources.resource_filename("src", "..")


def get_glove():
    import os
    import requests
    import zipfile

    data_external = "data/external"

    resp = requests.get("http://nlp.stanford.edu/data/glove.6B.zip")
    print("glove.6B downloaded", end="... ")
    zip_file = os.path.join(ROOT_DIR, data_external, "glove.6B.zip")
    with open(zip_file, 'wb') as f:
        f.write(resp.content)
    print("saved", end="... ")
    zip_ref = zipfile.ZipFile(zip_file, 'r')
    zip_ref.extractall(os.path.join(ROOT_DIR, data_external, "glove.6B"))
    zip_ref.close()
    print("extracted.")


def make_idx2vec(vocab_size, embed_dim, glove_file):
    import torch
    import os

    idx2vec = torch.zeros(vocab_size, embed_dim, dtype=torch.float)
    idx2vec[1] = torch.ones(embed_dim)
    idx2vec[2] = -torch.ones(embed_dim)
    i = 3
    with open(glove_file, 'rb') as f:
        for l in f:
            line = l.decode().split()
            idx2vec[i] = torch.tensor([float(x) for x in line[1:]], dtype=torch.float)
            i += 1

    torch.save(idx2vec, os.path.join(ROOT_DIR, "data/processed/idx2vec.pt"))
    print("idx2vec.pt saved")


def make_emb_ids(df, word2idx):
    import numpy as np
    import pandas as pd

    __UNKWORD__ = 1
    __SEPARATOR__ = 2

    special_chars = ["!", "?", "*", "_", "^", "&amp;", "%", "$", "@", "~", "(", ")", "[", "]",
                     "{", "}", "/", "\\", "|", '"', "`", "''"]

    def text2word_ids(s):
        s += " "

        # loop to surround special characters with spaces
        for c in special_chars:
            s = s.replace(c, " " + c + " ")

        s = s.replace("` `", " `` ")

        s = s.replace("´", "'")

        s = s.replace(". ", " . ")
        s = s.replace(".. . ", " ... ")

        s = s.replace(", ", " , ")
        s = s.replace(": ", " : ")
        s = s.replace("; ", " ; ")

        s = s.replace("' ", " ' ")
        s = s.replace(" '", " ' ")

        s = s.replace("n't", "nt")
        s = s.replace("s'", "s")
        s = s.replace("'s", "s")

        return [word2idx.get(x, __UNKWORD__) for x in s.lower().split()]

    id_df = pd.DataFrame(index=df.index)

    id_df["title_ids"] = df.title.map(text2word_ids)
    id_df["desc_ids"] = df.description.fillna("__UNKWORD__").map(text2word_ids)
    id_df["ids"] = id_df.apply(lambda x: list(np.concatenate([x.title_ids, [__SEPARATOR__], x.desc_ids])), axis=1)
    id_df["len_x"] = id_df.ids.map(len)
    id_df["y"] = df.y

    return id_df[["ids", "len_x", "y"]]


def main():
    import os
    import pandas as pd

    glove_file = os.path.join(ROOT_DIR, 'data/external/glove.6B/glove.6B.50d.txt')
    if not os.path.exists(glove_file):
        get_glove()

    idx2word = ['__PADWORD__', '__UNKWORD__', '__SEPARATOR__']
    with open(glove_file, 'r') as f:
        l = f.readline().split()
        embed_dim = len(l) - 1  # == 50
        idx2word.append(l[0])
        for l in f:
            idx2word.append(l.split()[0])
    vocab_size = len(idx2word)
    word2idx = dict(zip(idx2word, range(vocab_size)))

    train_df = pd.read_pickle(os.path.join(ROOT_DIR, "data/processed/train_df.pkl"))
    train_features_df = make_emb_ids(train_df, word2idx)
    train_features_df.to_pickle(os.path.join(ROOT_DIR, "data/processed/train_features_df.pkl"))

    test_df = pd.read_pickle(os.path.join(ROOT_DIR, "data/processed/test_df.pkl"))
    test_features_df = make_emb_ids(test_df, word2idx)
    test_features_df.to_pickle(os.path.join(ROOT_DIR, "data/processed/test_features_df.pkl"))

    if not os.path.exists(os.path.join(ROOT_DIR, "data/processed/idx2vec.pt")):
        make_idx2vec(vocab_size, embed_dim, glove_file)


if __name__ == "__main__":
    main()

    # idx2vec = torch.load(os.path.join(ROOT_DIR, "data/processed/idx2vec.pt"))
