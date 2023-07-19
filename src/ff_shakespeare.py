import numpy as np
import os, torch, urllib.request

from src import utils

def download(url, path):
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    print(f"Downloading {url} to {path}")
    urllib.request.urlretrieve(url, path)

def download_tiny_shakespeare(path):
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    download(url, path)

def get_shakespeare_partition(opt, partition) -> str:
    file_path = utils.get_original_cwd() + "/datasets/shakespeare/input.txt"
    if not os.path.exists(file_path):
        download_tiny_shakespeare(file_path)
    
    with open(file_path, "r") as f:
        text = f.read()

    if partition == "train":
        text = text[: int(0.8 * len(text))]
    elif partition == "val":
        text = text[int(0.8 * len(text)) : int(0.9 * len(text))]
    elif partition == "test":
        text = text[int(0.9 * len(text)) :]
    else:
        raise NotImplementedError
    
    return text

class FF_Shakespeare(torch.utils.data.Dataset):
    def __init__(self, opt, partition):
        self.opt = opt
        self.text = get_shakespeare_partition(opt, partition)
        self.tokens = sorted(set(self.text))
        self.stoi = {c: i for i, c in enumerate(self.tokens)}
        self.num_classes = len(self.tokens)

    def __getitem__(self, index):
        pos_sample, neg_sample, neutral_sample, target = self._generate_sample(
            index
        )

        inputs = {
            "pos": pos_sample,
            "neg": neg_sample,
            "neutral": neutral_sample,
        }
        labels = {"label": target}
        return inputs, labels

    def __len__(self):
        return len(self.text) - self.opt.input.shakespeare.sample_len

    def _get_pos_sample(self, sample, class_label):
        return sample

    def _get_neg_sample(self, sample, class_label):
        # choose random sample from self.tokens of same length as sample
        neg_sample = torch.randint_like(sample, len(self.tokens))
        return neg_sample

    def _get_neutral_sample(self, sample):
        return sample
    
    def char_to_index(self, char):
        return self.stoi[char]

    def _generate_sample(self, index):
        sample_len = self.opt.input.shakespeare.sample_len
        sample = self.text[index: index+sample_len+1]
        sample, class_label = sample[:-1], sample[-1]
        sample = torch.tensor([self.char_to_index(c) for c in sample], dtype=torch.float32)
        class_label = self.char_to_index(class_label)
        
        pos_sample = self._get_pos_sample(sample, class_label)
        neg_sample = self._get_neg_sample(sample, class_label)
        neutral_sample = self._get_neutral_sample(sample)
        return pos_sample, neg_sample, neutral_sample, class_label
