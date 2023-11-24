from torch.utils.data import Dataset
import transformers as t
import datasets as d


class DatasetReader(Dataset):
  def __init__(self, ds, tr):
    with open('./prompt.txt', 'r') as file:
      self.template = file.read()
    self.tokenizer = tr
    self.tokenizer.pad_token_id = 0
    self.tokenizer.padding_side = "right"
    self.ds = ds
    self.ds = self.ds.map(self.prompt, remove_columns=["question", "context", "answer"], load_from_cache_file=False, num_proc=8)
    self.ds = self.ds.map(self.tokenize, remove_columns=["prompt"], load_from_cache_file=False, num_proc=8)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, idx):
    return self.ds[idx]

  def prompt(self, elm):
    prompt = self.template.format(question=elm["question"], context=elm["context"])
    prompt = prompt + elm["answer"]
    return {"prompt": prompt}

  def tokenize(self, elm):
    res = self.tokenizer(elm["prompt"])
    res["input_ids"].append(self.tokenizer.eos_token_id)
    res["attention_mask"].append(1)
    res["labels"] = res["input_ids"].copy()
    return res
    # res["labels"] = res["input_ids"].copy()
    # res["labels"].append(self.tokenizer.eos_token_id)
    # res["labels"] = res["labels"][1:]
    # return res

  def max_seq_len(self):
    return max([len(elm["input_ids"]) for elm in self.ds])