import torch
from summarizers import Summarizers

device = "cuda" if torch.cuda.is_available() else "cpu"
summ = Summarizers(device=device)


def ctrl_sum(text, query=None):
    if query is not None:
        return summ(text, query=query)
    else:
        return summ(text)
