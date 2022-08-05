from typing import Optional, Tuple, Dict, List, Any

from transformers import (
    PreTrainedModel,
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification
)

import re
import torch as th
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict

model_name = '../models/' + 'seg-model-distilBERT-finetuned'


def _load_sent_seg_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    return tokenizer, model, data_collator


def _tokenize_and_align_labels(tokenizer, papers: DatasetDict, label_pos: str = 'last') -> DatasetDict:
    """
    Since the labels are not aligned with the tokens due to subwords, this method aligns them.

    Args:
        papers: DatasetDict of papers
        label_pos: position of the is_eos label in the tokens
    Returns:
        DatasetDict of papers with aligned labels
    """
    assert label_pos in ['all', 'first', 'last']

    tokenized_inputs = tokenizer(papers['source'], truncation=True)
    labels = []
    for i, label in enumerate(papers['is_eos']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        if label_pos == 'last':
            word_ids = word_ids[::-1]

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_pos == 'all' else -100)
            previous_word_idx = word_idx

        labels.append(label_ids[::-1] if label_pos == 'last' else label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def _clean(text: str) -> str:
    regex = [(r'\n+', ' '),
             (r'\([^\(\)]*\)|\[[^\[\]]*\]', ' '),
             (r'\$.*\$', ' '),
             (r'\\[^\s]+', ' '),
             (r'\d+\.\d+', ' '),
             (r'[^a-zA-Z\. ]', ' '),
             (r' *\. *', '. '),
             (r' +', ' ')]

    cleaned = text
    for pattern, repl in regex:
        cleaned = re.sub(pattern, repl, cleaned)

    return cleaned.strip()


def _construct_dataset(tokenizer, texts: List[str], puncuated=True):
    clean_texts = [_clean(text).lower().strip() for text in texts]
    if puncuated:
        dataset = Dataset.from_dict({'original': clean_texts,
                                     'source': [text.replace('.', '') for text in clean_texts],
                                     'is_eos': [[int(w[-1] == '.') for w in text.split()] for text in clean_texts]})

        dataset = dataset.map(_tokenize_and_align_labels, batched=True)
        dataset = dataset.remove_columns(['original', 'source', 'is_eos'])

    else:
        dataset = Dataset.from_dict(tokenizer(clean_texts, truncation=True))

    return dataset


def punctuate(texts: List[str]) -> List[str]:
    """
    Args:
        texts: STT완료한 non-puncuated 문장들, 꼭 List[str] 형식으로 넣어줄 것.
    Returns:
        주어진 texts들에 대해 온점을 찍은 문장들
    """
    tokenizer, model, data_collator = _load_sent_seg_model(model_name)
    dataset = _construct_dataset(tokenizer, texts, puncuated=False)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            collate_fn=data_collator)

    def insert_punct(token_ids, preds):
        args = th.argwhere(preds == 1)
        subwords = tokenizer.convert_ids_to_tokens(token_ids.squeeze())
        for arg in args:
            subwords[arg] += '.'
        return subwords

    pred_texts = []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        pred = th.argmax(logits, dim=2).squeeze()

        pred_text = insert_punct(input_ids, pred)
        pred_text = ' '.join(pred_text[1:-1]).replace(' ##', '')
        pred_texts.append(pred_text)

    return pred_texts
