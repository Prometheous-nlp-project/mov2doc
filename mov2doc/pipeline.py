from stt import stt_from_url
from punctuate import punctuate
from keyword import extract_keywords
from ctrlsum import ctrl_sum
from search import sementic_search

from time import time
from nltk import sent_tokenize


def pipeline(data, func_type="url", query=None, log=False):
    msg = ""
    if query == "":
        query = None
    s = time()
    if func_type == "url":
        try:
            text = stt_from_url(data)
        except Exception as e:
            return "", "", "", "", "", "Error: "+e

        text = punctuate([text])
        if any(len(sent.split()) > 32 for sent in text[0].split(". ")):
            msg += "Warning: STT performance might be low.\n"
    else:
        text = [data]

    keywords = extract_keywords(text)
    summarization = ctrl_sum(text[0])
    if query is not None:
        search_result = sementic_search(text[0], query)
        ctrlsum_qa = ctrl_sum(text, query)

    text = "\n".join(sent_tokenize(text[0]))
    keywords = "\n".join(
        map(lambda x: str(x[0]+1)+". "+x[1], enumerate(keywords[0])))
    if log:
        print("\ntext:")
        print(text)
        print()

        print("keywords:")
        print(keywords)
        print()

        print("summarization:")
        print(summarization)
        print()

    if query is not None:
        if log:
            print(f"content related to \"{query}\":")
            print(search_result)
            print()

            print(f"ctrlsum answer to \"{query}\"")
            print(ctrlsum_qa)
            print()
        return text, keywords, summarization, search_result, ctrlsum_qa, msg + f"Success: Time elapsed: {round(time()-s,4)}s"
    else:
        return text, keywords, summarization, "", "", msg + f"Success: Time elapsed: {round(time()-s,4)}s"
