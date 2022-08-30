from keybert import KeyBERT
import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

extractor = KeyBERT()


def extract_keywords(doc: list):
    '''
    Args
      doc : list -> contains a paragraph for each element

    Returns
      keywords : list -> contains keywords for each paragraph
    '''

    def is_noun(pos): return pos[:2] == 'NN'  # checks if nltk postag is Noun
    only_nouns = []  # paragraphs with only nouns
    # extracting nouns from each paragraph
    for paragraph in doc:
        tokenized = nltk.word_tokenize(paragraph)
        only_nouns.append(
            ' '.join([word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]))

    keywords_list = []

    # extracting keywords from each noun paragraph
    for paragraph in doc:
        keywords_list.append(extractor.extract_keywords(paragraph, stop_words='english',
                             keyphrase_ngram_range=(1, 2), top_n=5, nr_candidates=20, use_maxsum=True))

    # extracting only keywords from tuple
    keywords = []
    for i in range(len(keywords_list)):
        keywords.append([k[0] for k in keywords_list[i]])

    return keywords
