'''
Extract keywords from a given text using Rake and spaCy.
'''

import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from pprint import pprint
from nltk.tokenize import sent_tokenize, word_tokenize

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def compute_tfidf_score(article, noun_chunk):
    '''
    Compute a TF-IDF score for noun chunk with respect to the article
    '''
    # loop each token in the noun chunk, compute the tf-idf score, 
    # then return a average score
    tfidf_scores = []
    for token in word_tokenize(noun_chunk):
        tf = article.count(token) / len(article)
        try:
            idf = np.log(len(article) / article.count(token))
        except:
            # if the token is not in the article, skip
            continue
        tfidf_scores.append(tf * idf)
    return np.mean(tfidf_scores)

def remove_noise_from_keywords(keywords: list, max_len: int = 30):
    stop_words = set(stopwords.words('english'))
    punctuations = set(['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', '\'', '\"', '`', '~', '@', '#', '$', '%', '^', '&', '*', '_', '+', '=', '<', '>', '/', '\\', '|'])
    # remove stop words
    keywords = [word for word in keywords if not word in stop_words]
    # remove if contains punctuations
    keywords = [word for word in keywords if not any(p in word for p in punctuations)]
    return keywords

def extract_keywords_spacy(text: str, score: bool = True, min_token: int = 2, remove_articles: bool = True):
    '''
    Extract keywords from the given text using spaCy (for long text)
    '''
    text = text.strip()
    doc = nlp(text)
    noun_chunk_list = list(set([chunk.text for chunk in doc.noun_chunks]))
    if remove_articles:
        # remove articles
        noun_chunk_list = [noun_chunk for noun_chunk in noun_chunk_list if not noun_chunk.startswith('the ')]
        noun_chunk_list = [noun_chunk for noun_chunk in noun_chunk_list if not noun_chunk.startswith('a ')]
        noun_chunk_list = [noun_chunk for noun_chunk in noun_chunk_list if not noun_chunk.startswith('an ')]
    noun_chunk_list = [noun_chunk for noun_chunk in noun_chunk_list if len(word_tokenize(noun_chunk)) > (min_token-1)]
    noun_chunk_list = remove_noise_from_keywords(noun_chunk_list)

    # sort the noun_chunk_list by length, make [(score, word), ...]
    noun_chunk_list = [(compute_tfidf_score(text, noun_chunk)*len(noun_chunk), noun_chunk) for noun_chunk in noun_chunk_list]
    # noun_chunk_list = [(len(noun_chunk), noun_chunk) for noun_chunk in noun_chunk_list]
    noun_chunk_list.sort(reverse=True)
    if score:
        return noun_chunk_list
    return [noun_chunk[1] for noun_chunk in noun_chunk_list]

def extract_keywords(text: str, score: bool = True, spacy: bool = False):
    '''
    Extract keywords from the given text using Rake (for shorter text)
    '''
    ret_list = []
    # Extraction given the text.
    text = text.lower()
    if spacy:
        ret_list = extract_keywords_spacy(text, score=score)
    else:
        r = Rake()
        r.extract_keywords_from_text(text)
        if score:
            ret_list = r.get_ranked_phrases_with_scores()
        else:
            ret_list = r.get_ranked_phrases()
    
    if len(ret_list) == 0:
        # if no keywords are extracted, raise an error
        raise ValueError("No keywords are extracted from the given text")
        
    return ret_list

# UNUSED!
def batch_extract_keywords(texts: list, score: bool = True):
    '''
    NOTE! the results are all combined into one list!!!
    '''
    r = Rake()
    # remove noise from each text
    texts = [text.lower() for text in texts]

    # Extraction given the text.
    r.extract_keywords_from_sentences(texts)
    if score:
        return r.get_ranked_phrases_with_scores()
    return r.get_ranked_phrases()


# Example
if __name__ == '__main__':
    print("This is an example\n")
    r = Rake()

    sample = """spaCy is an open-source software library for advanced natural language processing, written in the programming languages Python and Cython. The library is published under the MIT license and its main developers are Matthew Honnibal and Ines Montani, the founders of the software company Explosion."""

    # Extraction given the text.
    print(sample)
    print()
    pprint(extract_keywords(sample, score=True))
    print()
    