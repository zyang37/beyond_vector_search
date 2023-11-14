import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from pprint import pprint

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

def remove_noise_from_keywords(keywords: list):
    stop_words = set(stopwords.words('english'))
    punctuations = set(['.', ',', '!', '?', ';', ':', '-', '(', ')', '[', ']', '{', '}', '\'', '\"', '`', '~', '@', '#', '$', '%', '^', '&', '*', '_', '+', '=', '<', '>', '/', '\\', '|'])
    # remove stop words
    keywords = [word for word in keywords if not word in stop_words]
    # remove if contains punctuations
    keywords = [word for word in keywords if not any(p in word for p in punctuations)]
    return keywords

def extract_keywords(text: str, score: bool = True):
    r = Rake()
    # Extraction given the text.
    text = text.lower()
    r.extract_keywords_from_text(text)
    if score:
        return r.get_ranked_phrases_with_scores()
    return r.get_ranked_phrases()

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