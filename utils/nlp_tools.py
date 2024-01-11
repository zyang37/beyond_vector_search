import sys
import nltk
import numpy as np
from tqdm.auto import tqdm

sys.path.append("../")
from utils.parse_arxiv import parse_title
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')

def sort_dict_by_key(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}

def sort_dict_by_val(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1], reverse=True)}

def token_limited_sentences(text, max_num_tokens=256):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Initialize an empty list to store processed sentences
    processed_sentences = []
    current_sentence = ""
    for sentence in sentences:
        # Tokenize the sentence into words
        tokenized_sentence = word_tokenize(sentence)

        # Check if adding this sentence exceeds the token limit
        if len(word_tokenize(current_sentence)) + len(tokenized_sentence) <= max_num_tokens:
            current_sentence += " " + sentence
        else:
            # Add the current sentence to the list if it's not empty
            if current_sentence:
                processed_sentences.append(current_sentence.strip())
            # Start a new sentence
            current_sentence = sentence

    # Add the last sentence if it's not empty
    if current_sentence:
        processed_sentences.append(current_sentence.strip())

    return processed_sentences
