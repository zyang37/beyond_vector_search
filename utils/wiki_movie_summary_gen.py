import pandas as pd
import pickle
import math
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import requests
from time import sleep
import multiprocessing
import random

# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
# model_name = "csebuetnlp/mT5_multilingual_XLSum"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

def sort_dict_by_key(d):
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[0])}

def dictVal_to_list(d):
    extend_list = []
    d = sort_dict_by_key(d)
    for k, v in d.items():
        extend_list.extend(v[:])
    return extend_list

def checkpoint_save(obj, path='../data/wiki_movies/checkpoint.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def wiki_scrape(url):
    # url = filtered_data['Wiki Page'].iloc[0]
    # response = requests.get(url)
    max_retries = 5
    retry_delay = 3  # seconds

    for attempt in range(max_retries):
        try:
            response = requests.get(url)
            # If the request is successful, break out of the loop
            break
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            sleep(retry_delay)
    else:
        # This else block executes if the loop completes without breaking
        print("All retries failed.")


    soup = BeautifulSoup(response.content, 'html.parser')
    # title = soup.find('h1', class_='firstHeading').text
    # Find and print the first paragraph
    # first_paragraph = soup.find('p').text
    # get short
    short = ''
    paragraphs = soup.find_all('p')
    for p in paragraphs:
        if len(p.text) >= 20:
            short = p.text
            break
    return short

def batch_wiki_scrape(args):
    i, urls, summary_dict = args
    short_list = []
    for url in tqdm(urls):
        short = wiki_scrape(url)
        # print(short)
        short_list.append(short)
        sleep(random.randint(0, 2))
    summary_dict[i] = short_list
    summary_ckpt = dictVal_to_list(summary_dict.copy())
    # print(sort_dict_by_key(summary_dict).keys())
    checkpoint_save(summary_ckpt)
    print("found {}".format(len(summary_ckpt)))

# def batch_summary_gen(model, tokenizer, article_texts, device):
#     summary = []
#     input_ids = tokenizer(
#         [WHITESPACE_HANDLER(text) for text in article_texts],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=512
#     )["input_ids"]
#     output_ids = model.generate(
#         input_ids=input_ids.to(device),
#         max_length=64,
#         no_repeat_ngram_size=2,
#         num_beams=4
#     )
#     for ids in output_ids:
#         summary.append(tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False))
#     return summary


if __name__ == "__main__":
    data = pd.read_csv('../data/wiki_movie_plots_deduped.csv')
    filtered_data = data[data['Genre'] != 'unknown']
    filtered_data = filtered_data[filtered_data['Cast'].notna()]
    filtered_data = filtered_data[filtered_data['Director'] != 'Unknown']
    # ckpt_path = '../data/wiki_movies/checkpoint.pkl'

    # filtered_data = filtered_data.sample(10)
    # batch_size = 64
    # summary_list = []
    # for i in tqdm(range(0, filtered_data.shape[0], batch_size)):
    #     plots = list(filtered_data['Plot'].iloc[i:i+batch_size])
    #     summaries = batch_summary_gen(model, tokenizer, plots, device)
    #     print(summaries)
    #     summary_list.extend(summaries)
    # filtered_data["summary_text"] = summary_list
    # filtered_data.to_csv('../data/wiki_movies/filtered_wiki_movies.csv')
    
    manager = multiprocessing.Manager()
    summary_dict = manager.dict()

    # for idx, row in tqdm(filtered_data.iterrows(), total=len(filtered_data)):
    #     url = row['Wiki Page']
    #     short = wiki_scrape(url)
    #     print(short)
    #     print(row['Plot'])
    #     print()
    #     summary_list.append(short)
    #     sleep(1)

    batch_size = 500
    urls_batch_args = []
    urls = list(filtered_data['Wiki Page'])
    for i, l in enumerate(range(0, len(urls), batch_size)):
        urls_batch_args.append((i, urls[l:l+batch_size], summary_dict))
    
    with multiprocessing.Pool(processes=32) as pool:
        pool.map(batch_wiki_scrape, urls_batch_args)
    
    summary_list = dictVal_to_list(summary_dict.copy())
    print(len(summary_list))
    filtered_data["scraped_summary"] = summary_list
    filtered_data.to_csv('../data/wiki_movies/filtered_wiki_movies_scraped.csv')
