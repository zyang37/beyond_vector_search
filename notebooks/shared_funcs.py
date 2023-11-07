from collections import Counter


def parse_authors(author_str):
    authors = []
    # remove noises
    author_str = author_str.replace(" ", "")
    author_str = author_str.replace("\n", "")
    parse_by_comma = author_str.split(",")
    for a in parse_by_comma:
        # parse by 'and'
        authors.extend(a.split(" and "))
    return authors


def parse_categories(cat_str):
    return cat_str


def parse_year(date_str):
    return date_str.split("-")[0]


def count_name_frequencies(name_list2d):
    # Flatten the list of lists into a single list of names
    all_names = [name for sublist in name_list2d for name in sublist]
    # Use Counter to count the frequency of each name
    name_frequencies = Counter(all_names)
    return dict(name_frequencies)


def merge_add_dict(a, b):
    return {key: a.get(key, 0) + b.get(key, 0) for key in set(a) | set(b)}


def gather_stats(df):
    intermediate_submitter_stats = dict(df["submitter"].value_counts())
    intermediate_authors_stats = count_name_frequencies(
        list(map(parse_authors, df["authors"]))
    )
    intermediate_authors_stats = merge_add_dict(
        intermediate_submitter_stats, intermediate_authors_stats
    )
    intermediate_cat_stats = dict(df["categories"].value_counts())
    # intermediate_year_stats = dict(df['update_date'].value_counts())
    return intermediate_authors_stats, intermediate_cat_stats
