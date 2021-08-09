from Bio import Entrez, Medline
import time
import csv


def get_pubmed_ids_from_csv(ids_csv_filepath):
    """
    Function retrieves PubMed ids from a csv file

    :param ids_csv_filepath: String - File path to CSV file containing PubMed IDs
    :return lines: List of PubMed article IDs
    """
    with open(ids_csv_filepath, 'r') as file:
        lines = file.read().split('\n')

    return lines

# 'C:/Users/melan/Google Drive/Mike\'s Documents/Projects/Milestone2/pubmed_ids_from_search.txt'


def extract_pubmed_articles(pubmed_ids):
    """
    Takes a list of PubMed IDs and extracts each corresponding article from PubMed
    Each article is slept every third of a second to avoid exceeding the PubMed URL Request limit of 3 per second
    :param pubmed_ids: List of Pubmed IDs
    :return articles: Dictionary containing full PubMed articles and associated metadata
    """
    articles = []
    for pubmed_id in pubmed_ids:
        handle = Entrez.efetch(db='pubmed', rettype='medline', retmode='text', id=pubmed_id)
        articles.append(Medline.parse(handle))
        time.sleep(0.35)  # use this to avoid exceeding the PubMed max pull of 3 URL requests per second
        return articles


def articles_to_csv(articles, save_filepath, filename):
    # pubmed_articles.csv
    csv_file = open(f'{save_filepath}/{filename}', 'w')
    dict_writer = csv.DictWriter(csv_file, articles[0].keys())
    dict_writer.writeheader()
    dict_writer.writerows(articles)
    csv_file.close()
