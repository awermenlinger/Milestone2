from Bio import Entrez
from configurations.config import *

# Goal of this python script is to fetch the pubids from pubmed and save them to a txt file

my_email = antoine_email
API = antoine_api_key


def search(query):
    Entrez.email = my_email
    Entrez.api_key = API
    handle = Entrez.esearch(
        db='pubmed',
        sort='pubdate',
        retmax='5000',
        retmode='xml',
        term=query
    )

    results = Entrez.read(handle)
    return results

#query = "(\"biotechnology\"[MeSH Terms] OR \"biotechnology\"[All Fields] OR \"biotechnologies\"[All Fields]) AND ((fha[Filter]) AND (clinicaltrial[Filter] OR randomizedcontrolledtrial[Filter]))"
query = "biotechnology AND has abstract[FILT] AND (clinical trial[FILT] OR randomized controlled trial[FILT])"

results = search(query)
id_list = results['IdList']

with open('pubmed_ids_from_search.txt', 'w') as f:
    for item in id_list:
        f.write("%s\n" % item)
