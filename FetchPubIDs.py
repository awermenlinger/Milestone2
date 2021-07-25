from Bio import Entrez
import pandas as pd

# Goal of this python script is to fetch the pubids from pubmed and save them to a csv file

my_email = 'awerm@umich.edu'
API = "ee71d15e8a446320e3aaacd01c88966efe08"


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
df_temp = pd.DataFrame(id_list)
df_temp.to_csv("Milestone2\pubmed_ids_from_search.csv")