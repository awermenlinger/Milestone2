from Bio import Entrez

# A brief example of how to pull an article abstract based on a key word search.
# Used "fever" as a key word search which returns the article IDs.
# Then used the first article ID to pull out the article title and abstract text as an example.


my_email = 'msmcmanu@umich.edu'

def search(query):
    Entrez.email = my_email
    handle = Entrez.esearch(
        db='pubmed',
        sort='relevance',
        retmode='xml',
        term=query
    )

    results = Entrez.read(handle)
    return results


results = search('fever')
id_list = results['IdList']
print(id_list)


def fetch_details(id_list):
    # ids = ','.join(id_list)
    Entrez.email = my_email
    handle = Entrez.efetch(
        db='pubmed',
        retmode='xml',
        id=id_list[0]
    )
    results = Entrez.read(handle)
    return results


query_1 = fetch_details(id_list)
title = query_1['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle']
abstract_text = " ".join(query_1['PubmedArticle'][0]['MedlineCitation']['Article']['Abstract']['AbstractText'][:])
print(f"{title}\n{abstract_text}")
