import pickle
dic_file = "models/trained_lda_dictionary.sav"
dictionary = pickle.load(open(dic_file, 'rb'))
print(dictionary[10000])