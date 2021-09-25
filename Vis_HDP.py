import pickle
import os
from pprint import pprint

#SETTINGS
#num_topics=10

print ("Loading model")
filename = 'models/trained_hdp.sav'
hdp_model = pickle.load(open(filename, 'rb'))

pprint(hdp_model.print_topics())