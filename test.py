import pickle
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import numpy as np
from gensim.corpora import dictionary
import pandas as pd
import gensim
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models
import pickle
import os
import logging
from tqdm import tqdm
from gensim.models.ldamulticore import LdaMulticore
from pprint import pprint

# load the model from disk
filename = 'models/trained_lda.sav'
#ldamodel = pickle.load(open(filename, 'rb'))
ldamodel = LdaMulticore.load(filename)

print(ldamodel.get_term_topics("breast"))