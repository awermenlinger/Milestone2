import pickle
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization", model="google/bigbird-pegasus-large-pubmed")

summary_per_topic = []

raw_text_file = "models/raw_texts.sav"
abstracts = pickle.load(open(raw_text_file, 'rb'))

result_arrays = 'models/bi_trained_NMF_model_sklearn_res_array.npy'
topic_weights = np.load(result_arrays)

df_weights = pd.DataFrame(topic_weights)

maxValueIndex = df_weights.idxmax()

i=1
for top_abs_topic in tqdm(maxValueIndex):
    abstract = abstracts[top_abs_topic]
    summary_text = summarizer(abstract, max_length=100, min_length=5, do_sample=False)[0]['summary_text']    
    topic = str(i) + ": " + summary_text
    summary_per_topic.append(topic)

print(summary_per_topic)
with open('results/bi_trained_NMF_topics.txt', 'w') as f:
    for item in summary_per_topic:
        f.write("%s\n" % item)



# @misc{zaheer2021big,
#       title={Big Bird: Transformers for Longer Sequences}, 
#       author={Manzil Zaheer and Guru Guruganesh and Avinava Dubey and Joshua Ainslie and Chris Alberti and Santiago Ontanon and Philip Pham and Anirudh Ravula and Qifan Wang and Li Yang and Amr Ahmed},
#       year={2021},
#       eprint={2007.14062},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }