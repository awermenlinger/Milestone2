import pickle
from transformers import pipeline
import pandas as pd
import os
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

summarizer = pipeline("summarization", model="google/bigbird-pegasus-large-pubmed")

text = """Using a day 1 and 8, every-3-week schedule, our purpose was to determine the maximum tolerated dose of irinotecan (Cis phase I trial, the maximum tolerated dose was defined as the dose level immediately below the level in which two of the first three patients in any cohort, or at least two of six patients in any expanded cohort, experienced dose-limiting toxicity. Dose-limiting toxicity pertained only to toxicity during the first cycle of treatment. Escalation of irinotecan was planned in groups of three patients, with three additional patients added at the first indication of dose-limiting toxicity. A total of 19 patients have been enrolled. Grade 4 diarrhea was the dose-limiting toxicity at the irinotecan dose of 115 mg/m2. Hematologic toxicity was not dose limiting. Three patients required canceling of the day 8 dose due to grade 3 myelosuppression. Three patients, two with pancreatic cancer and one with metastatic carcinoma of unknown primary, had a partial response. The maximum tolerated dose of irinotecan in this combination was 100 mg/m2/dose. The dose-limiting toxicity was diarrhea. The maximum tolerated dose is the recommended starting dose for phase II studies."""

summary_text = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
print(summary_text)


#### BELOW WORKS ####
# raw_text_file = "models/raw_texts.sav"
# abstracts = pickle.load(open(raw_text_file, 'rb'))

# result_arrays = 'models/bi_trained_NMF_model_sklearn_res_array.npy'
# topic_weights = np.load(result_arrays)

# df_weights = pd.DataFrame(topic_weights)

# maxValueIndex = df_weights.idxmax()
 
# print("Maximum values of columns are at row index position :")
# print(maxValueIndex)

# print(abstracts[maxValueIndex[0]])
