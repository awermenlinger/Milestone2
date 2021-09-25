import logging
import numpy as np
from gensim.topic_coherence import direct_confirmation_measure

log = logging.getLogger(__name__)

ADD_VALUE = 1

def custom_log_ratio_measure(segmented_topics, accumulator, normalize=False, with_std=False, with_support=False):
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for w_prime, w_star in s_i:
            w_prime_count = accumulator[w_prime]
            w_star_count = accumulator[w_star]
            co_occur_count = accumulator[w_prime, w_star]

            if normalize:
                # For normalized log ratio measure
                numerator = custom_log_ratio_measure([[(w_prime, w_star)]], accumulator)[0]
                co_doc_prob = co_occur_count / num_docs
                m_lr_i = numerator / (-np.log(co_doc_prob + direct_confirmation_measure.EPSILON))
            else:
                # For log ratio measure without normalization
                ### _custom: Added the following 6 lines, to prevent a division by zero error.
                if w_star_count == 0:
                    log.info(f"w_star_count of {w_star} == 0. Adding {ADD_VALUE} to the count to prevent error. ")
                    w_star_count += ADD_VALUE
                if w_prime_count == 0:
                    log.info(f"w_prime_count of {w_prime} == 0. Adding {ADD_VALUE} to the count to prevent error. ")
                    w_prime_count += ADD_VALUE
                numerator = (co_occur_count / num_docs) + direct_confirmation_measure.EPSILON
                denominator = (w_prime_count / num_docs) * (w_star_count / num_docs)
                m_lr_i = np.log(numerator / denominator)

            segment_sims.append(m_lr_i)

        topic_coherences.append(direct_confirmation_measure.aggregate_segment_sims(segment_sims, with_std, with_support))

    return topic_coherences