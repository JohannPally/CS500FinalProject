from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import re
import string
import numpy as np

NUM_MATCHES = 5

pat = re.compile('()')

# load acm text map
with open("ACM_Map_Full.tsv", "r") as inf:
	data = pd.read_csv(inf, header=0, delimiter='\t')

# Load word2vec for SE
word_vect = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)


CS124 = "Basic concepts in computing and fundamental techniques for solving computational problems. Intended as a first course for computer science majors and others with a deep interest in computing."
CS124 = CS124.translate(str.maketrans('', '', string.punctuation))
words = data['SEARCH'].to_list()
words = [pat.sub('', w) for w in words]
highest_rel = np.zeros(len(words))

for i,w in enumerate(words):
	similarity = word_vect.wmdistance(w, CS124)
	highest_rel[i] = similarity		
	# print(f"CS124 vs {w}: {similarity}")

highest_matches = np.argpartition(-highest_rel, NUM_MATCHES)[:NUM_MATCHES]
print(np.array(words)[highest_matches], np.array(highest_rel)[highest_matches])

