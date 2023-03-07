from nltk.corpus import words
from nltk import FreqDist
import numpy as np
from pprint import pprint
import pickle
# may need to run 
# >> import nltk
# >> nltk.download('words')

wordset = set(wrd.lower() for wrd in words.words() if wrd.isalpha())
print(f"Loaded {len(wordset)} words.\n")
single_freq = np.zeros((1,26), dtype=np.uint32)
pairwise_freq = np.zeros((26,26), dtype=np.uint32)

for w in wordset:
	# first letter
	single_freq[0, ord(w[0])-97] += 1
	# if only one letter (a, i, ...) go to next
	if len(w) < 2:
		continue
	#.. else loop over pairs in word
	for let in range(max(1, len(w)-1)):
		single_freq[0, ord(w[let+1])-97] += 1
		pairwise_freq[ord(w[let])-97, ord(w[let+1])-97] += 1

# convert to proportions
single_freq   = single_freq.astype(np.single)
pairwise_freq = pairwise_freq.astype(np.single)

single_freq = np.divide(single_freq, np.sum(single_freq))
pairwise_freq = np.divide(pairwise_freq, np.sum(pairwise_freq))

# report
pprint(single_freq)
pprint(pairwise_freq)

# dbl chx
print(f"Total single frequency = {np.sum(single_freq)}, total pairwise = {np.sum(pairwise_freq)}")

# # export list to pickle object
# with open('single_letter_freq.bin', 'wb') as outf:
	# pickle.dump(single_freq, outf)
# with open('single_letter_freq.npy', 'wb') as outf:
	# np.save(outf, single_freq)

# with open('letter_pair_freq.bin', 'wb') as outf:
	# pickle.dump(pairwise_freq, outf)
# with open('letter_pair_freq.npy', 'wb') as outf:
	# np.save(outf, pairwise_freq)