from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import re
import string
import numpy as np

NUM_MATCHES = 7  # number of top matches to associate with the course text

pat = re.compile('()')
hours = re.compile("([0-9] or )?[0-9] (under)?graduate hours")
prereq = re.compile("Prerequisite:")

outfile = "ACM_matched_CS_courses.tsv"

# load acm text map
with open("ACM_Map_Full.tsv", "r") as inf:
	data = pd.read_csv(inf, header=0, delimiter='\t')


#training wheels
# data = data[:10]


# load course list
with open("../uiuc_cs_courses2.csv", "r") as inf:
	courses = pd.read_csv(inf, header=0, delimiter=',')

# Get ACM target vocab
targets = data['SEARCH'].to_list()
targets = [pat.sub('', w) for w in targets]

# Load word2vec for SE
word_vect = KeyedVectors.load_word2vec_format("SO_vectors_200.bin", binary=True)


# CS124 = "Basic concepts in computing and fundamental techniques for solving computational problems. Intended as a first course for computer science majors and others with a deep interest in computing."
# CS124 = CS124.translate(str.maketrans('', '', string.punctuation))

# CS227 = ("Introduction to elementary concepts in algorithms and classical data structures with a focus on their applications in Data Science.", 
		# "Topics include algorithm analysis (ex: Big-O notation), elementary data structures (ex: lists, stacks, queues, trees, and graphs), basics of discrete algorithm design principles (ex: greedy, divide and conquer, dynamic programming), and discussion of discrete and continuous optimization.")
# CS227 = [C.translate(str.maketrans('', '', string.punctuation)) for C in CS227]
# print(CS227)

# write out header
with open(outfile, "w") as outf:
	outf.write("Course\tACM_Matches\tACM_Match_Score\n")


for row in courses.itertuples(index=False, name="Course"):
	# pre process source text 
	source = row.Description
	# exclude text following prerequisite listing
	pre = prereq.search(source)
	if pre is not None:
		source = source[:pre.span(0)[0]].strip()
	# exclude text about hours credit
	hrs = hours.search(source)
	if hrs is not None:
		source = source[:hrs.span(0)[0]].strip()
	
	# split text into list of sentences
	source = source.split(". ")
	source = [s for s in source if s != '']
	
	highest_rel = np.zeros(len(targets))
	
	output = []
	for sentence in source:
		for i,w in enumerate(targets):
			# similarity = word_vect.wmdistance(w.lower().split(), sentence.lower().split())
			similarity = word_vect.n_similarity(w.lower().split(), sentence.lower().split())
			if similarity == np.inf:
				similarity = 100
			highest_rel[i] = similarity		

		highest_matches = np.argpartition(-highest_rel, NUM_MATCHES)[:NUM_MATCHES]
		output.append(list(np.array(targets)[highest_matches]))
		# print(np.array(targets)[highest_matches], np.array(highest_rel)[highest_matches])
	with open(outfile, "a") as outf:
		outf.write(row.Title)
		outf.write('\t')
		outf.write(':'.join([''.join(k) for k in output]))
		outf.write('\t')
		outf.write(np.array(highest_rel)[highest_matches].__str__())
		outf.write("\n")
		
		
		
		
		
		
