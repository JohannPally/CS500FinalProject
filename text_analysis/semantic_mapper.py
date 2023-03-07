import os
import numpy as np
import pandas as pd
from scipy import sparse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pprint import pprint 
import densmap
import umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import argparse
import logging
import re
from datetime import datetime

pd.options.mode.chained_assignment = None  # default='warn'
# load single and multiple letter frequencies built from nltk words() from file
with open('single_letter_freq.npy', 'rb') as inf:
	single_freq = np.load(inf)
with open('letter_pair_freq.npy', 'rb') as inf:
	double_freq = np.load(inf)

# set stopwords
stopw = set(stopwords.words('english'))

def prob(input_str):
	if 0.0 <= float(input_str) <= 1.0:
		return float(input_str)
	else:
		raise argparse.ArgumentTypeError("Argument must be in range [0.0, 1.0]")
		


# old dict from ND site, built with a 'concise dictionary'
concise_dict_rel_alpha = np.array([.084966, .020720, .045388, .033844, .111607, .018121, .024705,
						.030034, .075448, .001965, .011016, .054893, .030129, .066544,
						.071635, .031671, .001962, .075809, .057351, .069509, .036308, 
						.010074, .012899, .002902, .017779, .002722], dtype=np.double)



# Comment out all code
# TODO clean up decompose and entropy

# TODO maybe split out dictionary loading and word calling?
# TODO save events like cross listing and fallback to entropy2 and write out to _stats file
# TODO check that course labels are correct to the umap results 
# TODO build matrix with output embedding coordinates and course labels.
# TODO add script for ANOVA 


def decompose(word, dim=2, amap=None, feat=False, f_wt=None):
	""" Apply embedding to single word using A1Z26 scheme for substrings.
		
		Parameters:
			-word 		= string, ONLY alphabetic characters
			-dim (opt)  = int, length dimension of substring to embed. DEFAULT, 2
			-amap(opt)  = iterable, existing map to append to instead of having to assign later DEFAULT, None
			-feat(opt)  = Boolean, apply a feature weight or not DEFAULT, False
			-f_wt(opt)  = numeric, weighting to use
		Returns:
			- list of numeric tokens if amap is not provided, else none.
 	"""
	if feat:
		if not f_wt:
			f_wt = 1
	
	# TODO figure out addtl feature adding??
	dec = []
	# loop over word
	for grp in range(max(1, len(word)-dim+1)):
		a1z26 = [(ord(a)-96) for a in word[grp:min(grp+dim, len(word))]] + [0]*max(0, dim-len(word))
		if not amap is None:
			amap.append(a1z26)
			return 
		else:
			dec.append(a1z26)
	
	# return result
	return dec if len(dec) > 0 else 1

def entropy2(word):
	""" Calculate the relative entropy of a word using single letter
		frequency from a provided dictionary.  Assumes letters are 
		independent probabilities. (Note: This is not a good assumption
		for structured language).
		
		Parameters:
			-word   = string, ONLY aphabetic characters
		Returns:
			float, the shannon entropy of the 
		
	"""
	num = map(lambda w: single_freq[0, ord(w)-97], word)
	return -np.sum([i*np.log2(i) for i in num])


def decompose2(word, amat, weight=True):
	""" Apply embedding to single word using A1Z26 scheme for pairs of letters.
		
		Parameters:
			-word 		  = string, ONLY alphabetic characters
			-amat	      = iterable, existing map to append to instead of having to assign later
			-weight(opt)  = Boolean, apply a feature weight or not DEFAULT, False
		Returns:
			- none.
 	"""
	global all_wts
	wtd = entropy2(word) if weight else 1
	all_wts.append(wtd)
	for let in range(1,len(word)):
		amat[ord(word[let-1])-97, ord(word[let])-97] += wtd
		
def entropy3(word):
	""" Calculate the relative entropy of a word using single letter
		frequency from a provided dictionary.  Assumes letters are 
		independent probabilities. (Note: This is not a good assumption
		for structured language).
		
		Parameters:
			-word   = string, ONLY aphabetic characters
		Returns:
			float, the shannon entropy of the 
		
	"""
	if len(word) < 2:
		freq = single_freq[0, ord(word[0])-97]
		return -np.sum([freq*np.log2(freq)])
	else:
		num = list(map(lambda w: double_freq[ord(word[w-1])-97, ord(word[w])-97], range(1,len(word))))
		if any([i==0 for i in num]):
			logging.debug(f"Got a zero for word: {word}.  Defaulting to entropy2...")
			return entropy2(word)
		return -np.sum([np.log2(float(i)) for i in num]) # -np.sum([float(i)*np.log2(float(i)) for i in num])


def decompose3(word, amat, weight=True):
	""" Apply embedding to single word using A1Z26 scheme for pairs of letters.
		
		Parameters:
			-word 		  = string, ONLY alphabetic characters
			-amat	      = iterable, existing map to append to instead of having to assign later
			-weight(opt)  = Boolean, apply a feature weight or not DEFAULT, False
		Returns:
			- none.
 	"""

	wtd = entropy3(word) if weight else 1
	for let in range(1,len(word)):
		amat[ord(word[let-1])-97, ord(word[let])-97] += wtd
		
	return wtd
	
	
# TODO add option to find all courses with keyword(s)
def get_all_courses(data, dep):
	courses = data[data['Title'].str.contains('^'+dep+' ')]
	logging.debug(f"Read in all {dep}: N={len(courses)}\n")
	return courses


def find_cross_listings(data, courses):
	
	is_cross = re.compile(r'Same as (?P<cref>[A-z]+ \d{1,3}).( )+?See ([A-z]+ \d{2,})')
	matches = [is_cross.match(descr) for descr in courses['Description']]
	mask = [False if match is None else True for match in matches]
	sub = mask.copy()
	for ind,clist in enumerate(mask):
		if clist:
			lookup = matches[ind].group('cref').strip()
			cname = courses.iloc[ind,0]
			logging.debug(f"Crosslist detected at {cname}. Fetching description for {lookup}")
			found = data.loc[data['Title']==lookup]
			
			# TODO catch error is search
			sub[ind] = found.Description.iloc[0]
			logging.debug(found['Description'].str[:20])
			# print(found.Description.iloc[0])
			courses.Description.iloc[ind] = found.Description.iloc[0]

	return (sum(mask), sub)


def build_encoding_from_text(courses, entropy):
	global all_wts
	alpha_map = []
	labels = []
	for ind in range(len(courses)):	
		title = courses.iloc[ind,0]
		logging.debug(f"processing {title}...")
		labels.append((title, int([ch for ch in title if ch.isnumeric()][0])))
		line = courses.iloc[ind,2]
		
		# split words
		linewords = word_tokenize(line)
		# remove stop words
		linewords = [w.lower() for w in linewords if w not in stopw and w.isalpha()]
		course_words = []
		course_wts = []
		amat = np.zeros((26,26), dtype=np.single)
		if len(linewords) < 1:
			logging.debug(f"Course {courses.iloc[ind,0]} empty. Skipping...")
			continue
		for word in linewords:
			# get semantic blur for each word 
			# sem_blur.append((word[0],len(word),word[-1]))
			wt = decompose3(word, amat, entropy)
			course_wts.append(wt)
		alpha_map.append(amat.flatten())
		all_wts.append(course_wts)
		
	try:
		logging.debug(f"\n course={courses.iloc[2,0]},\n mat={alpha_map[2][:25]}...")
	except IndexError:
		logging.debug(f"\n course count low...")
	return (labels, alpha_map)
	
	
def run_densmap(lab, alpha_map, nn, met, dens, verb, out_plot, ccount=-1): 
	global all_wts
	global outdir
	
	flat_weights = [wt for crs in all_wts for wt in crs]
	logging.info("running densmap...")
	[titles, labels] = zip(*lab)
	u_labels = np.unique(labels)
	N = len(alpha_map)
	K = len(u_labels)
		
	lvl = logging.getLogger().getEffectiveLevel()
	logging.getLogger().setLevel(logging.WARNING)
	if nn > len(labels):
		logging.warning("Neighbors set to larger number than num courses, N. Falling back to unique(N) or 3, whichever is larger.")
		nn = max(K,3)
		pprint(nn)
	emb, rinp, remb = densmap.densMAP(n_neighbors=nn, metric=met, dens_frac=dens,
							dens_lambda=.5, n_components=2, verbose=verb).fit_transform(alpha_map)
	logging.getLogger().setLevel(lvl)
	
	if out_plot>0:
		fig, ax = plt.subplots(1,1,figsize=[8,8])
		ax = [ax]
		color_int = np.array(u_labels, dtype=int)
		cmap = cm.viridis(np.linspace(0,1,K+1))
		
		logging.getLogger().setLevel(lvl)
		logging.debug(f"course levels: {color_int}")
		
		colors=np.asarray([cmap[i] for i in range(len(color_int), 0, -1)])
		markers = ['o', 's', 'P', '^', 'D', 'h', '*']
		
		logging.getLogger().setLevel(logging.WARNING)
		for i,u in enumerate(u_labels):
			xi = [emb[:,0][j] for j in range(len(emb[:,0])) if labels[j] == u]
			yi = [emb[:,1][j] for j in range(len(emb[:,1])) if labels[j] == u]
			# zi = [emb[:,2][j] for j in range(len(emb[:,2])) if labels[j] == u]
			ax[0].scatter(xi, yi, color=colors[i], marker=markers[i], s=60, alpha=.4)
			# xh = [rinp[k] for k in range(len(rinp)) if labels[k] == u]
			# yh = [remb[k] for k in range(len(remb)) if labels[k] == u]
			# ax[1].scatter(xh, yh, color=colors[i], marker=markers[i], s=60, alpha=.4)
		ax[0].legend([str(lab)+'00' for lab in u_labels],fontsize=14)
		# ax[1].legend([str(lab)+'00' for lab in u_labels],fontsize=14)
		ax[0].set_xlabel('Embedded Coordinate 0', fontsize=14)
		ax[0].set_ylabel('Embedded Coordinate 1', fontsize=14)
		
		ax[0].spines['right'].set_visible(False)
		ax[0].spines['top'].set_visible(False)
		ax[0].xaxis.set_ticks_position('bottom')
		ax[0].yaxis.set_ticks_position('left')
		
		ax[0].tick_params('both', direction='out')
		
		
		# ax[1].set_xlabel('Input Coordinate 0', fontsize=14)
		# ax[1].set_ylabel('Input Coordinate 1', fontsize=14)
		fig.tight_layout()
		
		# histogram weights
		fig2 = plt.figure(3)
		axes = plt.subplot(111)
		heights, bins, patches = axes.hist(flat_weights, color='white', edgecolor='black', linewidth=1.25)
		# histogram formatting to look like R:
		axes.spines['right'].set_color('none')
		axes.spines['top'].set_color('none')
		axes.xaxis.set_ticks_position('bottom')

		# was: axes.spines['bottom'].set_position(('data',1.1*X.min()))
		axes.spines['bottom'].set_position(('axes', -0.05))
		axes.yaxis.set_ticks_position('left')
		axes.spines['left'].set_position(('axes', -0.05))

		axes.set_xlim([0, np.ceil(max(bins))])
		axes.set_ylim([0,max(heights)+5])
		axes.xaxis.grid(False)
		axes.yaxis.grid(False)
		plt.xlabel('Word Letter-pair Entropy', fontsize=14)
		plt.ylabel('Count', fontsize=14)
		fig2.tight_layout()
		
		
		if out_plot>=2:
			dep = titles[0].split(' ')[0]
			# TODO Fix path name to be more robust
			fig.savefig(os.path.join(outdir,dep+'_embedding.png'))
			fig2.savefig(os.path.join(outdir,dep+'_wt_histogram.png'))
			now = datetime.now()
			dt_string = now.strftime("%Y/%m/%d %H:%M:%S")
			with open(os.path.join(outdir,dep+'_stats.txt'), 'w') as outf:
				outf.write(dep+",Run:"+dt_string+"\n")
				outf.write("num courses,"+str(N)+"\n")
				if ccount > -1:
					outf.write("num cross-listed courses,"+str(ccount)+"\n")
				outf.write("courses,"+";".join(titles)+"\n")
				outf.write("embedded coordinates,"+str(list(zip(emb[:,0],emb[:,1])))[1:-1]+"\n")
				outf.write("labels,"+";".join([str(i) for i in labels])+"\n")
				outf.write("weights,")
				outf.write(str(tuple([i for i in all_wts]))[1:-1])
				outf.write("\n")
				
		if out_plot < 3:
			plt.show()
		
	else:
		pprint(emb)
	plt.close(fig)
	plt.close(fig2)
	logging.getLogger().setLevel(lvl)
	
	
	return emb 

def main(fname, dep, entropy, nn, met, dens, verb, out_plot,dir=''):
		
	# TODO make this more robust and less stupid
	global outdir
	outdir = dir
	global all_wts
	all_wts = []
	# print(fname)
	with open(fname, 'r') as inf:
		data = pd.read_csv(inf, quotechar='"', header=0)
	logging.debug(f"Total courses availabe = {len(data)}\n") 
	
	# get course description text
	courses = get_all_courses(data, dep)
	
	# get cross listed courses
	(clist_count, clist_descr) = find_cross_listings(data, courses)
	logging.debug(f"Found {clist_count} cross listed courses.\n")
	
	# embed course text to numeric matrix pairwise
	(lab, alpha) = build_encoding_from_text(courses, entropy)
	
	redirect = False
	if np.sum(sum(alpha)) == 0:
		logging.warn(f"Encountered all empty courses for {dep}. Continuing...")
		redirect = True
	if len(courses) < 10 or redirect:
		logging.warning("WARNING: Low count encountered (N < 10) or empty set. Embedding by (mean,std) instead.")
		emb = np.zeros((len(courses),2), dtype = np.float32)
		for ind,each in enumerate(all_wts):
			emb[ind,:] = [np.mean(each), np.std(each)]
	else:
		# run densmap 
		emb = run_densmap(lab, alpha, nn, met, dens, verb, out_plot, clist_count)
	
	[titles, labels] = zip(*lab)
	retmat = pd.DataFrame(list(zip(labels, emb[:,0], emb[:,1], all_wts)), columns=['label', 'emb0', 'emb1', 'all_wts'])
	
	return retmat

if __name__ == '__main__':
	fname = 'uiuc_courses.csv'
	dep = 'AE'
	sem_blur = []
	# array to store word weights
	
	parser = argparse.ArgumentParser(description='Process a department worth of course descriptions.')
	parser.add_argument('-f', '--file', type=str, default='uiuc_courses.csv', 
						help='csv file containing 3 column entries for courses: Title, Name, Description.')
	parser.add_argument('-d', '--dep', type=str, default='AAS', 
						help='Alphabetic department/program of study, processed to upper()')
	parser.add_argument('-e', '--entropy', type=bool, default=True,	
						help='Process with entropy weighting for words')
	parser.add_argument('-p', '--plot', type=int, default=0,
						help='0=print embedding, 1=plot and show, 2=plot, show ad save results to text file, 3=save plots and text dont show.')
	parser.add_argument('-n','--neighbors', type=int, default=10,	
						help='Number of nearest neighbors to apply density weighting')
	parser.add_argument('-m', '--dist', type=str, default='canberra',
						nargs='?', choices=('euclidean', 'manhattan', 'mahalanobis', 'canberra',
						'hamming', 'braycurtis'))
	parser.add_argument('-r', '--dense', type=prob, default=0.3, 
						help='Proportion of inputs to use for density preservation')
	parser.add_argument('-o', '--outdir', type=str, default='',
						help='output directory for results')
	parser.add_argument('-v', '--verbose', type=bool, default=True,
						help='Increase verbose command line output for debugging')
	
	args = parser.parse_args()
	
	global outdir
	outdir = args.outdir
	
	if args.verbose:
		logging.basicConfig(level=logging.DEBUG)
	else:
		logging.basicConfig(level=logging.ERROR)
	
	main(args.file, args.dep.upper(), args.entropy, args.neighbors, args.dist, args.dense, args.verbose, args.plot)
	
	

