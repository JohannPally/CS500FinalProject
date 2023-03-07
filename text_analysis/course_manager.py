import numpy as np
import pandas as pd
import logging
from itertools import combinations as comb
# TODO bundle package to module?
import semantic_mapper
from sklearn.cluster import KMeans as kn
from sklearn.metrics import v_measure_score as vms
from sklearn.metrics import completeness_score as com
from sklearn.metrics import silhouette_score as sil

import statsmodels.api as sm
from statsmodels.formula.api import ols, glm
from statsmodels.stats.anova import anova_lm
from statsmodels.multivariate.manova import MANOVA
import statsmodels.stats.multicomp as multi
from statsmodels.stats.oneway import test_scale_oneway as osv
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from scipy import stats

logging.basicConfig(level=logging.WARNING)
fname = 'uiuc_courses.csv'
dirname = 'results4/'
targets = ['ACE', 'ALEC', 'ANSC', 'ASTR', 'BIOE', 'ECE', 'MATH', 'NPRE', 'PSYC']

with open(fname, 'r') as inf:
	data = pd.read_csv(inf, quotechar='"', header=0)

logging.debug(f"Total courses availabe = {len(data)}\n") 

# split out the alphabetic part of the course titles
data['dep'] = data['Title'].str.split('\d{1,}',0).str[0].str.strip()

prgrms = data['dep'].unique()
prgrms.sort()

K = len(prgrms)
print(f"found {K} unique programs of study in prgrms:\n {prgrms}")
lm_formula = 'label ~ emb0 + emb1 + emb0*emb1'

for ind,prgm in enumerate(prgrms):
	# tables=[]
	# args = courses file, department, use entropy weights, n-neighbors, dist metric, density prop., verbose, output type
	# returns: pandas dataframe with resulting labels and embedding
	logging.debug(f"Running {prgm}: {ind:03d}/{K}...")
	res = semantic_mapper.main(fname, prgm, True, 10, 'canberra', .3, False, 3, dirname)
	wts = res['all_wts'].to_numpy()
	res = res[['label', 'emb0', 'emb1']]
	
	flat_wts = [wt for crs in wts for wt in crs]
	

	num_k = len(res['label'].unique())
	if res['label'].size < 10:
		with open(dirname+prgm+'_anova.txt', 'w') as outf:
			outf.write('Too small to run ANOVA. Saving results...\n\n')
			res.to_string(outf)
			if len(flat_wts) >= 7:
				outf.write("\n\n ====== WEIGHTS STATISTICS ====== \n\n")
				outf.write(f"Mean = {np.mean(flat_wts)}\nStd Dev = {np.std(flat_wts)}\nSkewness={stats.skew(flat_wts)}\nKurtosis = {stats.kurtosis(flat_wts)}\n")
				normality = stats.kstest(flat_wts, 'norm', args = (np.mean(flat_wts), np.std(flat_wts)))
				outf.write("\n\n ====== NORMALITY OF WEIGHTS ====== \n\n")
				outf.write(f"KS-test of normality statitic: {normality[0]}: p-value: {normality[1]} \n\n")
		continue
	
	normality = stats.kstest(flat_wts, 'norm', args = (np.mean(flat_wts), np.std(flat_wts)))
	
	mi = pd.MultiIndex.from_product([['emb0', 'emb1'], res['label'].unique()], names=['cont','cat'])
	
	try:
		manmodel = MANOVA.from_formula('emb0 + emb1 ~ label', data=res)
	except ValueError:
		continue
	anova_res = manmodel.mv_test()
	
	truths = res['label'].to_numpy()
	logging.debug("Running K-Means on Results...")
	if len(res['label'].unique()) <= 2:
		maxk = 2
	else:
		maxk = len(res['label'].unique())
	
	kmeans_res = []
	for nk in range(2,maxk+1):
		kmeans = kn(n_clusters=nk, init='k-means++', max_iter=300, random_state=0).fit(np.asarray(list(zip(res['emb0'],res['emb1']))))	
		klab = kmeans.labels_
		
		silh = sil(list(zip(res['emb0'],res['emb1'])),klab)
		k_res = vms(truths, klab)
		cmp = com(truths, klab)
		# logging.debug(f"\ntrue labels: {truths}\npred labels: {klab}")
		logging.debug(f"\n  For k={nk}, K-means v measure score: {k_res}\n K-means completeness: {cmp}")
		kmeans_res.append((nk, k_res, cmp, silh))
	# TODO implement Levene's test for variance homogeneity across groups
	
	FS = []
	PS = []
	if num_k >= 2:
		# logging.debug("Running two-way ANOVAs...")
		grps0 = [res.loc[res['label']==i, 'emb0'] for i in res['label'].unique()]
		grps1 = [res.loc[res['label']==i, 'emb1'] for i in res['label'].unique()]
		
		# Levene's test for variance
		
		for pair in comb(list(range(num_k)), 2):
			[F0,P0] = stats.levene(grps0[pair[0]], grps0[pair[1]])
			[F1,P1] = stats.levene(grps1[pair[0]], grps1[pair[1]])
			FS.append((pair,F0,F1))
			PS.append((pair,P0,P1))
		
	mcomp1 = multi.MultiComparison(res["emb0"], res["label"])
	mcomp2 = multi.MultiComparison(res["emb1"], res["label"])
	pw_lmr1 = mcomp1.tukeyhsd(alpha=0.05)
	pw_lmr2 = mcomp2.tukeyhsd(alpha=0.05)
	
	
	lmr = glm(lm_formula, res, missing='drop').fit()
			
	# anova_res = sm.stats.anova_lm(lmr, typ=2)
	
	# tukey_res = pairwise_tukeyhsd(endog=, groups=res['label'],alpha=0.05)
	# logging.debug(pw_lmr1)
	# tables.append(anova_res)
	
	if prgm in targets:
		with open('_K_MEANS.txt', 'a') as outf:
			outf.write(f"{prgm.lower()}V = [{','.join([str(x[1]) for x in kmeans_res])}]\n")
			outf.write(f"{prgm.lower()}C = [{','.join([str(x[2]) for x in kmeans_res])}]\n")
			outf.write(f"{prgm.lower()}S = [{','.join([str(x[3]) for x in kmeans_res])}]\n")
	
	with open(dirname+prgm+'_anova.txt', 'w') as outf:
		# anova_res.to_string(outf)
		outf.write("\n\n ====== WEIGHTS STATISTICS ====== \n\n")
		outf.write(f"N = {len(flat_wts)}\nMean = {np.mean(flat_wts)}\nStd Dev = {np.std(flat_wts)}\nSkewness={stats.skew(flat_wts)}\nKurtosis = {stats.kurtosis(flat_wts)}\n")
		outf.write(" ====== NORMALITY OF WEIGHTS ====== \n\n")
		outf.write(f"KS-test of normality statitic: {normality[0]}: p-value: {normality[1]} \n\n")
		if len(PS) > 0:
			outf.write(" ====== LEVENE's TEST ====== \n\n")
			for ind,res in enumerate(PS):
				outf.write(f"For emb0 group {res[0][0]} : group {res[0][1]} = {FS[ind][1]}, p-val {PS[ind][1]}\n")
				outf.write(f"For emb1 group {res[0][0]} : group {res[0][1]} = {FS[ind][2]}, p-val {PS[ind][2]}\n\n")
			
		outf.write("Kmeans Clustering Results:\n\n")
		for km in kmeans_res:
			outf.write(f"For k={km[0]}, K-means v measure score: {km[1]}. K-means completeness: {km[2]}. Silhouette: {km[3]}\n")
			
		outf.write(anova_res.summary().as_text())
		outf.write('\n\n ======= LINEAR MODEL ====== \n\n')
		outf.write(lmr.summary().as_text())
		outf.write('\n\n ======= PAIRWISE EMB0 ====== \n\n')
		outf.write(pw_lmr1.summary().as_text())
		outf.write('\n\n ======= PAIRWISE EMB1 ====== \n\n')
		outf.write(pw_lmr2.summary().as_text())