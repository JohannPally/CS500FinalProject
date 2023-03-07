# run with conda env: meta
import sys
import numpy as np
import requests
import json
import time
import os
import re, string
from time import localtime, strftime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bsoup
import pandas as pd

from wordcloud import WordCloud, STOPWORDS


plt.style.use('ggplot')

retmax = 200
timeout = 100  # sec
delay = 0.5  # sec to wait between querries to avoid rate limit (4/sec MAX)
pagination = 20 # 10 20 50 are standard options.  More per page means slower load times per page.

run_whole = True
plot = False
# Use: AND, OR, ANDNOT 
outpath = 'C:\\Users\\Eliot\\Documents\\Python Scripts\\ACM_topics'


# swap characters for the web query, or swap a web query to human readable form
def swap_chars(query, readable=False):
	if readable:
		query = query.replace('+'  , '_')
		query = query.replace('%28', '')
		query = query.replace('%29', '')
		query = query.replace('%22', '-')
		query = query.replace('\\', '-OR-')
		query = query.replace('/', '-OR-')
	else:
		query = query.replace(' ', '+')
		query = query.replace('(', '%28')
		query = query.replace(')', '%29')
		query = query.replace('"', '%22')
		query = query.replace('/', '+OR+')
		query = query.replace('\\', '+OR+')
	
	return query 

# read in ACM map with search term pre-defined
with open(os.path.join(outpath, 'ACMmap_full.tsv'), 'r') as inf:
	data = pd.read_csv(inf, header=0, sep='\t')
	searches = data['SEARCH'].to_list()
	

# iterate through search terms and pull all abstract text from results 
for query in searches:
	print(f"Now processing {query}... ", end="")
	query = swap_chars(query)
	esearch = f'https://dl.acm.org/action/doSearch?AllField={query}&expand=all&startPage=0&pageSize=10'
	ret = requests.get(esearch)
	if ret.status_code == 200:
		soup = bsoup(ret.text, 'html.parser')
		# print(dir(soup))
	else:
		print("No response")
		sys.exit(1)
	
	num_found = soup.find('span', class_="result__count").text
	num_found=num_found.strip().split(' Result')[0]
	num_found = int(num_found.replace(',',''))
	# print(f"Got {num_found} results for that search")

	if not run_whole:
		sys.exit(0)

	titles = []
	summarys = []
	starttime = time.time()

	pages = int(np.ceil(int(num_found)/pagination))
	max_pages = int(retmax//pagination)

	abstractsearch = re.compile('.*issue-item__abstract.*')

	for page in range(min(pages, max_pages)):
		
		esearch = f'https://dl.acm.org/action/doSearch?AllField={query}&startPage={page}&pageSize={pagination}'
		# print("Being kind to the server, let's stop and smell the roses...")
		while time.time() < starttime + delay:
			time.sleep(0.25)
			
		ret = requests.get(esearch)
		starttime = time.time()
		if ret.status_code == 200:
			soup = bsoup(ret.text, 'html.parser')
			# print(dir(soup))
		else:
			print(" No response! ")
			sys.exit(1)
		
		# parse out responses
		for article in soup.find_all('div', class_="issue-item__content"):
			title = []
			root = article.find('h5', class_="issue-item__title")
			for child in root.children:
				title.append(child.text)
			titles.append(''.join(title))
			abstract = article.find("div", {"class" : abstractsearch})
			try:
				summarys.append(abstract.find("p").text)
			except AttributeError:
				# print(f"No abstract for {titles[-1]}")
				summarys.append('None')
				continue
	
	query = swap_chars(query, readable=True)
	savefile = strftime(f"{query}_acm_top{min(retmax, num_found)}", localtime())
	savefile = os.path.join(outpath, savefile)
	
	if plot:
		txt = ' '.join(summarys)
		txt = txt.lower()
		pattern = re.compile('[\W_]+')
		to_remove = [pattern.sub('', word) for word in query.split(' ')]
		to_remove.append('quot')

		for w in to_remove:
			txt = re.sub(w, '', txt).strip()
		wordc = WordCloud(width = 3000, height = 2000, random_state=1, background_color='dimgray', colormap='Pastel2', collocations=False, stopwords=STOPWORDS).generate(txt)
		f = plt.figure(figsize=(40, 30))
		# Display image
		plt.imshow(wordc) 
		# No axis details
		plt.axis("off");
		plt.tight_layout()
		plt.savefig(savefile+".png")
		plt.show()

	with open(savefile+".txt", 'w', encoding='utf-8') as outf:
		for pub,abt in zip(titles, summarys):
			outf.write(pub + "\t" + abt + "\n\n")
		
	print(f"Successfully processed query: {query}.")
		
	# with open(savefile+"_raw.json", 'w', encoding='utf-8') as outf:
		# outf.write(json.dumps(data, sort_keys=True, indent=4))



