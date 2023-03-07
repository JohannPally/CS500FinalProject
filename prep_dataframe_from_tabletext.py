import pandas as pd
import numpy as np
import os
import pickle

# load file from table prep'd from ripping the tables from pdf
# file was cleaned manually after ripping to fix errors and replace spacing with tab
with open('ACM_map.txt', 'r') as inf:
	data = pd.read_csv(inf, sep='\t', header=None, names=['CODE', 'NAME', 'TYPE', 'RELEVANCE', 'HRS'], skip_blank_lines=True)

# swap label chars with readable names
data['TYPE'] = data['TYPE'].apply(lambda x: 'Knowledge' if x=='k' else 'Comprehension' if x=='c' else 'Application' if x=='a' else '')
data['RELEVANCE'] = data['RELEVANCE'].apply(lambda x: 'Essential' if x=='E' else 'Desirable' if x=='D' else '')

print(data.head(10))
data.to_csv('ACMmap.csv', sep='\t')

