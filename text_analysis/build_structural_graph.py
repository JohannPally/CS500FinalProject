import numpy as np
import pandas as pd
import logging
import re
import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt

G = nx.OrderedDiGraph()

logging.basicConfig(level=logging.WARNING)
fname = 'uiuc_courses.csv'

targets = ['BIOE']
prereq = re.compile(r'[A-Z]{2,} [0-9]{3}')

with open(fname, 'r') as inf:
	data = pd.read_csv(inf, quotechar='"', header=0)

logging.debug(f"Total courses availabe = {len(data)}\n") 

# split out the alphabetic part of the course titles
data['dep'] = data['Title'].str.split('\d{1,}',0).str[0].str.strip()

prgrms = data['dep'].unique()
prgrms.sort()

bioe = data[data['Title'].str.contains('^BIOE ')]
logging.debug(f"Read in all BioE: N={len(bioe)}\n")

nodes = []
edges = []
lvls = []
for row in bioe.itertuples():
	node = row.Title
	if node not in nodes:
		nodes.append(node) 
		lvls.append(node.split(' ')[-1][0])
		
	if 'Prerequisite' in row.Description:
		prereqs = prereq.findall(row.Description.split('Prerequisite')[-1])
		if len(prereqs) > 0:
			for ind, pre in enumerate(prereqs):
				if not pre in nodes:
					nodes.append(pre)
					lvls.append(pre.split(' ')[-1][0])
					
				# add edge (from, to)
				edges.append((pre, node))

pos = {}
xind = .1
yind = .5
size = 5
padding = 5
curr_lvl = .1
max_x = 0
colors = []
# seen = set()
# dups = set(x for x in nodes if x in seen or seen.add(x))
# print(dups) 
# organize chart by level then draw positionally
for lvl,node in sorted(zip(lvls,nodes)):
	if int(lvl) > curr_lvl:
		curr_lvl = int(lvl)
		xind = .1
	G.add_node(node)
	if node in ('BIOE 120', 'BIOE 201', 'BIOE 202', 'BIOE 205', 'BIOE 210', 'BIOE 302', 'BIOE 414', 'BIOE 415', 'BIOE 476', 'BIOE 572'):
		colors.append('#a11868')
	else:
		colors.append('#4b006e') if 'BIOE' in node else colors.append('#020035') 
	pos[node] = (xind, yind*int(lvl))
	xind = round(xind + size + padding,2)
	if xind > max_x:
		max_x = xind

G.add_edges_from(edges)

print(pos)
print(xind)
plt.figure(figsize=(16,9))
plt.gca().set_xlim((0-padding,max_x+2*padding))
plt.gca().set_ylim((0,yind*curr_lvl+padding))
nx.draw_networkx(G, pos=pos, node_size=3000, font_size=10, font_color='#c8c8c8', font_weight='bold', node_color=colors, connectionstyle='arc3,rad=0.1')


plt.tight_layout()
plt.show()