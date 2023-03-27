import pandas as pd
from collections import defaultdict
import pprint as pp
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
 

# LOADING DATA
other_headers = ['Index','FALL_2018_GPA','SPRING_2022_GPA']
sem_headers = ['FALL_2018','SPRING_2019','FALL_2019','SPRING_2020','FALL_2020','SPRING_2021','FALL_2021','SPRING_2022']
#TODO need to redefine to be estimated class status
sem_trans_colors = {('SPRING_2020','FALL_2020'): 'red',
                ('FALL_2018','SPRING_2019'): 'orange', 
                ('SPRING_2019','FALL_2019'): 'yellow', 
                ('FALL_2019','SPRING_2020'): 'green', 
                ('FALL_2020','SPRING_2021'): 'blue', 
                ('FALL_2021','SPRING_2022'): 'purple', 
                ('SPRING_2021','FALL_2021'): 'brown'}

df = pd.read_csv('data.csv')
# print(df)

#HELPER FUNCTION FOR PARSING
def parse_courses(item):
    if type(item) == str:
        return item.split(', ')
    return None

# COUNTING RAW COUNTS FOR ENROLLMENT
raw_counts = defaultdict(lambda: 0)
for sem in sem_headers:
    for item in df[sem]:
        courses = parse_courses(item)
        if courses is not None:
            for crs in courses:
                raw_counts[crs] += 1

# print(raw_counts)

# COUNTING TRANSITIONS FOR ENROLLMENT ALONG SEMESTER
# TODO estimate class status depending on empty cells left or right for student
course_transitions = defaultdict(lambda: defaultdict(lambda: 0))

for ind in range(len(df)):
    prev_sem = sem_headers[0]
    for curr_sem in sem_headers[1:]:
        prev_courses = parse_courses(df[prev_sem][ind])
        curr_courses = parse_courses(df[curr_sem][ind])

        if prev_courses is not None and curr_courses is not None:
            for pc in prev_courses:
                for cc in curr_courses:
                    course_transitions[(pc, cc)][(prev_sem,curr_sem)] += 1
            
        prev_sem = curr_sem

# print(course_transitions[('427', '222')])

# keys = list(course_transitions.keys())
# values = list(course_transitions.values())
# sorted_value_index = np.argsort(values)
# sorted_transitions = {keys[i]: values[i] for i in sorted_value_index}
 
# print(sorted_transitions)

trans_graph = nx.DiGraph()

for crs in raw_counts:
    #size is num enrollment all time
    trans_graph.add_node(crs, size = raw_counts[crs])
    #TODO might be missing courses with 0 enrollment

for crs_trans in course_transitions:
    for sem_trans in course_transitions[crs_trans]:
        #weight is num students
        c0, c1 = crs_trans
        trans_graph.add_edge(c0, c1, sem = sem_trans)
        #color = sem_trans_colors[sem_trans]
        # print(trans_graph[c0][c1])

pos_ = nx.circular_layout(trans_graph)

# DEFINING GRAPH DRAWING
edge_traces = []
for edge in trans_graph.edges():
    c0, c1 = edge
    sem_trans = trans_graph[c0][c1]['sem']
    weight = np.log(course_transitions[edge][sem_trans]) + .1
    # print(sem_trans, weight)

    x0, y0 = pos_[c0]
    x1, y1 = pos_[c1]
    
    #TODO can add text/hoverinfo later
    edge_traces.append(go.Scatter(
        x = [x0, x1, None],
        y = [y0, y1, None],
        line = dict(width = weight, color = sem_trans_colors[sem_trans]),
        hoverinfo = 'none',
        mode = 'lines'
    ))

node_trace = go.Scatter(x = [],
                        y = [],
                        text = [],
                        textposition = 'top center',
                        textfont_size = 10,
                        mode = 'markers+text',
                        hoverinfo='none',
                        marker = dict(color = [],
                                      size = [],
                                      line = None))
\
for node in trans_graph.nodes():
    x, y = pos_[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['marker']['color'] += tuple(['cornflowerblue'])
    node_trace['marker']['size'] += tuple([5*np.log(trans_graph.nodes()[node]['size'])])
    node_trace['text'] += tuple(['<b>' + node + '</b>'])

fig = go.Figure()
for trace in edge_traces:
    fig.add_trace(trace)
fig.add_trace(node_trace)

fig.show()

