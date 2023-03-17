import plotly.graph_objects as go
import json
# adapted from https://plotly.com/python/sankey-diagram/


## source JSON fmt
"""
{
	"data":[{
		"type": "sankey",
		"domain": {
			"x":[0,1],
			"y":[0,1]
		},
		"orientation": "v",
		"valueformat": ".0f",
		"valuesuffix": "TWh",
		"node": {
			"pad": 15,
			"thickness": 15,
			"line": {
				"color": "black",
				"width": 0.5
			},
			"label":[
			#course names here
			...
			],
			"color":[
			"rgba(num,num,num,alpha)",
			...
			]
		},
		"link": {
			"source": [
			# nums in order
			...
			],
			"target": [
			# nums in order
			...
			],
			"value": [
			# num weights/counts for link width
			...
			],
			"color": [
			"rgba(num,num,num,alpha)",
			...
			],
			"label": [
			# list of str labels
			...
			]
		}
	}]
}
"""
with open("test_data_energy.json", "r") as inf:
	data = json.loads(inf.read())

# override gray link colors with 'source' colors
opacity = 0.4
# change 'magenta' to its 'rgba' value to add opacity
data['data'][0]['node']['color'] = ['rgba(255,0,255, 0.8)' if color == "magenta" else color for color in data['data'][0]['node']['color']]
data['data'][0]['link']['color'] = [data['data'][0]['node']['color'][src].replace("0.8", str(opacity))
                                    for src in data['data'][0]['link']['source']]

fig = go.Figure(data=[go.Sankey(
    valueformat = ".0f",
    valuesuffix = "TWh",
    # Define nodes
    node = dict(
      pad = 15,
      thickness = 15,
      line = dict(color = "black", width = 0.5),
      label =  data['data'][0]['node']['label'],
      color =  data['data'][0]['node']['color']
    ),
    # Add links
    link = dict(
      source =  data['data'][0]['link']['target'],
      target =  data['data'][0]['link']['source'],
      value =  data['data'][0]['link']['value'],
      label =  data['data'][0]['link']['label'],
      color =  data['data'][0]['link']['color']
))])
fig.update_traces(orientation=data['data'][0]['orientation'], domain={"x":[.2,.8], "y":[0,1]})

fig.update_layout(title_text="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>",
                  font_size=10)
fig.show()