import matplotib.pyplot as plt

def Rhist(x, bins=None, xlab='', savename='', color='w', edgecolor='k', figsize=(8,6), offset=5):
    """
    Makes histograms that look like R
    Inputs:
    - x: a numpy array or pandas series
    - bins: number of bins, default (None) is mpl default
    - xlab: text label for x axis, default '' (empty)
    - savename: full name and path of saved figure,
      if '' (default) nothing saved
    - color: fill color of bars, default 'w' (white)
    - edgecolor: outline color of bars, default 'k' (black)
    - figsize: width, heighth of figure in inches (default 8x6) 
    - offset: how far to separate axis, default=5
    """
    plt.style.use('seaborn-ticks')

    def adjust_spines(ax, spines, offset):
	"""
	This is mostly from
	https://matplotlib.org/examples/pylab_examples/spine_placement_demo.html 
	"""
	for loc, spine in ax.spines.items():
	    if loc in spines:
		spine.set_position(('outward', offset))  # outward by offset points
		spine.set_smart_bounds(True)
	    else:
		spine.set_color('none')  # don't draw spine

	# turn off ticks where there is no spine
	if 'left' in spines:
	    ax.yaxis.set_ticks_position('left')
	else:
	    # no yaxis ticks
	    ax.yaxis.set_ticks([])

	if 'bottom' in spines:
	    ax.xaxis.set_ticks_position('bottom')
	else:
	    # no xaxis ticks
	    ax.xaxis.set_ticks([])