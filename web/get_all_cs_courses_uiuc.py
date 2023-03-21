from bs4 import BeautifulSoup as bsoup
from urllib.request import urlopen
import re
import pickle
import binascii

""" Course info is stored online by links to programs of study
 which are the XXX or XXXX alphabetic categories from the 
 course titles (e.g. ACCY == Accountancy).  
 Each program is a link in a <div> with class = <atozindex>.
 
 By looping over these links we can pull the individual course
 info from those pages.
 
 For this project, we're interested in the program of study,
 the course title, and the text description of what is covered
 in the course.  We'll pull all of the course info to use the 
 text as a toy dataset and save it to file.
"""


# Illinois course catalog page with all CS courses offered/cataloged
soup = bsoup(urlopen('http://catalog.illinois.edu/courses-of-instruction/cs/'),features='html.parser')

complete_catalog = {}

# homepage which all course pages are linked from
root = 'http://catalog.illinois.edu'

# regex patterns to pull text from the html tags.
# courses are stored with format:
# Title (space) number (space, quote) Course name text (spaec, quote) credit: number hours.
# XXX(X)\xa0 ### \xE2\x80 Course name text \xE2\x80 credit: # hours.

# course title (e.g. ACCY 201)
cs = re.compile('[A-Z]+(Â| |\xa0)+?\d{1,4}')
# course name (e.g. Introduction to Accountancy Ethics).
# only pull second capture group here. 
cn = re.compile('[A-Z]+(Â| |\xa0)+?\d{1,4}(.*?)(credit)')
# extracting credit hours info from description as they 
# don't add anything for the purpose of this dataset
sp = re.compile('\d (under)?graduate hours.')

# identifying course crosslistings.
is_cross = re.compile(r'Same as (?P<cref>[A-z]+ \d{1,4})')
crossref = re.compile(r'.( )+?See (?P<crossdesc>[A-z]+ \d{2,})')

# remove non-breaking spaces
nbsp = re.compile(r'\s+')


# loop over courses on the CS page to get titles, names, and descriptions
for course in soup.find_all('div', class_='courseblock'):
	# drill down into each course listing to get name and link 
	course_title = course.find('p', class_='courseblocktitle')
	title = course_title.text
	
	# # For debug. un-comment to save list of 
	# # course titles to text file.
	# with open('titles2.text', 'a') as outt:
		# outt.write(title+"\n")
	
	# split out course text into title and name
	# grab course title from regex search result
	res = cs.search(title)
	if not res:
		print("fail no title")
		continue
	
	course_title = res.group(0).strip()
	# replace non-encodable characters with spaces (mostly quotes and nbsp)
	course_title = re.sub(r'[^\x00-\x7F]+',' ', course_title)
	print(course_title)
	# grab course name from regex search result
	coursename = cn.match(title)
	coursename = coursename.group(2).strip()
	# print("    "+course_title)
	
	# now pull description text 
	course_descr = course.find('p', class_='courseblockdesc')
	desc = nbsp.sub(' ', course_descr.text).strip()
	
	# detect if the course is crosslisted with another department, and if so, grad the descriptions from there!
	crosslisted = is_cross.match(desc)
	if crosslisted is not None:
		print(f"Got a crosslisting for {coursename}...")
		for hlink in course_descr.find_all('a', class_='bubblelink code'):
			# there may be a few other anchors on the page which do not link 
			# to course pages.  Skip them.
			try:
				hlink['href']
			except KeyError:
				print("no link")
				continue
			
			# pull HTML from program link 
			cross_soup = bsoup(urlopen(root + hlink['href']),features='html.parser')
			# now pull description text 
			course_descr = cross_soup.find('p', class_='courseblockdesc')
			desc = course_descr.text.strip()
			
			# several courses may be cross listed multiple times (e.g. CS107 = IS107 = STAT107)
			# in this case, STAT107 is the only one with an actual description. So check this condition
			# and keep going if needed.
			if is_cross.match(desc) is not None:
				continue
	
	# trim and compile description for storage in dictionary
	if len(desc) < 1:
		print("WARNING: no desc")
		complete_catalog[course_title] = (coursename," ")
		continue
	else:
		desc = re.sub(r'[^\x00-\x7F]+',' ', desc)
		hours = sp.search(desc)
		if not hours:
			complete_catalog[course_title] = (coursename,desc.replace('"','\''))
			# print(f"###### {course_title}: {coursename}: {desc[:20]}")
		else:
			hours = sp.finditer(desc)
			to_remove = []
			for rem in hours:
				remove = rem.span()
				to_remove.append(desc[remove[0]:remove[1]])
			trimmed = re.sub(re.escape(r'|'.join(to_remove)), '', desc)
			trimmed = trimmed.replace('"','\'')
			
			complete_catalog[course_title] = (coursename,trimmed)
				

# save results to file
# NOTE: Excel demands no space between comma and doublequote chars for 
# text to be interpreted as raw text
with open('uiuc_cs_courses2.csv', 'w', encoding="utf-8") as outf:
	outf.write('Title,Course,Description\n')
	for title,(course,descr) in complete_catalog.items():
		# print(f"{title}: {course}: {descr[:20]}")
		outf.write(title+',"'+course+'","' + descr + '"\n')