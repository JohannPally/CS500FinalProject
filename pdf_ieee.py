import PyPDF2

print("Opening PDF...", end = "")
pdf_file = open('res\\acm_se2014.pdf', 'rb')
print("...", end = "")
read_pdf = PyPDF2.PdfFileReader(pdf_file)
number_of_pages = read_pdf.getNumPages()
terms = []
row = 0
to_remove = []
print("\nReading PDF...", end = "")
for p in range(27, 37):
	page = read_pdf.getPage(p)
	page_content = page.extractText()
	text_list = page_content.split('\n')
	# text_list = list(map(lambda e:e.replace('\n',''), text_list))
	# for i,e in enumerate(text_list):
		# if len(e) > 0:
			# if e[0].islower():
				# text_list[i] = ' '.join([text_list[i-1], text_list[i]])
				# to_remove.append(i-1)
	# # print(".", end = "")
	# text_list = [e for i,e in enumerate(text_list) if i not in to_remove]
	
	for term in text_list:
		# terms.append(["","","","",""])
		print(term)
		
# print("\nSaving results to csv...", end = "")
# with open('results.csv', 'w') as outfile:
	# for i in range(len(terms)):
		# for each in terms[i]:
			# outfile.write(each)
			# outfile.write(",")
		# outfile.write("\n")
# print("\nDone.")
pdf_file.close()