import os
import zipfile
import re

os.chdir('C:/Users/jblauvelt/Desktop/projects/podcast_micro_categories/raw')

# List all the files in the directory and separate zip files from data files
files = os.listdir()
zip_files = [i for i in files if re.search('\\.zip$', i)]
data_files = [i for i in files if re.search('\\.csv$', i)]

# Identify the subgenre for each datafile
data_file_subgenres = [re.sub('ep\\_|pod\\_|\\.csv$', '', i) for i in data_files]

# For each subgenre, put the data files for the subgenre into their own zip
for subgenre in set(data_file_subgenres):
	print('Zipping ' + subgenre)
	
	# Check if zip file already exists, and if so, skip
	if [i for i in zip_files if i == subgenre + '.zip']:
		print("Zip already exists - moving on")
		continue

	# Create the zip
	with zipfile.ZipFile(subgenre + '.zip', 'w', zipfile.ZIP_DEFLATED) as zz:
		# Write all files with that subgenre (should just be two - ep_ and pod_)
		for idx, fl in enumerate(data_files): 
			if data_file_subgenres[idx] == subgenre:
				zz.write(fl)

		# After all files have been added, close the zip
		zz.close()