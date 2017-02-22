from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select
from selenium.common.exceptions import NoSuchElementException

import os
import random
import time
import lxml.html
import unicodedata

import re
import numpy as np
import pandas as pd
import json


def random_sleep():
	time.sleep(max(random.gauss(2.5, 1), random.gauss(1.05, .1), .72))


# Initate driver and open Firefox window
driver = webdriver.Firefox()
driver.implicitly_wait(15)
driver.maximize_window()

# Navigate to podcast main page
driver.get("https://itunes.apple.com/us/genre/podcasts/id26?mt=2")

# Get list of genres [TODO: Map these to sub-genres]
genre_names = driver.find_elements_by_xpath("//div[@id='genre-nav']//a[@class='top-level-genre']")

# Get list of subgenres
sub_genre_els = driver.find_elements_by_xpath("//ul[@class='list top-level-subgenres']//a")
sub_genre_list =  [(i.text, i.get_attribute('href')) for i in sub_genre_els]
# Manually add subgenres that don't have subcategories
sub_genre_list.append(['Comedy', 'https://itunes.apple.com/us/genre/podcasts-comedy/id1303?mt=2'])
sub_genre_list.append(['Kids & Family', 'https://itunes.apple.com/us/genre/podcasts-kids-family/id1305?mt=2'])
sub_genre_list.append(['Music', 'https://itunes.apple.com/us/genre/podcasts-music/id1310?mt=2'])
sub_genre_list.append(['News & Politics', 'https://itunes.apple.com/us/genre/podcasts-news-politics/id1311?mt=2'])
sub_genre_list.append(['TV & Film', 'https://itunes.apple.com/us/genre/podcasts-tv-film/id1309?mt=2'])

# For each subgenre, iterate through each of the top 240 podcasts
#FOR TESTING: ss = sub_genre_list[0]; sub_idx=0
for sub_idx, ss in enumerate(sub_genre_list):
	sub_genre_name = ss[0]
	sub_genre_href = ss[1]

	# Wait a few seconds and navigate to the subgenre
	random_sleep()
	print("Stating subgenre " + sub_genre_name + ' (' + 
		str(sub_idx+1) + '/' + str(len(sub_genre_list)) + ')')
	driver.get(sub_genre_href)

	# obtain the main genre from the breadcrumbs
	genre = driver.find_elements_by_xpath("//ul[@class='list breadcrumb']/li/a")[1].text

	# obtain a list of the podcast links in that subgenre
	podcast_els = driver.find_elements_by_xpath("//div[@id='selectedcontent']//a")
	podcast_list =  [(i.text, i.get_attribute('href')) for i in podcast_els]

	# Create lists to store episode JSON objects and show information, respectively
	ep_list = []
	show_list = []

	# go through each podcast's individual page to obtain detailed show
	# and episode information
	#FOR TESTING: pp = podcast_list[0]; pod_idx=0
	for pod_idx, pp in enumerate(podcast_list):

		pod_name =  unicodedata.normalize('NFKD', pp[0]).encode('ascii', 'replace').decode()
		pod_href = pp[1]

		# Print podcast parsing progress 
		#if pod_idx % 48 == 0:
		print("Starting %d / %d: %s" % (pod_idx, len(podcast_list), pod_name))

		# TEMP: only do a few per genre
		#if pod_idx > 2:
		#	print("Stopping at 3 per subgenre")
		#	break

		# Check for english
		if not re.search('[a-zA-Z]', pod_name):
			print("Doesn't seem to be in english")
			continue

		# Navigate to the next podcast page
		random_sleep()
		driver.get(pod_href)

		# Get main show description
		show_desc = driver.find_element_by_xpath("//div[@class='product-review']/p").text

		# Do another check for UTF-8 characters
		show_desc = unicodedata.normalize('NFKD', show_desc).encode('ascii', 'replace').decode()
		if re.search('\\?\\?\\?', show_desc):
			print("Doesn't seem to be in english (based on show description)")
			continue

		# Get the list of script elements with the episode descriptions
		script_els = driver.find_elements_by_xpath("//table[@class='track-list-inline-details']//script")

		# Get rating and number of ratings 
		try:
			rating = driver.find_element_by_xpath("//div[@class='rating']/span").get_attribute('innerHTML')
			num_ratings = driver.find_element_by_xpath("//span[@class='rating-count']").text
			num_ratings = re.sub(' Ratings?', '', num_ratings)
		except NoSuchElementException:
			print("No rating for this one")
			rating = 'N/A'
			num_ratings = 'N/A'

		# Get Author (by)
		by = driver.find_element_by_xpath("//div[@class='left']/h2").text
		by = re.sub('^By ', '', by)

		# Get "More By ..."
		more_by_els = driver.find_elements_by_xpath("//div[@class='extra-list more-by']//a[@class='name']")
		more_by_vals = [i.text for i in more_by_els] 

		# Get "Listeners also subscribed to ..."
		also_sub_els = driver.find_elements_by_xpath("//div[@class='swoosh lockup-container podcast large']//a[@class='name']/span")
		assert len(also_sub_els) == 5
		also_sub_vals = [i.text	for i in also_sub_els]

		# Get website
		try:
			website = driver.find_element_by_xpath("//div[@class='extra-list']/ul/li/a").get_attribute('href')
		except NoSuchElementException:
			print("No website for this one")
			website = 'N/A'

		# Create a dict with show-level data
		pod_dict = {
			'rating': rating,
			'genre': genre,
			'subgenre': sub_genre_name,
			'podcast_name': pod_name,
			'by': by,
			'num_ratings': num_ratings,
			'also_sub_1': also_sub_vals[0],
			'also_sub_2': also_sub_vals[1],
			'also_sub_3': also_sub_vals[2],
			'also_sub_4': also_sub_vals[3],
			'also_sub_5': also_sub_vals[4],
			'website': website
		}

		# Add the "more by" podcasts, up to 10
		for idx, val in enumerate(more_by_vals):
			if idx > 10: break
			pod_dict['more_by_' + str(idx+1)] = val

		# Add the pod_dict to the show_list
		show_list.append(pod_dict)

		# Loop through script elements and extract JSON objects
		for sc_idx, sc in enumerate(script_els):
			txt = sc.get_attribute('innerHTML')

			# Check for bad script elements
			if not re.search('release_date', txt):
				print("Some script elements no good!")
				continue

			# save / parse the JSON to a table or document store, after
			# making sure that there is a group like {..}
			search = re.search('\\{.+\\}', re.sub('\n', ' ', txt))
			if search:
				jj = json.loads(search.group(0))	
				del jj['desc_popup_additional_css_classes']
				del jj['desc_popup_type']
				del jj['release_date_label']
				jj['podcast_name'] = pod_name
				ep_list.append(jj)
			else:
				print("Script element #" + str(sc_idx+1) + " appeared to have release date but didn't parse correctly!")

		# Quick pause before moving to the next one
		random_sleep()

	# Subgenre complete - print status
	print("Subgenre " + sub_genre_name + ' complete')

	# Now that we've gone through all the top podcasts in the subgenre,
	# combine the episode data and show-level data into two separate
	# pandas dataframes and export to CSV
	df_ep = pd.DataFrame(ep_list)
	df_pod = pd.DataFrame(show_list)

	df_ep.to_csv('C:/Users/jblauvelt/Desktop/projects/podcast_micro_categories/raw/ep_' + sub_genre_name + '.csv', index=False)
	df_pod.to_csv('C:/Users/jblauvelt/Desktop/projects/podcast_micro_categories/raw/pod_' + sub_genre_name + '.csv', index=False)


# Don't forget comedy! might not have been captured
# add podcast website