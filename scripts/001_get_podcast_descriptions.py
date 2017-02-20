from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

import os
import random
import time
import lxml.html

import re
import numpy as np
import pandas as pd


def random_sleep():
	time.sleep(max(random.gauss(2.5, 1), random.gauss(1.05, .1), .72))



driver = webdriver.Firefox()
driver.implicitly_wait(30)
driver.maximize_window()

# Navigate to podcast main page
driver.get("https://itunes.apple.com/us/genre/podcasts/id26?mt=2")

# Get list of genres [TODO: Map these to sub-genres]
genre_names = driver.find_elements_by_xpath("//div[@id='genre-nav']//a[@class='top-level-genre']")

# Get list of subgenres
sub_genre_names = driver.find_elements_by_xpath("//ul[@class='list top-level-subgenres']//a")

# For each subgenre, iterate through each of the top 240 podcasts
#FOR TESTING: sub_genre = sub_genre_names[0]
for sub_genre in sub_genre_names:

	# Wait a few seconds and click on the subgenre
	random_sleep()
	print(sub_genre)
	sub_genre.click()

	# obtain a list of the podcast links in that subgenre
	podcast_links = driver.find_elements_by_xpath("//div[@id='selectedcontent']//a")

	# go through each podcast's individual page to obtain detailed show
	# and episode information
	for pp in podcast_links:
		random_sleep()
		# pp = podcast_links[0]
		if not re.search('[a-zA-Z]', pp.text):
			print("Doesn't seem to be in english")
			continue

		pp.click()

		# Get main show description
		show_desc = driver.find_element_by_xpath("//div[@class='product-review']/p").text

		# Get the list of script elements with the episode descriptions
		ep_table = driver.find_element_by_xpath("//table[@class='track-list-inline-details']")
		script_els = ep_table.find_elements_by_xpath("//script")
		
		# Loop through script elements and extract JSON objects
		for sc in script_els:
			txt = sc.get_attribute('innerHTML')

			# For an unknown reason, some of the script elements pulled in 
			#	are empty or don't contain the JSON information. Ignore these.
			if re.search('release_date', txt):
				# WIP ...
				pass

			# TODO: save / parse the JSON to a table or document store

		# TODO: return to subgenre list of podcasts

	# TODO: return to full list of subgenres

# TODO: save the final table or document store
