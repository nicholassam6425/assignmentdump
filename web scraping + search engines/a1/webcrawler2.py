'''
The followinng uses selenium to scrape data about a researcher on Google Scholar.
It starts by opening a selenium webdriver and clicking the "View More" button until it is disabled. This ensures all papers are available to scrape.
It follows this up by checking if there is a "View All" coauthors button. If it does exist, it takes note of that for later, and clicks it.
If it doesn't exist, it takes note of it for later.
From here, it just scrapes data by element ID and element class.
If coauthors was expanded, it accesses different element IDs and classes.
Once the data scraping is done, it puts it all into a dictionary and outputs it to hash.json
'''

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ActionChains
import sys
import hashlib
import pathlib
import json

current_dir = pathlib.Path(__file__).parent.resolve()

url = sys.argv[1]

#hash
m = hashlib.sha256(bytes(url, 'utf-8')).hexdigest()

driver = webdriver.Chrome()
driver.maximize_window()
driver.get(url)
driver.implicitly_wait(10)

#click show more until it is disabled
view_more_papers = driver.find_element(By.ID, "gsc_bpf_more")
while (view_more_papers.is_enabled()):
    ActionChains(driver).pause(1).click(view_more_papers).pause(1).perform()

#find the view all coauthors button
if (not driver.find_elements(By.ID, "gsc_lwp_mndt_lnk")): #if driver does find this element, click it
    expanded_coauthors = True
    view_all_coauthors = driver.find_element(By.ID, "gsc_lwp_mndt_lnk")
    ActionChains(driver).pause(1).click(view_all_coauthors).pause(2).perform()
else:
    #otherwise, note that it doesn't exist
    expanded_coauthors = False

researcher_name = driver.find_element(By.ID, "gsc_prf_in").text
researcher_caption = driver.find_element(By.CLASS_NAME, "gsc_prf_il").text
researcher_institution = driver.find_element(By.CLASS_NAME, "gsc_prf_ila").text
researcher_keywords = ""
keywords_list = driver.find_element(By.ID, "gsc_prf_int").find_elements(By.CLASS_NAME, "gs_ibl")
if not keywords_list:
    for keyword in driver.find_element(By.ID, "gsc_prf_int").find_elements(By.CLASS_NAME, "gs_ibl"):
        researcher_keywords = researcher_keywords + ","
    researcher_keywords = researcher_keywords[:-1]
researcher_imgURL = driver.find_element(By.ID, "gsc_prf_pup-img").get_attribute("src")
researcher_citations = {}
researcher_hindex = {}
researcher_i10index = {}
table = driver.find_element(By.ID, "gsc_rsb_st")
rows = table.find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr")
researcher_citations['all'] = rows[0].find_elements(By.TAG_NAME,"td")[1].text
researcher_citations['since2018'] = rows[0].find_elements(By.TAG_NAME,"td")[2].text
researcher_hindex['all'] = rows[1].find_elements(By.TAG_NAME,"td")[1].text
researcher_hindex['since2018'] = rows[1].find_elements(By.TAG_NAME,"td")[2].text
researcher_i10index['all'] = rows[2].find_elements(By.TAG_NAME,"td")[1].text
researcher_i10index['since2018'] = rows[2].find_elements(By.TAG_NAME,"td")[2].text

researcher_coauthors = []
if expanded_coauthors:
    coauthors = driver.find_element(By.ID, "gsc_codb_content").find_elements(By.CLASS_NAME, "gs_scl")
    for coauthor in coauthors:
        coauth = {}
        coauth['coauthor_name'] = coauthor.find_element(By.CLASS_NAME, "gs_ai_name").find_element(By.TAG_NAME, "a").text
        coauth['coauthor_title'] = coauthor.find_element(By.CLASS_NAME, "gs_ai_aff").text
        coauth['coauthor_link'] = coauthor.find_element(By.CLASS_NAME, "gs_ai_name").find_element(By.TAG_NAME, "a").get_attribute("href")
        researcher_coauthors.append(coauth)
else:
    coauthors = driver.find_element(By.CLASS_NAME, "gsc_rsb_a").find_elements(By.TAG_NAME, "li")
    for coauthor in coauthors:
        coauth = {}
        coauth['coauthor_name'] = coauthor.find_element(By.CLASS_NAME, "gsc_rsb_a_desc").find_element(By.TAG_NAME, "a").text
        coauth['coauthor_title'] = coauthor.find_element(By.CLASS_NAME, "gsc_rsb_a_desc").find_element(By.CLASS_NAME, "gsc_rsb_a_ext").text
        coauth['coauthor_link'] = coauthor.find_element(By.CLASS_NAME, "gsc_rsb_a_desc").find_element(By.TAG_NAME, "a").get_attribute("href")
        researcher_coauthors.append(coauth)

researcher_papers = []
papers_table = driver.find_element(By.ID, "gsc_a_b")
for row in papers_table.find_elements(By.TAG_NAME, "tr"):
    paper = {}
    paper['paper_title'] = row.find_element(By.CLASS_NAME, "gsc_a_at").text
    paper['paper_authors'] = row.find_element(By.CLASS_NAME, "gsc_a_t").find_elements(By.TAG_NAME, "div")[0].text
    paper['paper_journal'] = row.find_element(By.CLASS_NAME, "gsc_a_t").find_elements(By.TAG_NAME, "div")[1].text
    paper['paper_citedby'] = row.find_element(By.CLASS_NAME, "gsc_a_c").find_element(By.TAG_NAME, "a").text
    paper['paper_year'] = row.find_element(By.CLASS_NAME, "gsc_a_y").find_element(By.TAG_NAME, "span").text
    researcher_papers.append(paper)

researcher_dict = {
    "researcher_name": researcher_name,
    "researcher_caption": researcher_caption,
    "researcher_institution": researcher_institution,
    "researcher_keywords": researcher_keywords,
    "researcher_imgURL": researcher_imgURL,
    "researcher_citations": researcher_citations,
    "researcher_hindex": researcher_hindex,
    "researcher_i10index": researcher_i10index,
    "researcher_coauthors": researcher_coauthors,
    "researcher_papers": researcher_papers
}

#write json file
with open(m + ".json", "w", encoding='utf-8') as f:
    json.dump(researcher_dict, f)

#write the txt file
with open(m + ".txt", "w", encoding='utf-8') as f:
    f.write(driver.page_source)

driver.quit()