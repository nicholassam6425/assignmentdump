'''
The following program uses regex to determine where a page's main content area is.
The main content area is defined by the area that maximizes the number of words, while minimizing the number of tags.
It is done by first fetching the page source, then converting all tags to 1's, and all words to 0's. 
After this conversion, the output is cleaned of any remaining characters that aren't 1 or 0.
From here, it iterates through all i and j, where 0 <= i < j <= N, where N is the number of words and tags, and inputs them into the f(i, j) which maximizes tags before i, maximizes words between i and j, and maximizes tags after j.
Once i and j have been determined, it counts the number of 0's before i, and the number of 0's between i and j to determine where the main content is located.
Once the main content has been located, it outputs the main content to hash.txt
Afterwards, it uses matplotlib to plot a graph of the values of f at various i's and j's
'''

from selenium import webdriver
import sys
import hashlib
import pathlib
import re
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

current_dir = pathlib.Path(__file__).parent.resolve()

url = sys.argv[1]

#hash
m = hashlib.sha256(bytes(url, 'utf-8')).hexdigest()

driver = webdriver.Chrome()
driver.maximize_window()
driver.get(url)
driver.implicitly_wait(10)

deleted_digits = re.sub('[10]+', '', driver.page_source) #delete all 1's and 0's from the page 
tags_replaced = re.sub('<[^>]*>', '1', deleted_digits) #replace all tags with 1
words_replaced = re.sub('[^\d\W]+', '0', tags_replaced) #replace everything that isn't a digit with 0
cleaned = re.sub('[^10]', '', words_replaced)

with open(m + ".html", "w", encoding='utf-8') as f:
    f.write(driver.page_source) #write page source into html
driver.quit()
def f(i, j):
    f = 0
    for char in cleaned[:i]:
        f += int(char)
    for char in cleaned[i:j]:
        f += 1 - int(char)
    for char in cleaned[j:]:
        f += int(char)
    return f
print(cleaned)
length = len(cleaned)
max_f = f(0,1)
max_i = 0
max_j = 1
all_f = [ [0]*length for i in range(length)]
for i in range(0, len(cleaned)):
    for j in range(i+1, len(cleaned)):
        current_f = f(i, j)
        all_f[i][j] = current_f
        if current_f > max_f:
            max_f = current_f
            max_i = i
            max_j = j
print(f'i: {max_i}, j: {max_j}, f: {max_f}')


words = re.findall(r'[^1\W]+', tags_replaced)
main_content_start = 0
for char in cleaned[:max_i]:
    if char == '0':
        main_content_start += 1
main_content_len = 0
for char in cleaned[max_i:max_j]:
    if char == '0':
        main_content_len += 1
print(f'start: {main_content_start}, len: {main_content_len}')
print(words[main_content_start:main_content_start+main_content_len])
with open(m + ".txt", "a", encoding='utf-8') as f:
    for word in words[main_content_start:main_content_start+main_content_len]:
        f.write(word + " ")

all_f = np.array(all_f)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y = np.meshgrid(np.arange(all_f.shape[0]),np.arange(all_f.shape[1]))
ax.plot_surface(x,y,all_f)
ax.set_xlabel('i')
ax.set_ylabel('j')
ax.set_zlabel('f(i, j)')
plt.show()

plt.imshow(all_f, cmap='hot', interpolation='nearest')
plt.show()