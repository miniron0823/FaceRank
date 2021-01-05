#!/usr/bin/env python3
# Anchor extraction from HTML document
from bs4 import BeautifulSoup
from urllib.request import urlopen

response = urlopen('https://www.naver.com')
soup = BeautifulSoup(response, 'html.parser')
i = 1
f = open("C:\새파일.txt", 'w')
for anchor in soup.find_all('a'):
    data = str(i) + "위 : " + anchor.get_text() + "\n"
    i = i + 1
    f.write(data)
f.close()
    
