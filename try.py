# coding:utf-8

import re

text = """a b c 
c d d 
"""

out = re.sub('\s\n', '\n', text)

with open('text', 'w') as file:
    file.write(out)