import sys, re
from InstantMarkup.util import *

print('<html><head><title>...</title><body>')

title = True
file = open('test_input.txt')
for block in blocks(file):
    block = re.sub(r'\*(.+?)\*', r'<em>\1</em>', block)
    if title:
        print('<h1')
        print(block)
        print('</h1>')
        title = False
    else:
        print('<p')
        print(block)
        print('</p>')

file.close()

print('</body></html>')