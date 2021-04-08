from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

# print(findFiles('data/names/*.txt'))

import unicodedata
import string

# grabbing all the english alphabet letters (lowercase+ uppercase) + ,.;
all_letters = string.ascii_letters + " .,;'"

n_letters = len(all_letters)

def unicodeToAscii(s):
    """
    Turn a Unicode string to plain ASCII
    """

    return ' '.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicodeToAscii('Ślusàrski'))  # turns into S l u s a r s k i

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

def readLines(filename):
    """
    Read a file and split into lines
    """

    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

    
print(all_categories)
n_categories = len(all_categories)