#! /usr/bin/env python3

import re
starts_with_hash = 0

# look at each line in the file use a regex to see if it starts with '#' if it does, add 1
# to the count.

with open('input.txt','r') as file:
    for line in file:
        if re.match("^#",line):
            starts_with_hash += 1
