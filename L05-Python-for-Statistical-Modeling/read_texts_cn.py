#! /usr/bin/python3


data = []
with open('texts-data.txt','r') as file:
    for line in file:
        data.append(line)

print(data)
