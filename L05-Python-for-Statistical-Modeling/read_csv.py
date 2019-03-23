#! /usr/bin/env python3

import pandas

data2 = pandas.read_csv('stocks.csv', delimiter='\t',header=None)
len(data2)
type(data2)
