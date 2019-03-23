#! /usr/bin/env python3

import csv

data = {'date':[], 'symbol':[], 'closing_price' : []}
with open('stocks.csv', 'r') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        data['date'].append(row["date"])
        data['symbol'].append(row["symbol"])
        data['closing_price'].append(float(row["closing_price"]))
