import csv

data = [[1,2,3],[2,3,4],[3,4,5]]
with open('test_csv.csv', mode='w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for item in data:
        writer.writerow(item)
