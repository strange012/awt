from svmmodel import test, predict
from datetime import datetime
import json
# for i in [2, 6, 8, 10, 12, 14, 15, 17, 18, 20, 22]:
#     print(test(start=datetime(2018, 1, 3), end=datetime(2018, 7, 1), hour=i, plot=False), i)


with open('input.json', 'r') as inp:
    data = json.load(inp)
prediction = []

for d in data:
    prediction.append(dict(d, **predict(**d)))

with open('output.json', 'w') as out:
    json.dump(prediction, out, indent=4)