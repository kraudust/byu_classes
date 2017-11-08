import pandas as pd
from pdb import set_trace as stop
data = pd.read_csv('answers.csv')
text = data.Body

file = open("answers.txt", "w")
for i in xrange(10000):
    file.write(text[i])
file.close()
