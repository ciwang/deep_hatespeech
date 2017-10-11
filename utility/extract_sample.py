"""
Extract small sample for hand labeling from data file
"""

from randomAccessReader import RandomAccessReader
import random
import json
import csv

NUM_SAMPLES = 10
NUM_LINES = 84658503
IN_FILE = 'reddit/RC_2017-08'
OUT_FILE = 'reddit/sample_keywords.csv'
bufsize = 100
# NUM_SAMPLES = 1
# NUM_LINES = 5
# IN_FILE = 'reddit/RC_test'
# OUT_FILE = 'reddit/sampled_users_test.csv'

samples = []

#reader = RandomAccessReader(IN_FILE)
# for i in random.sample(xrange(NUM_LINES), NUM_SAMPLES):
#   line = json.loads(reader.get_lines(i)[0])
#   samples.append([line['body']])

with open(IN_FILE,'rb') as input_file:
  reader = csv.reader(input_file)
  while len(samples) < NUM_SAMPLES:
    lines = input_file.readlines(bufsize)
    if not lines:
      break
    for line in lines:
      comment = json.loads(line.decode('utf-8'))
      if comment['body']:
        samples.append([comment['body'].encode('utf-8')])

with open(OUT_FILE,'wb') as output_file:
  writer = csv.writer(output_file, quoting=csv.QUOTE_MINIMAL)
  writer.writerows(samples)