import csv
import argparse
import pandas as pd
import re
import os
import numpy as np
from sklearn.model_selection import train_test_split

# TWITTER_USERNAME = r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z0-9_]+)'
# TWITTER_URL = r'http:\/\/t\.co\/[A-Za-z0-9_]+'

def setup_args():
  parser = argparse.ArgumentParser(description='Preprocess the data.')
  parser.add_argument('--twitter', dest='twitter', action='store_const',
                     const=True, default=False,
                     help='This is Twitter data')
  parser.add_argument('--rewrite', dest='rewrite', action='store_const',
                     const=True, default=False,
                     help='Clean and write data again')
  parser.add_argument('datafile', metavar='df', help='The data file to clean')

  return parser.parse_args()

def split_hashtag(hashtag):
  hashtag_body = hashtag[1:]
  if hashtag_body.upper() == hashtag_body:
    result = '<hashtag> ' + hashtag_body + ' <allcaps>'
  else:
    result = '<hashtag> ' + ' '.join(hashtag_body.split('(?=[A-Z])'))
  return result

def clean(text, twitter=True):
  text = re.sub(r'&#8220;|&#8221;|"', '', text)
  text = re.sub(r'&#[0-9]{3,8};', ' <sym> ', text)
  text = re.sub(r'&gt;', '>', text)
  text = re.sub(r'&lt;', '<', text)

  if twitter:
    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"
    smile = eyes + nose + '[)dp]+|[(dp]+' + nose + eyes
    sadface = eyes + nose + '\(+|\)+' + nose + eyes
    neutralface = eyes + nose + '[\/|l*]'

    text = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*'," <url> ", text)
    text = re.sub("/"," / ", text) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
    text = re.sub(r'@\w+', " <user> ", text)
    text = re.sub(smile+ '|^[._]^', "<smile>", text, flags=re.IGNORECASE)
    text = re.sub(sadface, "<sadface>", text)
    text = re.sub(neutralface + '|-[._]-', "<neutralface>", text)
    text = re.sub(r'<3|&lt;3',"<heart>", text)
    text = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', "<number>", text)
    text = re.sub(r',', ' ', text)
    text = re.sub(r'#\S+', lambda x: split_hashtag(x.group()), text) 
    text = re.sub(r'([!?.<>]){2,}', lambda x: x.group()[1] + " <repeat> ", text)
    text = re.sub(r'\b(\S*?)(.)\2{3,}\b', lambda x: ''.join(x.group()) + ' <elong> ', text)
    text = re.sub(r'[A-Z]{2,}', lambda x: x.group().lower() + ' <allcaps> ', text)

  text = text.lower()
  return text

def binarize(count):
  return 1 if count > 0 else 0

def read_write_dataset(datafile, write=False, twitter=True):
  cleaned_data = datafile.split('.')[0] + '_cleaned.csv'

  if write:
    with open(datafile,'rb') as data_file:
      data = pd.read_csv( data_file, header = 0, index_col = 0, quoting = 0 )
      data['tweet'] = data.apply(lambda row: clean(row['tweet'], twitter), axis=1)
      data['hate_speech'] = data.apply(lambda row: binarize(row['hate_speech']), axis=1)
      data['offensive_language'] = data.apply(lambda row: binarize(row['offensive_language']), axis=1)
      data['neither'] = data.apply(lambda row: binarize(row['neither']), axis=1)

    with open(cleaned_data,'wb') as output_file:
      data.to_csv( output_file, header = True, quoting = 0, columns=['tweet', 'hate_speech', 'offensive_language', 'neither', 'class'] )
  
  else:
    with open(cleaned_data,'rb') as data_file:
      data = pd.read_csv( data_file, header = 0, quoting = 0, 
        dtype = {'hate_speech': np.int32, 'offensive_language': np.int32, 'neither': np.int32, 'class': np.int32} )

  return data

def save_files(prefix, tier, indices, data):
  tier_data = data.ix[indices] if indices else data

  with open(os.path.join(prefix, tier + '.x'), 'wb') as x_file:
    tier_data.to_csv( x_file, header = True, quoting = 0, columns=['tweet'] )

  with open(os.path.join(prefix, tier + '.y'), 'wb') as y_file:
    tier_data.to_csv( y_file, header = True, quoting = 0, columns=['hate_speech', 'offensive_language', 'neither', 'class'] )

def split_tier(data_prefix, data, train_percentage = 0.8, shuffle=False):
  train_i, test_i = train_test_split( data.index , train_size = train_percentage, random_state = 44 )
  save_files(data_prefix, 'train', train_i, data)
  save_files(data_prefix, 'test', test_i, data)

if __name__ == '__main__':
  args = setup_args()
  data = read_write_dataset(args.rewrite, args.datafile, args.twitter)

  print("Splitting the dataset into train and validation")
  data_prefix = os.path.join("data", "twitter_davidson")
  # split_tier(data_prefix, data, 0.8)
  save_files(data_prefix, 'all', None, data)

