import csv
import argparse
import pandas as pd
import re

# TWITTER_USERNAME = r'(?<=^|(?<=[^a-zA-Z0-9-\.]))@([A-Za-z0-9_]+)'
# TWITTER_URL = r'http:\/\/t\.co\/[A-Za-z0-9_]+'

parser = argparse.ArgumentParser(description='Preprocess the data.')
parser.add_argument('--twitter', dest='twitter', action='store_const',
                   const=True, default=False,
                   help='This is Twitter data')
parser.add_argument('datafile', metavar='df', help='The data file to clean')

args = parser.parse_args()

def split_hashtag(hashtag):
  hashtag_body = hashtag[1:-1]
  if hashtag_body.upper() == hashtag_body:
    result = '<hashtag> ' + hashtag_body + ' <allcaps>'
  else:
    result = '<hashtag> ' + ' '.join(hashtag_body.split('(?=[A-Z])'))
  return result

def clean(text):
  text = re.sub(r'&#8220;|&#8221;|"', '', text)
  text = re.sub(r'&#[0-9]{3,6};', ' <sym> ', text)

  if args.twitter:
    # Different regex parts for smiley faces
    eyes = "[8:=;]"
    nose = "['`\-]?"
    smile = eyes + nose + '[)d]+|[)d]+' + nose + eyes
    lolface = eyes + nose + 'p+'
    sadface = eyes + nose + '\(+|\)+' + nose + eyes
    neutralface = eyes + nose + '[\/|l*]'

    text = re.sub(r'https?:\/\/\S+\b|www\.(\w+\.)+\S*',"<url>", text)
    text = re.sub("/"," / ", text) # Force splitting words appended with slashes (once we tokenized the URLs, of course)
    text = re.sub(r'@\w+', "<user> ", text)
    text = re.sub(smile, "<smile>", text, flags=re.IGNORECASE)
    text = re.sub(lolface, "<lolface>", text, flags=re.IGNORECASE)
    text = re.sub(sadface, "<sadface>", text)
    text = re.sub(neutralface, "<neutralface>", text)
    text = re.sub(r'<3',"<heart>", text)
    text = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', "<number>", text)
    text = re.sub(r'#\S+', lambda x: split_hashtag(x.group()), text) 
    text = re.sub(r'([!?.]){2,}', lambda x: x.group()[1] + " <repeat>", text)
    text = re.sub(r'\b(\S*?)(.)\2{3,}\b', lambda x: ''.join(x.group()) + ' <elong>', text)
    text = re.sub(r'[A-Z]{2,}', lambda x: x.group().lower() + ' <allcaps>', text)

  text = text.lower()
  return text

def binarize(count):
  return 1 if count > 0 else 0

with open(args.datafile,'rb') as input_file:
  data = pd.read_csv( input_file, header = 0, index_col = 0, quoting = 0 )
  data['tweet'] = data.apply(lambda row: clean(row['tweet']), axis=1)
  data['hate_speech'] = data.apply(lambda row: binarize(row['hate_speech']), axis=1)
  data['offensive_language'] = data.apply(lambda row: binarize(row['offensive_language']), axis=1)
  data['neither'] = data.apply(lambda row: binarize(row['neither']), axis=1)

# outfile_name = args.datafile.split('.')[0] + '_cleaned.csv'
# with open(outfile_name,'wb') as output_file:
#   data.to_csv( output_file, header = True, quoting = 0, columns=['tweet', 'hate_speech', 'offensive_language', 'neither', 'class'] )
