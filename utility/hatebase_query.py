from json import loads
from hatebase import HatebaseAPI
import csv

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('api_key', help='Hatebase API key')
args = parser.parse_args()

HATEBASE_PAGE_LEN = 100

hatebase = HatebaseAPI({"key": args.api_key})
labels = ['about_ethnicity', 'about_nationality', 'about_religion', 'about_gender', \
            'about_sexual_orientation', 'about_disability', 'about_class']

output = "json"
query_type = "vocabulary"
page_num = 1

all_words = {}
while True:
    filters = {'language': 'eng', 'page': str(page_num)}
    response = hatebase.performRequest(filters, output, query_type)
    response = loads(response) # response -> datapoint: [obj, obj ...]
    for result in response['data']['datapoint']:
        word = result['vocabulary']
        if word not in all_words:
            all_words[word] = result

    if int(response['number_of_results_on_this_page']) < 100:
        break
    page_num += 1

with open('data/hatebase.csv', 'wb') as output_file:
    csvwriter = csv.writer(output_file)
    header = ['vocabulary', 'about_class', 'about_religion', 'about_gender', \
                'about_ethnicity', 'about_nationality', 'about_sexual_orientation', \
                'about_disability', 'offensiveness', 'number_of_sightings']
    csvwriter.writerow(header)

    for word in all_words.values():
        csvwriter.writerow([word[c].lower() for c in header])
    
