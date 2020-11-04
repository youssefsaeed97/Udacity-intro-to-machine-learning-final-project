# *** >> changes between pc & colab

import numpy as np
import pandas
import sys
import pickle
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.feature_selection import SelectPercentile, f_classif
from statistics import mean


sys.path.append("myedits/")
from pkl_edit import pkledit

sys.path.append("tools/")
from parse_out_email_text import parseOutText



with open(pkledit("final_project_dataset"), "rb") as data_file:
    data_dict = pickle.load(data_file)


def collect_parse(data_dict):
    '''
    collects all 'from & to emails' of each person, parse each out,
    and returns a corpus of all the emails & an email_count.
    arg: data_dict
    return: list of emails_content, dictonary of each person's email_count
    '''
    emails_content = []
    email_count = {}
    i1 = 0
    i2 = 0
    #temp_counter = 0
    missed_emails = 0

    for nth, person in enumerate(sorted(data_dict.keys())):
        print(missed_emails, 'missed emails for past identity')
        print('collecing_parsing identity number', nth+1)
        no_emails = 0
        #if temp_counter == 3: break
        #temp_counter += 1

        email_address = data_dict[person]['email_address']
        if email_address == 'NaN': continue

        # to is way more than from
        path0_0 = 'emails_by_address/from_' + email_address + '.txt'
        #path0_1 = 'emails_by_address/to_' + email_address + '.txt'

        # you can add path0_1 to the next list
        for i, path0 in enumerate([path0_0]):
            try: path_dir = open(path0)
            except:
                if i == 0: i1 += 1
                if i == 1: i2 += 1
                continue

            for path1 in path_dir:
                # *** period(.) is removed while dealing with colab
                path1 = path1.rsplit('.\n')[0]

                # *** in pc, worked without even 'try' method and all the 'except' part
                try:
                  with open(path1, "r", encoding='utf-8') as email:
                    text_content = parseOutText(email)
                    emails_content.append(text_content)
                    no_emails += 1

                except:
                    missed_emails += 1
                    continue

            path_dir.close()
            email_count[person] = no_emails

    print("no. of skipped_people's from_email out of 10:", i1)
    print("no. of skipped_people's to_email out of 10:", i2)
    print('Done with collecting/parsing')

    return emails_content, email_count


def vectorize_text(corpus):

    print('vectorizing text')
    vectorizer = TfidfVectorizer(max_df=0.3, analyzer='word', stop_words=stop_words.ENGLISH_STOP_WORDS)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    print('type:', type(tfidf_matrix))
    print('Done vectorzing')
    return tfidf_matrix


def extract_add_feature(tfidf_matrix, email_count, data_dict):
    start = 0
    new_data_dict = {}

    for nth, person in enumerate(sorted(data_dict.keys())):
        print('extracting_adding_feature for indentity no.', nth+1)

        # working with the 1st part only
        '''
        if nth+1 < 74: break
        '''

        # working with the 2nd part only
        '''
        if nth+1 < 74:
          try:
            rnge = email_count[person]
          except:
            continue
          start += rnge
          continue
        '''

        tfidf_sums = set()

        try:
            rnge = email_count[person]
        except:
            data_dict[person]['mean_top500_150tfidfsums'] = 'NaN'
            new_data_dict[person] = data_dict[person]
            continue

        tfidf_array = tfidf_matrix[np.array([index for index in range(start, start+rnge)]), :].toarray()

        for i in range(rnge):

            tfidf_sums.add(sum(sorted(tfidf_array[i], reverse=True)[:150]))

        start += rnge

        data_dict[person]['mean_top500_150tfidfsums'] = mean(set(tfidf for i, tfidf in zip(range(0, 500), sorted(tfidf_sums, reverse=True))))
        new_data_dict[person] = data_dict[person]

    print('Done extraction')
    return new_data_dict




start0 = time.time()
corpus, email_count = collect_parse(data_dict)
end0 = time.time()
print('"collect_parse" execution time:', end0 - start0)

start1 = time.time()
tfidf_matrix = vectorize_text(corpus)
end1 = time.time()
print('"vectorize_text" execution time:', end1 - start1)

start2 = time.time()
data = extract_add_feature(tfidf_matrix, email_count, data_dict)
end2 = time.time()
print('"extract_add_feature" execution time:', end2 - start2)
# for dealing with the 1st Part
'''
print(data['ALLEN PHILLIP K'])
print(data['BANNANTINE JAMES M'])
pickle.dump(data, open("NEW_data_dict1.pkl", "wb"))
'''
# for dealing with the 2nd Part
'''
try:
  print(data['ALLEN PHILLIP K'])
except:
  print('Not there')
try:
  print(data['BANNANTINE JAMES M'])
except:
  print('Not there')
pickle.dump(data, open("NEW_data_dict2.pkl", "wb"))
'''
pickle.dump(data, open("NEW_data_dict.pkl", "wb"))
# if the data was seperated then comment the above line & execute the next
'''
NEW_data_dict1 = pickle.load(open('NEW_data_dict1.pkl', "rb"))
NEW_data_dict2 = pickle.load(open('NEW_data_dict2.pkl', "rb"))
NEW_data_dict2.update(NEW_data_dict1)
pickle.dump(NEW_data_dict2, open(NEW_data_dict, "wb"))
'''
print('FINALLY DONE')
