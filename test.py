import sys
import pickle

sys.path.append("../myedits/")
from pkl_edit import pkledit

'''
with open(pkledit("corpus"), "rb") as data_file:
    corpus = pickle.load(data_file)
'''

'''
corpus = pickle.load(open('corpus.pkl', "rb"))
'''

'''
features = ['mean_top500_150tfidfsums', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees', 'to_messages', 'email_address', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
'''

'''
data_dict = pickle.load(open('NEW_data_dict.pkl', "rb"))
[data_dict.pop(key, 0) for key in ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK']]
'''

'''
num = {}
for person in data_dict:
    for feature in features:
        if data_dict[person][feature] == "NaN":
            num[feature] = num.get(feature, 0) + 1

for feat in num:
    print(feat, num[feat])
'''
'''
NEW_data_dict1 = pickle.load(open('NEW_data_dict1.pkl', "rb"))
NEW_data_dict2 = pickle.load(open('NEW_data_dict2.pkl', "rb"))
NEW_data_dict2.update(NEW_data_dict1)
pickle.dump(NEW_data_dict2, open('NEW_data_dict.pkl', "wb"))
'''
