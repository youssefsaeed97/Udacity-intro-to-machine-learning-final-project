import pickle

data_dict = pickle.load(open('NEW_data_dict.pkl', 'rb'))

for person in data_dict:

    try:
        data_dict[person]['to_poi_%'] = data_dict[person]['from_this_person_to_poi'] / data_dict[person]['from_messages']
    except:
        data_dict[person]['to_poi_%'] = 'NaN'

    try:
        data_dict[person]['from_poi_%'] = data_dict[person]['from_poi_to_this_person'] / data_dict[person]['to_messages']
    except:
        data_dict[person]['from_poi_%'] = 'NaN'

n1 = 0
n2 = 0

for person in data_dict:
    to = data_dict[person]['to_poi_%']
    frm = data_dict[person]['from_poi_%']
    if to == 'NaN': n1 += 1
    print('to:', to)
    if frm == 'NaN': n2 += 1
    print('from:', frm)

print(n1, n2)

#NEW_data_dict1 = pickle.load(open('NEW_data_dict1.pkl', "rb"))

pickle.dump(data_dict, open('UPDATED_data_dict.pkl', "wb"))
