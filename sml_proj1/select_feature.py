import pandas as pd
import numpy as np

train = pd.read_csv("train.txt", header = None)
test = pd.read_csv("test-public.txt", header = None)

follow_dict = {}
for i in range(len(train)):
    tmp_list = train[0][i].split('\t')
    follow_dict[tmp_list[0]] = tmp_list[1::]
    
all_id = set()
for i in range(len(train)):
    tmp_set = set(train[0][i].split('\t'))
    all_id.update(tmp_set)
    

all_id_list = list(all_id)
random_id = list(np.random.permutation(all_id_list))


test_list = []
for i in range(1,len(test)):
    tmp_tuple = tuple(test[0][i].split('\t'))
    test_list.append(tmp_tuple)
    
count = 0
for i in range(2000):
    if test_list[i][1] in follow_dict.keys():
        count += 1

be_followed_dict = {}
for i in range(len(train)):
    tmp_list = train[0][i].split('\t')
    for item in tmp_list[1::]:
        if item in be_followed_dict.keys():
            be_followed_dict[item].append(tmp_list[0])
        else:
            be_followed_dict[item] = [tmp_list[0]]
            
            
not_be_followed_dict = {}

for i in range(2000):
    key = test_list[i][2]
    tmp_follow_set = set(be_followed_dict[key])
    partial = set(follow_dict.keys())
    tmp_not_follow_set = partial.difference(tmp_follow_set)
    tmp_not_follow_list = list(tmp_not_follow_set)
    not_be_followed_dict[key] = tmp_not_follow_list
    

not_follow_dict = {}

for i in range(2000):
    if i % 100 == 0:
        print(i)
    key = test_list[i][1]
    tmp_follow_set = set(follow_dict[key])
    partial = set(be_followed_dict.keys())
    tmp_not_follow_set = partial.difference(tmp_follow_set)
    tmp_not_follow_list = np.random.permutation(list(tmp_not_follow_set))[0:40]
    not_follow_dict[key] = list(tmp_not_follow_list)


def X_follow_Y(X, Y):
    if X not in follow_dict.keys():
        return 1   # stand for unsure
    elif Y in follow_dict[X]:
        return 1   # stand for yes
    else:
        return 0   # stand for no


def X_follow_m_and_m_follow_Y(X, Y):       # number of m
    if X not in follow_dict.keys():
        return 0
    else:
        count = 0
        for m in follow_dict[X]:
            if m in follow_dict.keys() and Y in follow_dict[m]:
                count += 1
        return count

def jaccard_score(src, sink):
    set_src = set(be_followed_dict[src])
    set_sink = set(be_followed_dict[sink])
    inter = set_src.intersection(set_sink)
    union = set_src.union(set_sink)
    return len(inter)/len(union)

# resource allocation
def weight_score(src, sink):   
    set_src = set(be_followed_dict[src])
    set_sink = set(be_followed_dict[sink])
    inter = set_src.intersection(set_sink)
    weighted_sum = 0
    for person in inter:
        weighted_sum += 1/len(follow_dict[person])
    return weighted_sum


start = []
end = []
training_Y = []

threshold = 20

for i in range(2000):

    sink = test_list[i][2]     # sink in test set
    src = test_list[i][1]      # source in test set
    
    
    count_1 = 0
    for person in be_followed_dict[sink]:
        if count_1 == 50:
            break
        start.append(person)
        end.append(sink)
        training_Y.append(1)
        count_1 += 1
        

    count_0 = 0
    for person in not_be_followed_dict[sink]:
        if count_0 == count_1:
            break
        start.append(person)
        end.append(sink)
        training_Y.append(0)
        count_0 += 1
      
    
    count_1 = 0
    for person in follow_dict[src]:
        if count_1 == threshold:
            break
        start.append(src)
        end.append(person)
        training_Y.append(1)
        count_1 += 1
        
    count_0 = 0
    for person in not_follow_dict[src]:
        if count_0 == count_1:
            break
        start.append(src)
        end.append(person)
        training_Y.append(0)
        count_0 += 1
        
        
count = 0
f = open('node_pairs.csv','w')
for i in range(len(start)):
    s = start[i]
    e = end[i]
    label = training_Y[i]
    tmp = str(s) + ',' +  str(e) + ',' + str(label) + '\n'
    f.write(tmp)
    count += 1

for i in range(2000):
    src = test_list[i][1]
    sink = test_list[i][2]
    f.write(str(src) + ',' + str(sink) + '\n')
    count += 1
f.close()

################### for feature extraction #################


pairs = pd.read_csv("node_pairs.csv", header = None)
training_X_feature_1 = []
training_X_feature_2 = []
training_X_feature_3 = []
training_X_feature_4 = []
training_X_feature_5 = []
training_X_feature_6 = []
training_X_feature_7 = []
training_X_feature_8 = []
training_Y = []


for i in range(len(pairs)):
    
    person = str(pairs[0][i])
    sink = str(pairs[1][i])
    label = str(pairs[2][i])
    if i % 10000 == 0:
        print(i)
        
    training_X_feature_2.append(X_follow_Y(sink, person))
    tmp = X_follow_m_and_m_follow_Y(person, sink)
    training_X_feature_3.append(tmp)
    if tmp > 0:
        training_X_feature_1.append(1)
    else:
        training_X_feature_1.append(0)
    training_X_feature_4.append(jaccard_score(person, sink))
    training_X_feature_5.append(weight_score(person, sink))
    
    training_Y.append(label)
            
f = open('features_data.csv','w')
f.write('f1,f2,f3,f4,f5,label\n')
for i in range(len(training_Y)):
    f1 = training_X_feature_1[i]
    f2 = training_X_feature_2[i]
    f3 = training_X_feature_3[i]
    f4 = training_X_feature_4[i]
    f5 = training_X_feature_5[i]
    y = training_Y[i]
    tmp = str(f1) + ',' + str(f2) + ',' + str(f3) + ',' + str(f4) + ',' + str(f5) + ',' + str(y) + '\n'
    f.write(tmp)
f.close()