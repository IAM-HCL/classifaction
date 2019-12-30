import numpy as np

reviews = ''
train_rev = ''
dev_rev = ''
test_rev = ''

train_labes = ''
dev_labes = ''
test_labes = ''

##1 数据处理
lineCNT = 0
with open("./data/train.txt",'r',encoding='utf-8') as rf:
    for line in rf:
        if line != '':
            lineCNT = lineCNT + 1
            if lineCNT % 2 == 1:
                train_rev = train_rev + line
            else:            
                train_labes = train_labes + line
print(lineCNT)#16908
lineCNT = 0
with open("./data/dev.txt",'r',encoding='utf-8') as rf:
    for line in rf:
        if line != '':
            lineCNT = lineCNT + 1
            if lineCNT % 2 == 1:
                dev_rev = dev_rev + line
            else:            
                dev_labes = dev_labes + line
print(lineCNT)#2188
lineCNT = 0
with open("./data/test.txt",'r',encoding='utf-8') as rf:
    for line in rf:
        if line != '':
            lineCNT = lineCNT + 1
            if lineCNT % 2 == 1:
                test_rev = test_rev + line
            else:            
                test_labes = test_labes + line
print(lineCNT)#4400
reviews = train_rev + dev_rev + test_rev

from string import punctuation
all_text = ''.join([c for c in reviews if c not in punctuation])
#print(all_text[:1000])  str

reviews = all_text.split('\n')
all_text = ' '.join(reviews)
words = all_text.split()
reviews.pop()

train_rev = ''.join([c for c in train_rev if c not in punctuation])
train_rev = train_rev.split('\n')
train_rev.pop()

dev_rev = ''.join([c for c in dev_rev if c not in punctuation])
dev_rev = dev_rev.split('\n')
dev_rev.pop()

test_rev = ''.join([c for c in test_rev if c not in punctuation])
test_rev = test_rev.split('\n')
test_rev.pop()
'''
reviews list 句子的合集
all_text str 所有句子数据
words   list 词语合集

'''

'''
with open("./data/allText.txt",'w',encoding='utf-8') as wf:
    wf.write(all_text)
'''

## 2 get dic for encoding the words
from collections import Counter
count = Counter(words)

vocab = sorted(count,key=count.get,reverse=True)

vocab_to_int = {word:i for i,word in enumerate(vocab,1)}

print("len(vocab_to_int) = " + str(len(vocab_to_int))) #21337


## 3 encoding word and label train dev test 
seq_len = 52
from tensorflow.contrib.keras import preprocessing

train_reviews_ints = []
for each in train_rev:
    train_reviews_ints.append([vocab_to_int[word] for word in each.split()])


'''
review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

'''
'''
features = np.zeros((len(train_reviews_ints),seq_len),dtype=int)
features = preprocessing.sequence.pad_sequences(train_reviews_ints,52)

labels = train_labes.split('\n')
labels.pop()
labels = np.array([int(each) for each in labels])

np.save("./data/trainFea.npy",features)
np.save("./data/trainLabel.npy",labels)

dev_reviews_ints = []
for each in dev_rev:
    dev_reviews_ints.append([vocab_to_int[word] for word in each.split()])

features = np.zeros((len(dev_reviews_ints),seq_len),dtype=int)
features = preprocessing.sequence.pad_sequences(dev_reviews_ints,52)

labels = dev_labes.split('\n')
labels.pop()
labels = np.array([int(each) for each in labels])

np.save("./data/devFea.npy",features)
np.save("./data/devLabel.npy",labels)
'''
test_reviews_ints = []
for each in test_rev:
    test_reviews_ints.append([vocab_to_int[word] for word in each.split()])

features = np.zeros((len(test_reviews_ints),seq_len),dtype=int)
features = preprocessing.sequence.pad_sequences(test_reviews_ints,52)

labels = test_labes.split('\n')
labels.pop()
labels = np.array([int(each) for each in labels])

np.save("./data/testFea.npy",features)
np.save("./data/testLabel.npy",labels)
print(features.shape)#(2200, 52)
print(labels.shape)#(2200,)