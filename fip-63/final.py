#-----------------------------------------------------------------
from keras.models import load_model
import json
descriptions = None
with open("my_dictionary_file.txt",'r') as f:
    descriptions= f.read()
    
json_acceptable_string = descriptions.replace("'","\"")
descriptions = json.loads(json_acceptable_string)

vocab = set()
for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]

vl=list(vocab)
total_words = []
for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]

import collections

counter = collections.Counter(total_words)
freq_cnt = dict(counter)

# Sort this dictionary according to the freq count
sorted_freq_cnt = sorted(freq_cnt.items(),reverse=True,key=lambda x:x[1])

# Filter
threshold = 10
sorted_freq_cnt  = [x for x in sorted_freq_cnt if x[1]>threshold]
total_words = [x[0] for x in sorted_freq_cnt]

#-------------------------------------------------------------------------

#Since we only need image features
model_new = load_model('model_new_resnet50.h5')
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.preprocessing import image
def preprocess_img(img):
    img = image.load_img(img,target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img,axis=0)
    # Normalisation
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_img(img)    # done
    feature_vector = model_new.predict(img)  # done
    feature_vector = feature_vector.reshape((-1,))  #done
    return feature_vector

word_to_idx = {}
idx_to_word = {}

for i,word in enumerate(total_words):
    word_to_idx[word] = i+1
    idx_to_word[i+1] = word

# Two special words
idx_to_word[1846] = 'startseq'
word_to_idx['startseq'] = 1846

idx_to_word[1847] = 'endseq'
word_to_idx['endseq'] = 1847

vocab_size = len(word_to_idx) + 1
print("Vocab Size",vocab_size)
from keras.models import load_model
model = load_model('model_19.h5')

from math import log
from keras.preprocessing.sequence import pad_sequences

def predict(image, beam_width = 5, alpha = 0.7,max_len = 35):
    l = [('startseq', 1.0)]
    for i in range(max_len):
        temp = []
        for j in range(len(l)):
            sequence = l[j][0]
            prob = l[j][1]
            if sequence.split()[-1] == 'endseq':
                t = (sequence, prob)
                temp.append(t)
                continue
            encoding = [word_to_idx[word] for word in sequence.split() if word in word_to_idx]   # pending
            encoding = pad_sequences([encoding], maxlen = max_len, padding = 'post')   # done
            pred = model.predict([image, encoding])[0] 
            pred = list(enumerate(pred))
            pred = sorted(pred, key = lambda x: x[1], reverse = True)
            pred = pred[:beam_width]
            for p in pred:
                if p[0] in idx_to_word:
                    t = (sequence + ' ' + idx_to_word[p[0]], (prob + log(p[1])) / ((i + 1)**alpha))
                    temp.append(t)
        temp = sorted(temp, key = lambda x: x[1], reverse = True)
        l = temp[:beam_width]

    caption = l[0][0]
    caption = caption.split()[1:-1]
    caption = ' '.join(caption)
    return caption

temp=encode_image("download (4).jpg")   # done
temp2=temp.reshape((1,2048))
caption2=predict(temp2)
print("---------------------------------------------------------------------- Caption 2 : "+caption2)


temp=encode_image("Screenshot12.png")   # done
temp2=temp.reshape((1,2048))
caption2=predict(temp2)
print("---------------------------------------------------------------------- Caption 2 : "+caption2)


temp=encode_image("DOWNLOAD1.jpg")   # done
temp2=temp.reshape((1,2048))
caption2=predict(temp2)
print("---------------------------------------------------------------------- Caption 2 : "+caption2)