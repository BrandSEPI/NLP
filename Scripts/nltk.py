import nltk 
import pandas as pd
import numpy as np
from collections import Counter
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import os
from os import path
from PIL import Image



def word_bag(path):
    print("0 %")

    df = pd.read_fwf("Data/"+path)
    # print(df.shape[1])
    if (df.shape[1]>1):
        maxCount = df.shape[1]
        count = 1
        while count < maxCount:
            print("delete : "+ str(df.columns[1]))
            df = df.drop(df.columns[1],axis=1)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            count+=1
        print(df)

    print("12.5 %")
    df.columns=['text']

    print("25 %")
    df["clean_text"] = df.text.apply(lambda s: ' '.join(re.sub("(w+://S+)", " ", str(s)).split()))

    def clean_text(input_txt): 
        stop = set(stopwords.words('english'))
        words = input_txt.lower().split()
        noise_free_words = [word for word in words if word not in stop] 
        alpha = [word for word in noise_free_words if word.isalpha()]
        alpha_only = " ".join(alpha) 
        return alpha_only

    print("37.5 %")
    df['clean_text'] = df['clean_text'].apply(lambda s: clean_text(s))
    #print(df)

    def tokenized_text(tt):
        tokens = nltk.word_tokenize(tt)
        return tokens

    print("50 %")
    df['clean_text'] = df['clean_text'].apply(lambda s: tokenized_text(s))
    #print(df)

    print("62.5 %")
    result = df['clean_text']
    # print(result)

    print("75 %")
    data = pd.DataFrame({'words': result})
    # print(data)


    print("87.5")
    dataf = pd.Series(sum([item for item in data.words], [])).value_counts()
    # print(dataf)
    print("100 %")
    return dataf


def word_cloud(word_bag,name):
    
    print("[.........]")
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    print("[###......]")
    alice_coloring = np.array(Image.open(path.join(d, "mask/"+name+".jpg")))
    print("[######...]")


    res = " ".join(i for i in word_bag.index)
    # print(res)
    print("[#######..]")
    WordCloud(background_color="white",mask=alice_coloring).generate(res).to_file("img/"+name+".png")
    print("[#########]")
