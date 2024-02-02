import nltk 
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import os
from os import path
from PIL import Image



def word_bag(path):

    df = pd.read_fwf("sourceFolder/"+path)
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

    df.columns=['text']

    df["clean_text"] = df.text.apply(lambda s: ' '.join(re.sub("(w+://S+)", " ", str(s)).split()))

    def clean_text(input_txt): 
        stop = set(stopwords.words('english'))
        words = input_txt.lower().split()
        noise_free_words = [word for word in words if word not in stop] 
        alpha = [word for word in noise_free_words if word.isalpha()]
        alpha_only = " ".join(alpha) 
        return alpha_only

    df['clean_text'] = df['clean_text'].apply(lambda s: clean_text(s))
    #print(df)

    def tokenized_text(tt):
        tokens = nltk.word_tokenize(tt)
        return tokens

    df['clean_text'] = df['clean_text'].apply(lambda s: tokenized_text(s))
    #print(df)

    result = df['clean_text']
    # print(result)

    data = pd.DataFrame({'words': result})
    # print(data)


    dataf = pd.Series(sum([item for item in data.words], [])).value_counts()
    # print(dataf)
    return dataf


def word_cloud(word_bag, name):
    print("[.........]")
    d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    print("[###......]")
    mask_path = path.join(d, "mask/"+name+".jpg")
    mast_coloring = np.array(Image.open(mask_path)) if path.exists(mask_path) else None
    print("[######...]")

    res = " ".join(i for i in word_bag.index)
    print("[#######..]")
    WordCloud(background_color="white", mask=mast_coloring).generate(res).to_file("img/"+name+".png")
    print("[#########]")
