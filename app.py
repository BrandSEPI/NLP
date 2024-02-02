from Scripts import nltk
from os import listdir


files = [f for f in listdir("Data")]
# print(files)

for item in listdir("Data"):
    item_bag = nltk.word_bag(item)
    print(item_bag)
    nltk.word_cloud(item_bag,item)
# test = nltk.word_bag('Data/An Introductory Course of Quantitative Chemical Analysis.txt')
# print(test)
# print(test)

