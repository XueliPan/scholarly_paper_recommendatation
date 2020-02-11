# python 3.7
# -*- coding: utf-8 -*-
# @Time    : 2020-01-10 22:32
# @Author  : Xueli
# @File    : text_cleaning.py
# @Software: PyCharm

import re, string, unicodedata
import nltk
import contractions
import inflect
from bs4 import BeautifulSoup
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk import wordnet
from nltk.corpus import words as words_dict
import os
import os.path


def rejoin_words_with_hyphen(input_str):
    """rejoin words that are split across lines with a hyphen in a text file"""
    input_str = input_str.replace("-\n", "")
    return input_str

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def tokenizer(text):
    """tokenize input text into a list of words"""
    words = nltk.word_tokenize(text)
    return words

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

# def replace_numbers(words):
#     """Replace all interger occurrences in list of tokenized words with textual representation"""
#     p = inflect.engine()
#     new_words = []
#     for word in words:
#         if word.isdigit():
#             new_word = p.number_to_words(word)
#             new_words.append(new_word)
#         else:
#             new_words.append(word)
#     return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def remove_words_not_in_dict(words):
    """remove words that do not exist in a certain english vocabulary,here we use WordNet"""
    new_words = []
    word_list = words_dict.words()
    for word in words:
        if word in word_list:
            new_words.append(word)
    return new_words

def joined_words(words):
    # Rejoin meaningful words as a string
    joined_words = (" ".join(words))
    return joined_words

def text_cleaning(input_str):
    input_str = rejoin_words_with_hyphen(input_str)
    input_str = replace_contractions(input_str)
    words = tokenizer(input_str)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    # words = replace_numbers(words)
    words = remove_stopwords(words)
    words = lemmatize_verbs(words)
    words = remove_words_not_in_dict(words)
    cleaned_text = joined_words(words)
    return cleaned_text

def file_cleaning(file, newRootDir):
    """input a txt file and a new directory, output a new cleaned text in a new file in new directory"""
    txtFilePath = file
    # get new cleaned text file name
    txtFileName = os.path.basename(txtFilePath)
    portion = os.path.splitext(txtFileName)
    if portion[1] == ".txt":
        cleanedTxtFileName = "cleaned_" + portion[0] + ".txt"

    # txt file cleaning
    f = open(file, 'r')
    input_str = f.read()
    output_str = text_cleaning(input_str)

    # write text into txtFileName and save to current directory()
    f = open(newRootDir + cleanedTxtFileName, "w+")
    f.write(output_str)
    f.close()
    return output_str

def iter_files(rootDir):
    """Traverse the root directory to return a list of txt file paths for all leaf nodes"""
    file_name_list = []
    for root,dirs,files in os.walk(rootDir):
        for file in files:
            file_name = os.path.join(root,file)
            file_name_list.append(file_name)
    return file_name_list

# do text cleaning for all txt files in user profiles
rootDir = '/Users/sherry/Desktop/user_profiles/user_profiles_txt_format/'
newRootDir = '/Users/sherry/Desktop/user_profiles/user_profile_after_text_cleaning/'
file_name_list = iter_files(rootDir)
for file in file_name_list:
    output_str = file_cleaning(file,newRootDir)
    print(output_str)