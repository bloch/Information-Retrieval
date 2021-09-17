from collections import Counter
import json
import math
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
from nltk.corpus import stopwords
import xml.etree.ElementTree as ET
import os
from nltk.stem import PorterStemmer

ps = PorterStemmer()
xml_names = ["cf74.xml", "cf75.xml", "cf76.xml", "cf77.xml", "cf78.xml", "cf79.xml"]
stop_words = set(stopwords.words('english'))

def get_xml_names(corpus_directory_path):
    xml_files = list()
    for filename in os.listdir(corpus_directory_path):
        if filename in xml_names:
            xml_files.append(filename)
    return xml_files

def build_index(corpus_directory_path):
    xml_names = get_xml_names(corpus_directory_path)
    inverted_index = {}
    doc_words = {}
    D = 0
    for xml_name in xml_names:
        xml_file_path = os.path.join(corpus_directory_path, xml_name)
        root = ET.parse(xml_file_path).getroot()

        records = root.findall(".//RECORD")
        D += len(records)

        for record in records:
            record_num = record.findall(".//RECORDNUM")
            title = record.findall(".//TITLE")
            abstract = record.findall(".//ABSTRACT")
            extract = record.findall(".//EXTRACT")

            title_tokens = nltk.word_tokenize(title[0].text)
            title_tokens = [ps.stem(token) for token in title_tokens if token.isalpha() and token.lower() not in stop_words]
            abstract_tokens = []
            for i in range(len(abstract)):
                tmp_abstract_tokens = nltk.word_tokenize(abstract[i].text)
                tmp_abstract_tokens = [ps.stem(token) for token in tmp_abstract_tokens if token.isalpha() and token.lower() not in stop_words]
                abstract_tokens += tmp_abstract_tokens

            extract_tokens = []
            for i in range(len(extract)):
                tmp_extract_tokens = nltk.word_tokenize(extract[i].text)
                tmp_extract_tokens = [ps.stem(token) for token in tmp_extract_tokens if token.isalpha() and token.lower() not in stop_words]
                extract_tokens += tmp_extract_tokens

            tokens = title_tokens + abstract_tokens + extract_tokens
            doc_words[int(record_num[0].text.strip())] = tokens

    for key, value in doc_words.items():
        counter = Counter(value)
        most_common = counter.most_common(1)[0]
        for word in set(value):
            if word in inverted_index:
                hash_tuple = {"doc_id": key, "tf": counter[word]/most_common[1]}
                inverted_index[word]["list"].append(hash_tuple)
                inverted_index[word]["df_i"] += 1
                inverted_index[word]["idf_i"] = math.log(D/inverted_index[word]["df_i"], 2)
            else:      #new token.
                hash_tuple = {"doc_id": key, "tf": counter[word] / most_common[1]}
                hash_dict = {"df_i": 1, "idf_i": math.log(D, 2), "list": [hash_tuple]}
                inverted_index[word] = hash_dict

    json_dict = {"words": inverted_index, "lengths": {}, "D": D}

    lengths = [0.0 for _ in range(D)]
    for token, value in inverted_index.items():
        I = value["idf_i"]
        words_list = value["list"]
        for inner_dict in words_list:
            tf = inner_dict["tf"]
            doc_id = inner_dict["doc_id"]
            lengths[doc_id - 1] += math.pow(tf*I, 2)

    lengths = [math.sqrt(lengths[i]) for i in range(len(lengths))]

    for i in range(len(lengths)):
        json_dict["lengths"][i+1] = lengths[i]

    with open('vsm_inverted_index.json', 'w', encoding='utf-8') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)