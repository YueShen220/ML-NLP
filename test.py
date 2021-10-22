docs = [] # a list of string (sentences)
with open("./data/full_texts.txt",'r',encoding='utf-8') as f:
    for line in f:
        docs.append(line)
f.close()

print(docs[12])