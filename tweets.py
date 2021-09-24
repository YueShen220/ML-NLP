import json
from pathlib import Path

tweets=[]
with open('./data/tweets_us_v_0.json','r') as input_file:
    for line in input_file:
        tweets.append(json.loads(line))
input_file.close()

out_file = open("./data/full_texts.txt",'w', encoding="utf-8")
for line in tweets:
    ft = line["full_text"]
    out_file.write(ft)
out_file.close()
    