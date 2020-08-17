import json
f=open("./lm_output/test_results.json")
sentence={}
line=f.read()
text=json.loads(line)
for sen in text:
    s="".join([sen["tokens"][i]["token"] for i in range(len(sen["tokens"]))])
    if sen["ppl"]<=10:
        sentence[s]=sen["ppl"]
print(len(sentence))