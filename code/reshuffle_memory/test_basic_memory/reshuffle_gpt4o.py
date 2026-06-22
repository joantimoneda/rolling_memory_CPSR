

from openai import OpenAI 
from tqdm import tqdm, trange 
import pandas as pd
import os 
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# data = pd.read_excel("data/nostalgia_data_clean_sample_new.xlsx", index_col=0)
# data.nostalgic.value_counts()


# for i in trange(0, 10):
#     data_sh = data.sample(frac=1).reset_index(drop=True)
#     data_sh.to_excel("data/nostalgia_shuffled_" + str(i) + ".xlsx")

data = pd.read_excel("data/nostalgia_shuffled_9.xlsx", index_col=0)
data.nostalgic.value_counts()


prompt = "Please classify the following sentence as 'nostalgic' or 'not nostalgic'. A nostalgic text should reflect \
          predominately positive emotions that are associated with recalling memories of important or momentous \
          events, usually experienced with close others. \n Return only the name of the category."

## MAIN APPROACH, keeping message history in the form of a "chat", but only rolling 200:

def no_restart(sentence, i):
    
    if i > 100:
        del message_history[1:3]
    
    message_history.append({"role": "user", "content": sentence}) 
    message_history_all.append({"role": "user", "content": sentence}) 
    
    try:      
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message_history,  
            temperature=0.7,  
            max_tokens=10,  
        )
        resp = response.choices[0].message.content.strip()
        message_history.append({"role": "assistant", "content": resp})
        message_history_all.append({"role": "assistant", "content": resp}) 
        return resp
    except:
        print(f"API Error")

    
message_history_all = [{"role": "system", "content": prompt}] # overall history to SAVE, not send to model

sentences = data.text

message_history = [{"role": "system", "content": prompt}] # history to send to model

#j=3

# Run 1:
response = []    

for i in trange(0, len(sentences)):
    r = no_restart(sentences[i], i)
    response.append(r)

response = [x.replace("\'", "").replace("\"", "").lower() for x in response]
response = pd.Series(response)
response.value_counts(dropna=False)
response = response.map({"nostalgic": 1, "not nostalgic": 0}).astype("Int64")

export = pd.concat([data, response], axis=1)
export.rename(columns={0:"prediction"}, inplace=True)
export = export.dropna().reset_index(drop=True)
export 

f1_score(export.nostalgic, export.prediction, labels=[0], average="weighted") # 0.904
f1_score(export.nostalgic, export.prediction, labels=[1], average="weighted") # 0.765
f1_score(export.nostalgic, export.prediction, average="weighted") # 0.869

export.to_excel("code/reshuffle_memory/test_basic_memory/reshuffle_test_20.xlsx")











