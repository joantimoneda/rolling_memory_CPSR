
from openai import OpenAI 
from tqdm import tqdm, trange 
import pandas as pd
import os 
from sklearn.metrics import f1_score
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

it = 10

data = pd.read_excel("../../../data/nostalgia_data_clean_sample.xlsx", index_col=0)
data = data.sample(frac=1).reset_index(drop=True)

filename = "peak_reshuffled_data_" + str(it) + ".xlsx"
if not os.path.exists(filename):
    data.to_excel(filename)
    print(f"Saved: {filename}")
else:
    print(f"ERROR: NOT SAVED. File already exists!")

data.nostalgic.value_counts()
data

prompt = "Please classify the following sentence as 'nostalgic' or 'not nostalgic'. A nostalgic text should reflect \
          predominately positive emotions that are associated with recalling memories of important or momentous \
          events, usually experienced with close others. \n Return only the name of the category."

## MAIN APPROACH, keeping message history in the form of a "chat", but only rolling 100:

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

export.to_excel("results_peak_run1_reshuffle_" + str(it) + ".xlsx")


## find stetch of 100 obs with best performance and save it
f1_split = []

for i in range(0, (len(export)-100)): 
    split = export.iloc[i:i+100,:]
    f1_split.append([f1_score(split.nostalgic, split.prediction, average="weighted")]) 
    
f1 = pd.concat([pd.DataFrame(x) for x in f1_split], ignore_index=True)
f1.idxmax() # initial index for 100-obs best performance stetch

peak_history = message_history_all[(f1.idxmax()*2-1)[0]:((f1.idxmax()*2-1)[0]+200)] #multiply by 2 because for each message there is an assistant response
new_message_history = [message_history_all[0]] + peak_history

filename = "peak_reshuffled_data_" + str(it) + "_PEAK_HISTORY.txt"
if not os.path.exists(filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(new_message_history, f)
        print(f"Saved: {filename}")
else:
    print(f"ERROR: NOT SAVED. File already exists!")














