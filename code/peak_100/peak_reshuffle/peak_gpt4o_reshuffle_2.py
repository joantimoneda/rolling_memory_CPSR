
from openai import OpenAI 
from tqdm import tqdm, trange 
import pandas as pd
import os 
from sklearn.metrics import f1_score
import json
import math

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(_HERE)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

it = 10

data = pd.read_excel("peak_reshuffled_data_" + str(it) + ".xlsx", index_col=0)
data.nostalgic.value_counts()
data

# load conversation history
with open("peak_reshuffled_data_" + str(it) + "_PEAK_HISTORY.txt", "r", encoding="utf-8") as f:
    new_message_history = json.load(f)

new_message_history
len(new_message_history)

prompt = "Please classify the following sentence as 'nostalgic' or 'not nostalgic'. A nostalgic text should reflect \
          predominately positive emotions that are associated with recalling memories of important or momentous \
          events, usually experienced with close others. \n Return only the name of the category."

## MAIN APPROACH, keeping message history in the form of a "chat", but only rolling 100:

def no_restart(sentence):
      
    new_message_history.append({"role": "user", "content": sentence})
    
    try:      
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=new_message_history,  
            temperature=0.7,  
            max_tokens=10,  
        )
        resp = response.choices[0].message.content.strip()
        new_message_history.append({"role": "assistant", "content": resp})
        del new_message_history[1:3]
        return resp
    except:
        print(f"API Error")
    
    
sentences = data.text
sentences = [x if not isinstance(x, float) or not math.isnan(x) else " " for x in sentences]

# Run 2:
response = []    

for i in trange(0, len(sentences)):
    r = no_restart(sentences[i])
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

export.to_excel("results_peak_run2_reshuffle_" + str(it) + ".xlsx")








# RUNS TOOK: 
    # 17.28
    # 17.54
    # 17.22
# cost: 
    # $3.67







