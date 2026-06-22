

from openai import OpenAI 
from tqdm import tqdm, trange 
import pandas as pd
import os 
from sklearn.metrics import f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.abspath(os.path.join(_HERE, "..", "..")))

data = pd.read_excel("data/nostalgia_shuffled_9.xlsx", index_col=0)
data.nostalgic.value_counts()

prompt = "Please classify the following sentence as containing 'nostalgic' or 'not nostalgic' ideas or \
          subtexts about politics, society, or the past. \n Return only the name of the category."

## MAIN APPROACH, keeping message history in the form of a "chat", but only rolling 200:

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def input(prompt, sentence):    
    
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "system", "content": prompt},
                {"role": "user", "content": sentence}],      
      temperature=0.7,
      max_tokens=10,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
    )

    # client.close()

    # del client

    return response.choices[0].message.content.strip()
    

sentences = data.text

j=1

# Run 1:
response = []  
    
for i in trange(0, len(sentences)):
    r = input(prompt, sentences[i])
    response.append(r)


#response = response[0]
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

export.to_excel("code/full_restart/full_restart_reshuffle/reshuffle_full_9.xlsx")

# 100%|██████████| 600/600 [07:18<00:00,  1.37it/s]
# 100%|██████████| 600/600 [06:51<00:00,  1.46it/s]
