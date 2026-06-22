

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



prompt = "Please classify the following sentence as containing 'nostalgic' or 'not nostalgic' ideas or subtexts about politics, society, or the past. Here are some examples: \n \
        Text: We disagree with the globalization that aims to extinction of local and national cultures and traditions, \
        promoting enforce their replacement with a single system of values ​​based on multiculturalism and religious syncretism. \n Answer: Nostalgic  \n \
        Reasoning: The sentence expresses a longing or preference for the preservation of local and national cultures and traditions, suggesting a \
        resistance to change brought by globalization. This aligns with a nostalgic sentiment, as it values traditional systems and may reflect \
        a desire to maintain or return to a previous state in which these local cultures were more prominent or uninfluenced by global homogenization.  \n \
        Text: The illegal contracts pocket nullity sanctions should be to prevent the land registration records maintained. \n Answer: not nostalgic \n \
        Reasoning: The sentence does not appear to contain nostalgic ideas or subtexts about politics, society, or the past. It discusses a current \
        legal issue related to land registration and contract law, focusing on the enforcement of sanctions and maintenance of records. There is no \
        reference to or sentiment about a previous era or way of life. Therefore, it is 'not nostalgic'.\n \
        Text: Of course, the balance between economic viability of livestock and environmental concerns must be preserved as much as possible. \n Answer: not nostalgic \n \
        Reasoning: The sentence discusses the present need to balance economic viability with environmental concerns, indicating a forward-looking approach \
        rather than a longing for the past. There is no mention of a return to previous practices or idealized times, nor does it reflect on how things used to be. \
        Instead, it focuses on addressing current issues and maintaining sustainability. \
        Text: It is the only party with stability and consistency has demonstrated for over nine decades that is by the people and would not betray him. \n Answer: nostalgic \n \
        Reasoning: The sentence references a political party's \"stability and consistency\" over \"over nine decades,\" suggesting a long-standing, reliable, and \
        trusted history. This evokes a sense of nostalgia by idealizing the past achievements and reliability of the party. It implies a yearning for or \
        appreciation of these long-standing qualities as a reason to trust the party currently, which is typical of nostalgic rhetoric in politics \
        Text: When the 1945 Labour government established the NHS, it created one of the central institutions of fairness of the 20th century. \n Answer: nostalgic \n \
        Reasoning:The sentence reflects a nostalgic idea by looking back at a historical event, the establishment of the NHS by the 1945 Labour government, and \
        framing it as an act of fairness and a pivotal achievement of the 20th century. This retrospection and the positive appraisal imply a longing or appreciation \
        for the past political action and its impact on society. \
        Text: the introduction of the dual system of vocational education \n Answer: not nostalgic \n \
        Reasoning: The sentence merely mentions the introduction of a system, specifically the dual system of vocational education, without expressing any sentiment of \
        longing or idealization of the past. It does not reflect on any past conditions or imply a preference for things as they were previously. Instead, it appears to \
        be stating a factual point or development, lacking any inherent nostalgic ideas or subtexts. \
        Text: The industrialization of the 19th century and the impoverishment associated more of the population has already shown that \
        full freedom of contract can lead to socially intolerable results.  \n Answer: not nostalgic \n \
        Reasoning: This sentence does not convey nostalgia but instead offers a critical analysis of the past. It reflects on the negative consequences of the 19th century's \
        industrialization and associated social issues, such as the impoverishment of the population. The focus is on the detrimental outcomes of unchecked \"freedom of contract,\" \
        suggesting a need to learn from history to avoid repeating such mistakes. It does not idealize or romanticize past conditions but rather highlights the lessons learned from them.\
        Text: In this context, we insist on heritage recovery built on previous resettlement policies, where landscaping, infrastructure or social facilities were ignored. \n Answer: nostalgic \n \
        Reasoning: The sentence expresses a desire to recover heritage, indicating a valuing of past elements that may have been neglected in previous resettlement policies. By emphasizing \"heritage recovery,\" \
        it suggests a recognition of the importance of past cultural or historical elements, contrasting them with past policies that ignored aspects like landscaping, infrastructure, or \
        social facilities. This implies a wish to restore or revisit past values or practices that were previously overlooked, which aligns with a nostalgic perspective. \
        Now classify the following sentence as 'nostalgic' or 'not nostalgic'. Return only the name of the category:"


def zero_shot(sentence, i):

    message_history = [{"role": "system", "content": prompt},
                       {"role": "user", "content": sentence}]

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=message_history,  
            temperature=0.7,  
            max_tokens=10,
        )
        resp = response.choices[0].message.content.strip()
        return resp
    except:
        print(f"API Error")

    
message_history_all = [{"role": "system", "content": prompt}] # overall history to SAVE, not send to model

sentences = data.text

#j=3

# Run 1:
response = []    

for i in trange(0, len(sentences)):
    r = zero_shot(sentences[i], i)
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

export.to_excel("code/no_memory_runs/test_fs/reshuffle_fs_20.xlsx")











