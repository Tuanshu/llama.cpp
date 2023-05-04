./main -m ./models/gpt4all-7B/gpt4all-lora-quantized-ggml.bin -n 512 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
<!-- Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
User:西瓜是什麼?
Bob: That would be pineapple in Mandarin.
User: -->

gpt4-x-alpaca-13b-native-ggml-q4_0.pt (這是GPTQ的)沒有經過ggml轉換, 就跑不過, 副檔名可能是pt或safetensor


./main -m C:/Users/TuanShu/languege_models/ggml-gpt-x-alpaca/gpt4-x-alpaca-13b-native-ggml-q4_0.bin -n 512 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt


約需要8.2 GB memory
<!-- User:hello, can you write python code?
Bob: Certainly. Here's a sample of Python code:

def print_hello_world():
    print("Hello, World!")

User:please write a python script, to transfer a dataframe to markdown format.
Bob: Sure, I'll write a script that takes a DataFrame as input and converts it to Markdown format. Please provide an example DataFrame and the desired Markdown format.
User:dataframe: please generate random-like data; markdown format: table format  
Bob: Here's a sample of Python code that takes a DataFrame called 'df' and converts it to Markdown table format:

def to_markdown(df):
    return df.to_markdown(index=False, columns=False)

User:sun and moon, which is bigger?
Bob: The Sun is bigger than the Moon. However, the Moon appears bigger due to its elliptical orbit around the Earth. When the Moon is at its closest point to Earth (called the perigee), it appears slightly larger than average.
User:earth and moon, which is bigger?
Bob: The Earth is bigger than the Moon. However, the Moon appears bigger due to its elliptical orbit around the Earth. When the Moon is at its closest point to Earth (called the perigee), it appears slightly larger than average.

User:what is the largest country on the earth?
Bob: Russia is the largest country on Earth. However, Canada has the second-largest land area.
User: -->

<!-- User:hey, can you write a simple flask program for me as example?
Bob: Of course! Here's a simple flask program example:

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run() -->
<!-- User:西瓜是什麼?
Bob: 你想表达一切可能性的苹果一切，但是，西瓜一切都不一切。。。
User: -->



約需要8.2 GB memory~10GB (會越來越大, 無法一直運作)

./main -m C:/Users/TuanShu/languege_models/vicuna-13B-1.1-GPTQ-4bit-128g-GGML/vicuna-13B-1.1-GPTQ-4bit-128g.GGML.bin -n 512 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt


<!-- User:hey, can you write a simple flask program for me as example?
Bob: Of course. Here is a simple Flask program that displays "Hello, World!" on the main page:
```
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()
```
User: -->
 <!-- Transcript of a dialog, where the User interacts with an Assistant named Bob. Bob is helpful, kind, honest, good at writing, and never fails to answer the User's requests immediately and with precision.

User: Hello, Bob.
Bob: Hello. How may I help you today?
User: Please tell me the largest city in Europe.
Bob: Sure. The largest city in Europe is Moscow, the capital of Russia.
User:西瓜是什麼?

Bob: 西瓜是一種水果，通常是紅色或橙色的。它的味道可以用來製作各種甜點，例如蛋糕、沙爆和甜品。
User:華為是什麼?

Bob: 華為（Papaver somniferum）是一種植物，它的芬藻可以用來製作藥物，例如激素和誘導性藥物。它也被用來製作某些藥品，例如麻辣輔和舒緩劑。
User: -->





llama
alpaca: gpt4-x, 好像是社群, 在gpt4上訓練4 epoch
Vicuna: TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g-GGML,這個readme把關係講得很清楚
https://github.com/oobabooga/text-generation-webui/discussions/752
https://github.com/oobabooga/text-generation-webui/discussions/727

以下我還沒試過, 可能不用試 (還沒看到4bit, ggml版本)
https://huggingface.co/liuhaotian/LLaVA-13b-delta-v0
Model type: LLaVA is an open-source chatbot trained by fine-tuning LLaMA/Vicuna on GPT-generated multimodal instruction-following data. It is an auto-regressive language model, based on the transformer architecture.

但這個好像跟miniGPT有關係?
Intended use
Primary intended uses: The primary use of LLaVA is research on large multimodal models and chatbots.

Primary intended users: The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.

Training dataset
595K filtered image-text pairs from CC3M. 150K GPT-generated multimodal instruction-following data.

Evaluation dataset
A preliminary evaluation of the model quality is conducted by creating a set of 90 visual reasoning questions from 30 unique images randomly sampled from COCO val 2014 and each is associated with three types of questions: conversational, detailed description, and complex reasoning. We utilize GPT-4 to judge the model outputs. We also evaluate our model on the ScienceQA dataset. Our synergy with GPT-4 sets a new state-of-the-art on the dataset. See https://llava-vl.github.io/ for more details.



一些說明:  --wbits 4 --groupsize 128
這是https://github.com/qwopqwop200/GPTQ-for-LLaMa 的參數 https://github.com/IST-DASLab/gptq

順序為:
1. 原生model, 例如LLAMA
2. fine-tune, 例如alpaca
https://github.com/tatsu-lab/stanford_alpaca
Hyperparameter	LLaMA-7B	LLaMA-13B
Batch size	128	128
Learning rate	2e-5	1e-5
Epochs	3	5
Max length	512	512
Weight decay	0	0

3. 之後使用GPTQ做量化, 通常為4bit (也就是16bit -> 4bit), 好像還有3 bit的選擇
做完之後, 副檔名為pt or safetensor, 這個東西可以built for cuda or triton, 
後者是openai的類似cuda之package, 據說是pythonic
可能他們是想要用python當unified language吧
(量化時之效能需求: 待確認)


--wbits 4: 4 bit
 --groupsize 128: 
 --act-order (quantizing columns in order of decreasing activation size) 
 --true-sequential (performing sequential quantization even within a single Transformer block).
 Those fix GPTQ's strangely bad performance on the 7B model (from 7.15 to 6.09 Wiki2 PPL) 

 Wiki2 PPL	FP16	4bit-RTN	4bit-GPTQ	3bit-RTN	3bit-GPTQ	3g128-GPTQ
LLaMa-7B	5.68	6.29	6.09	25.54	8.07	6.61
LLaMa-13B	5.09	5.53	5.36	11.40	6.63	5.62
LLaMa-30B	4.10	4.54	4.45	14.89	5.69	4.80
LLaMa-65B	3.53	3.92	3.84	10.59	5.04	4.17

(perplexity scores, 困惑度, 越低越好, 數學定義類似entropy)


4. 之後再做ggml, 這個功能的目的, 是在cpu上執行NLP model
llama.cpp就是要在做過ggml之model上做, llama.cpp也有內建腳本做轉換 (轉換時之效能需求: 待確認)
(兩個repos是相同作者)

build了exe後, 就可以直接執行exe with arugment (如何在此階段串接待確認, 好像不一定要用python binding)

5. python binding
https://github.com/abetlen/llama-cpp-python


my example
```python 
# note: this package is using the package installed by "pip install llama-cpp-python", but not the package in the current directory.


import argparse
import json

from llama_cpp import Llama

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="C:/Users/TuanShu/languege_models/ggml-gpt-x-alpaca/gpt4-x-alpaca-13b-native-ggml-q4_0.bin")
args = parser.parse_args()

llm = Llama(model_path=args.model)

output = llm(
    "Question: What are the names of the planets in the solar system? Answer: ",
    max_tokens=48,
    stop=["Q:", "\n"],
    echo=True,
)

print(json.dumps(output, indent=2))
```


```python 
>>> from llama_cpp import Llama
>>> llm = Llama(model_path="./models/7B/ggml-model.bin")
>>> output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
>>> print(output)
{
  "id": "cmpl-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "object": "text_completion",
  "created": 1679561337,
  "model": "./models/7B/ggml-model.bin",
  "choices": [
    {
      "text": "Q: Name the planets in the solar system? A: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto.",
      "index": 0,
      "logprobs": None,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 14,
    "completion_tokens": 28,
    "total_tokens": 42
  }
}
```

web server:
```python 
pip install llama-cpp-python[server]
export MODEL=./models/7B/ggml-model.bin
python3 -m llama_cpp.server
```

low-level api
The low-level API is a direct ctypes binding to the C API provided by llama.cpp. The entire API can be found in llama_cpp/llama_cpp.py and should mirror llama.h.


gpt4all是用pyllamacpp, 這是他們自己做的binding



6. LangChain
https://zhuanlan.zhihu.com/p/599688026
让llm可以使用本地工具，读取本地的数据，当llm想了解更多信息时输出特定格式文本，这个框架检测到特定文本后调用本地的工具，把结果返回给llm，llm再根据新的数据做相应，直到llm觉得可以输出给用户了，再做出给用户的特定输出


7. openai API

介紹azure cloud上之gpt4 service
https://learn.microsoft.com/zh-tw/azure/cognitive-services/openai/chatgpt-quickstart?tabs=command-line&pivots=programming-language-studio


關於每次對話之token數量評估               <-- 重要, 或token和文字之關係
1 Token 約等於英文4 個字元 1 Token 約等於¾ 個單字 100 Tokens 約等於75 個單字
https://www.shopjkl.com/pages/wintogether
根據《OpenAI》的建議，你可以把每個 token 看成一個帶有 4 個字母的英文單字，每 100 個 token 約可寫出 75 字的英文文章。

輸入輸出都要token嗎?
ConversationChain 每次新队话，都是需要把历史对话重新喂给openai吧？
如果对话30轮，岂不是太消耗 token了
 

 	繁體中文	英文
每 1 個 token 的平均字數	0.49	0.84
 

 	繁體中文	英文
每個字耗費的平均 token數	2.03	1.25

因為每個中文字或英文字所消耗的 token（詞元）數量不一樣，而標點符號、空白鍵，甚至「換行」等，都代表不同的 token 數目。
比如說「你好嗎」是 3 個中文字，但使用了 7 個 token。
而「我很好」一樣也是 3 個中文字，但使用的 token 數為 6 個。


介紹autogpt
支援gpt3-5, gpt4, azure hosted gpts

和benchmark有關資訊:
https://www.listendata.com/2023/03/open-source-chatgpt-models-step-by-step.html
https://huggingface.co/docs/datasets/metrics


8. whisper model: 
https://github.com/openai/whisper
speech to text
Model	Disk	Mem
tiny	75 MB	~280 MB
base	142 MB	~430 MB
small	466 MB	~1.0 GB
medium	1.5 GB	~2.6 GB
large	2.9 GB	~4.7 GB

9. 從API講起



10. benchmarking
LLAMA 7B（Of course, if you have a larger machine(such as 3090Ti) can be replaced with a 13B, LLAMA13B is numerically superior to 175B GPT3）

11. 中文支援, LLAMA預訓練時沒有用中文資料 (這是指哪個階段?)
https://github.com/ymcui/Chinese-LLaMA-Alpaca
https://github.com/Facico/Chinese-Vicuna

注意Chinese-Vicuna不是基於Vicuna, 而是基於LLAMA 7B,
應該是撞名了

12. 英文的Vicuna / FastChat (lm-sys)
https://github.com/lm-sys/FastChat 這個是Vicuna的chatbot api
https://github.com/lm-sys/FastChat#vicuna-weights
https://github.com/Vision-CAIR/MiniGPT-4/blob/main/PrepareVicuna.md
70K conversations collected from ShareGPT.com.
https://sharegpt.com/
Vicuna-7B
This conversion command needs around 30 GB of CPU RAM. See the "Low CPU Memory Conversion" section below if you do not have enough memory.

python3 -m fastchat.model.apply_delta \
    --base /path/to/llama-7b \
    --target /output/path/to/vicuna-7b \
    --delta lmsys/vicuna-7b-delta-v1.1

他們好像也自幹了一個cpp interface

13. shareGPT一個lm-sys的網頁, 但資料庫未公開(有爭議)
https://github.com/lm-sys/FastChat/issues/90
因為lm-sys的Vicuna會說中文, 這個資料集應該有中文吧 但leak版本可能沒有, 或者藏在HTML_cleaned_raw_dataset
之後都先下載好了

14. gpt4all有trainning script
但是是train在LLAMA 7B, 原始model為16GB, not affortable now, 但如果我加裝到32GB, 那可能可以在cpu上train看看 ()


15. openai finetuning
https://vocus.cc/article/63a8accdfd8978000142c2b0
https://community.openai.com/t/how-does-fine-tuning-really-work/39972