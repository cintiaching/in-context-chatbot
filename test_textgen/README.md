Attempt to use [text-generation-webui](https://github.com/oobabooga/text-generation-webui) for the chatbot

Reason:
- text-generation-webui contains multiple model loaders, e.g. ExLlamaV2, AutoGPTQ, for quantization of llm, which will increase performance of influence.
- has been integrated with langchain: [TextGen](https://python.langchain.com/docs/integrations/llms/textgen)

## Install (mac)
1. install

```
git clone https://github.com/oobabooga/text-generation-webui.git
cd text-generation-webui
pip install -r requirements.txt

pip install -U --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
```
2. Run the `start_macos.sh`

3. download a llm

https://github.com/oobabooga/text-generation-webui/blob/main/download-model.py

Downloads models from Hugging Face to models/username_modelname.

```
python download-model.py facebook/opt-1.3b
```

4. start the web ui

```
cd ~/text-generation-webui
python server.py
```

4.select a model in the web ui
5.press control-c in the terminal to shut it down

## Use with langchain
https://python.langchain.com/docs/integrations/llms/textgen

Get the OpenAI-compatible API URL using:

```
python server.py --listen --api --api-port 1000
```


