For llama2 model
https://python.langchain.com/docs/integrations/llms/llamacpp
Make sure to get the one with Metal support if running on 

Example installation with Metal Support:
```
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python
```
If you have already installed a cpu only version of the package, you need to reinstall it from scratch: consider the following command:
```
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
```

Choose a model with balanced quality
https://huggingface.co/TheBloke/Llama-2-13B-GGUF
