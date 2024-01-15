from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json

# 加载模型和tokenizer
model_name = "llama-2-70b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def model_inference(json_file):
    # 打开并读取json文件
    with open(json_file, 'r') as f:
        data = json.load(f)

    results = []
    for text, input_tokens, output_tokens in data:
        # 使用tokenizer对输入文本进行编码
        inputs = tokenizer.encode(text, return_tensors='pt')

        # 将编码的输入传递给模型进行推理
        outputs = model.generate(inputs, max_length=output_tokens, do_sample=True)

        # 将模型的输出解码为文本
        result = tokenizer.decode(outputs[0])
        results.append(result)

    return results