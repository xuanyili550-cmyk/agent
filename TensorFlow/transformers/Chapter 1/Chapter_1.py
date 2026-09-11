from transformers import (pipeline,BertTokenizer,AutoTokenizer,
                          AutoModel,AutoModelForSequenceClassification,BertModel)
import torch
tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
encoded_input=tokenizer("Hello, I'm a single sentence!")
print(encoded_input)

#{'input_ids': [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102],
 #'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
 #'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

#input_ids：令牌的数值表示
#token_type_ids：这些参数告诉模型输入的哪一部分是句子 A，哪一部分是句子 B。
#attention_mask：这指示哪些标记应该被关注，哪些不应该被关注。
tokenizer.decode(encoded_input["input_ids"])

encoded_input = tokenizer("How are you?", "I'm fine, thank you!")
print(encoded_input)

#{'input_ids': [[101, 1731, 1132, 1128, 136, 102], [101, 1045, 1005, 1049, 2503, 117, 5763, 1128, 136, 102]],
 #'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
 #'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]}
 #当传递多个句子时，分词器会为每个句子返回一个列表，对应字典中的每个值。我们还可以让分词器直接从 PyTorch 返回张量
encoded_input = tokenizer("How are you?", "I'm fine, thank you!", return_tensors="pt")
print(encoded_input)

#{'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102],
#         [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]),
 #'token_type_ids': tensor([[0, 0, 0, 0, 0, 0],
 #        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 #'attention_mask': tensor([[1, 1, 1, 1, 1, 1],
 #        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}


#填充输入
#如果我们要求分词器对输入进行填充，它会通过向比最长句子短的句子添加特殊的填充标记，使所有句子的长度相同：
encoded_input = tokenizer(
    [ "How are you?" , "I'm fine, thank you!" ], padding= True , return_tensors= "pt"
)
print (encoded_input)

#{'input_ids': tensor([[  101,  1731,  1132,  1128,   136,   102,     0,     0,     0,     0],
#         [  101,  1045,  1005,  1049,  2503,   117,  5763,  1128,   136,   102]]),
 #'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]),
 #'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
 #        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
#填充标记已被编码为 ID 为 0 的输入 ID，并且它们的注意力掩码值也为 0。
# 这是因为这些填充标记不应该被模型分析：它们不属于实际句子的一部分



#截断输入
#张量可能过大，模型无法处理。例如，BERT 仅使用长度不超过 512 个 token 的序列进行预训练，
# 因此无法处理更长的序列。如果序列长度超过模型的处理能力，则需要使用以下truncation参数对其进行截断
encoded_input = tokenizer(
    "This is a very very very very very very very very very "
    "very very very very very very very very very very very very very very very very very very "
    "very very very very very very very very very very very very very very very very very "
    "very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])

#通过结合填充和截断参数，您可以确保张量具有所需的确切大小
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)

#添加特殊标记
#特殊标记（或者至少是它们的概念）对于 BERT 及其衍生模型尤为重要。
# 这些标记用于更好地表示句子边界，例如句子的开头（[CLS]）或句子之间的分隔符（[SEP]）。
encoded_input = tokenizer("How are you?")
print(encoded_input["input_ids"])
tokenizer.decode(encoded_input["input_ids"])

#[101, 1731, 1132, 1128, 136, 102]
#'[CLS] How are you? [SEP]'

sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
#return_tensors="pt"    tokenizer 已经直接返回 PyTorch Tensor。
#encoded_input=tokenizer(sequences,padding=True,return_tensors="pt")

#Tokenizer 会自动把短句补齐。padding=True
encoded_input=tokenizer(sequences,padding=True)
encoded_sequences=encoded_input["input_ids"]
model_inputs = torch.tensor(encoded_sequences)
print(model_inputs)

#加载使用与 BERT 相同的检查点训练的 BERT 分词器的方法与加载模型的方法相同，使用了BertTokenizer
tokenizer=BertTokenizer.from_pretrained("bert-base-cased")
#与此类似AutoModel，AutoTokenizer该类会根据检查点名称从库中获取正确的标记化器类，并可直接与任何检查点一起使用

#分词过程是通过tokenize()分词器的方法完成的：
tokenizer=AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens=tokenizer.tokenize(sequence)
print(tokens)
#['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
encoded_input=tokenizer(sequences)
#{'input_ids': [[101, 146, 112, 1396, 1151, 2613, 1111, 170, 20164, 10932, 2271,
# 7954, 1736, 1139, 2006, 1297, 119, 102],
# [101, 146, 4819, 1142, 1177, 1277, 106, 102]],
# 'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# [0, 0, 0, 0, 0, 0, 0, 0]],
# 'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
# [1, 1, 1, 1, 1, 1, 1, 1]]}

print(encoded_input)
#输入 ID 的转换由convert_tokens_to_ids()分词器方法处理
#这些输出一旦转换为适当的框架张量，就可以用作模型的输入
ids=tokenizer.convert_tokens_to_ids(tokens)
#[7993, 170, 13809, 23763, 2443, 1110, 3014]
print(ids)
#解码过程则相反：我们希望从词汇索引中获取一个字符串。这可以通过decode()以下方法实现
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)


#多个序列
checkpoint="distilbert-base-uncased-finetuned-sst-2-english"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequence="I've been waiting for a HuggingFace course my whole life."
tokens=tokenizer.tokenize(sequence)
ids=tokenizer.convert_tokens_to_ids(tokens)
input_ids=torch.tensor([ids])
print("input id :",input_ids)
output=model(input_ids)
print("logits:",output.logits)

#input id：[[ 1045 ,   1005 ,   2310 ,   2042 ,   3403 ,   2005 ,   1037 ,
# 17662 , 12172 ,   2607 , 2026 ,   2878 ,   2166 ,   1012 ]]
#logits：[[- 2.7276 ,   2.8789 ]]

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequence1_ids = [[200, 200, 200]]
sequence2_ids = [[200, 200]]
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

print(model(torch.tensor(sequence1_ids)).logits)
print(model(torch.tensor(sequence2_ids)).logits)
print(model(torch.tensor(batched_ids)).logits)
#tensor([[ 1.5694, -1.3895]], grad_fn=<AddmmBackward>)
#tensor([[ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)
#tensor([[ 1.5694, -1.3895],
#       [ 1.3373, -1.2163]], grad_fn=<AddmmBackward>)


# 注意力面具
# 注意力掩码是与输入 ID 张量形状完全相同的张量，填充有 0 和 1：1 表示应该关注相应的标记，
# 0 表示不应该关注相应的标记（即，它们应该被模型的注意力层忽略）

batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],
]

outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
print(outputs.logits)
# tensor([[ 1.5694, -1.3895],
#         [ 0.5803, -0.4125]], grad_fn=<AddmmBackward>)


checkpoint="distilbert-base-uncased-finetuned-sst-2-english"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
sequence="I've been waiting for a HuggingFace course my whole life."
model_inputs=tokenizer(sequence)

sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]
model_inputs = tokenizer(sequences)
# 将序列填充至最大序列长度
model_inputs = tokenizer(sequences, padding="longest")
# 将序列填充至模型最大长度（BERT 或 DistilBERT 为 512）
model_inputs = tokenizer(sequences, padding="max_length")
# 将序列填充至指定的最大长度
model_inputs = tokenizer(sequences, padding="max_length", max_length=8)

sequences = [ "I've been waiting for a HuggingFace course my whole life.", "So have I!" ]
# 将截断长度超过模型最大长度的序列# (BERT或DistilBERT为512)
model_inputs = tokenizer(sequences, truncation= True )
# 将截断长度超过指定最大长度的序列
model_inputs = tokenizer(sequences, max_length= 8 , truncation= True )

# 返回PyTorch张量
model_inputs = tokenizer(sequences, padding= True , return_tensors= "pt" )
# 返回NumPy数组
model_inputs = tokenizer(sequences, padding= True , return_tensors= "np" )


sequence = "I've been waiting for a HuggingFace course my whole life."

model_inputs = tokenizer(sequence)
print(model_inputs["input_ids"])

tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
# [101, 1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012, 102]
# [1045, 1005, 2310, 2042, 3403, 2005, 1037, 17662, 12172, 2607, 2026, 2878, 2166, 1012]

#反编译解密
print(tokenizer.decode(model_inputs["input_ids"]))
print(tokenizer.decode(ids))
# "[CLS] i've been waiting for a huggingface course my whole life. [SEP]"
# "i've been waiting for a huggingface course my whole life."


# 从分词器到模型
# 现在我们已经了解了该对象应用于文本时使用的所有步骤tokenizer，
# 让我们最后再看一下它如何使用其主要 API 处理多个序列（填充！）、
# 非常长的序列（截断！）以及多种类型的张量

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
model=AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [ "I've been waiting for a HuggingFace course my whole life.", "So have I!" ]
tokens=tokenizer(sequences,padding=True,truncation=True,return_tensors="pt")
output=model(**tokens)
print(output)


classifier=pipeline("sentiment-analysis")
result = classifier([
     "I have been waiting for the Hugging Face course my whole life.",
    "I hate this!"
])
print(result)

checkpoint='distilbert-base-uncased-finetuned-sst-2-english'
tokenizer=AutoTokenizer.from_pretrained(checkpoint)
print(tokenizer)

raw_inputs=[
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
#指定要返回的张量类型（PyTorch 或普通 NumPy），我们使用以下return_tensors参数
##{'input_ids': tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
         # 2607,  2026,  2878,  2166,  1012,   102],
       # [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,
     #        0,     0,     0,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     #   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
      #  [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])}#
inputs=tokenizer(raw_inputs,padding=True,truncation=True,return_tensors='pt')
print(inputs)

#Transformers 提供了一个AutoModel类，其中也包含一个from_pretrained()方法
model = AutoModel.from_pretrained(checkpoint)
outputs=model(**inputs)
print (outputs.last_hidden_state.shape)
#torch.Size([2, 16, 768])
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
#输出的形状，就会发现维度要低得多：模型头部以我们之前看到的高维向量作为输入，并输出包含两个值（每个标签一个值）的向量
print (outputs.logits.shape)
#torch.Size([2, 2])
print(outputs.logits)

#这些不是概率，而是logits，即模型最后一层输出的原始、未经归一化的分数。要将其转换为概率，
# 需要经过一个SoftMax层（所有Transformer模型都输出logits，
# 因为训练损失函数通常会将最后一个激活函数（例如SoftMax）与实际损失函数（例如交叉熵）融合）：
predictions = torch.nn.functional.softmax(outputs.logits, dim= -1 )
print (predictions)
model=AutoModel.from_pretrained("bert-base-cased")
#两个模型save_pretrained()保存模型权重和架构配置的方法完全相同
model.save_pretrained("directory_on_my_computer")
print(model)
model = BertModel.from_pretrained( "bert-base-cased" )
print(model)
#两个模型save_pretrained()保存模型权重和架构配置的方法完全相同
#这将在您的磁盘上保存两个文件：
#ls directory_on_my_computer
#config . json model.safetensors
model.save_pretrained("directory_on_my_computer")

#要重用已保存的模型，用from_pretrained()
model = AutoModel.from_pretrained( "directory_on_my_computer" )
