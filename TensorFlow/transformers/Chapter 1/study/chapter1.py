import sys
import re
import torch
from transformers import BertTokenizer,BertForSequenceClassification
from torch.nn.functional import softmax

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device="mps"
else:
    device='cpu'

pretrained = "sanshizhang/Chinese-Sentiment-Analysis-Fund-Direction"
model=BertForSequenceClassification.from_pretrained(pretrained)
tokenizer=BertTokenizer.from_pretrained(pretrained)
model=model.to(device)
model.eval()

def predict_sentiment(text:str):
    encoding=tokenizer(
        text,
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids= encoding['input_ids'].to(device)
    attention_mask=encoding['attention_mask'].to(device)
    with torch.no_grad():
        outputs=model(input_ids=input_ids,attention_mask=attention_mask)
        probs=softmax(outputs.logits,dim=1)
    probs_ids=torch.argmax(probs,dim=1).cpu().numpy()[0]
    return probs,probs_ids
def clean_text(text: str) -> str:
    return re.sub(
        r"[^一-鿿\d.a-zA-Z%+\-。！？，、；：（）【】《》“”‘’]",
        "",
        text,
    )
SENTIMENT_LABELS = {0: "negative", 1: "positive", 2: "neutral"}
def analyze(text:str)->None:
    text=clean_text(text)
    probs, pred_idx=predict_sentiment(text)
    label=SENTIMENT_LABELS[pred_idx]
    confidence=probs[0][pred_idx].item()
    print(f"文本: {text}")
    print(f"情感: {label}，置信度: {confidence:.4f}")

if __name__ == "__main__":
    args = sys.argv[1:]  # 忽略脚本名，取命令行后面的词
    if args:
        analyze(" ".join(args))
    else:
        for s in [
            "这家公司业绩大涨，前景一片光明！",
            "基金又跌了，亏麻了，再也不买了。",
            "今天大盘震荡，走势不明朗。",
        ]:analyze(s)


