from transformers import AutoTokenizer
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather?"},
]

tok=AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
text=tok.apply_chat_template(messages,tokenize=False)
enc = tok.apply_chat_template(messages, tokenize=True)

simple = [{"role": "system", "content": "You are helpful."},
          {"role": "user", "content": "Hi"}]
for name in ["HuggingFaceTB/SmolLM2-135M-Instruct", "Qwen/Qwen2.5-0.5B-Instruct"]:
    t = AutoTokenizer.from_pretrained(name)
    formatted = t.apply_chat_template(simple, tokenize=False)

infer_msgs = [{"role": "system", "content": "You are helpful."},
              {"role": "user", "content": "Name a color."}]
without = tok.apply_chat_template(infer_msgs, tokenize=False)
withgen = tok.apply_chat_template(infer_msgs, tokenize=False, add_generation_prompt=True)

