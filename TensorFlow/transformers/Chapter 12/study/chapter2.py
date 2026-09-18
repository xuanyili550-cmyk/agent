import argparse
import re
import torch

def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

SYSTEM = "Answer the question concisely and correctly."
QA = [
    ("What is 2 plus 3? Answer with the number.", "5"),
    ("What is the capital of France?", "Paris"),
    ("What color do you get by mixing blue and yellow?", "green"),
    ("How many legs does a spider have?", "8"),
]
PROMPTS = [q for q, _ in QA]
ANSWERS = [a for _, a in QA]

def reward_fn(texts,answers):
    return torch.tensor([1.0 if a.lower() in t.lower() else 0.0 for t,a in zip(texts,answers)],dtype=torch.float32)

def seq_logprob(model,input_ids,attn_mask,comp_mask):
    logits=model(input_ids=input_ids, attention_mask=attn_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    logp = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    mask = comp_mask[:, 1:].float()  # 只统计补全 token(不含提示/padding)
    return (logp * mask).sum(dim=1)  # (N,) 每条序列的补全总 logp
def main(cfg):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    device = pick_device()
    name = "HuggingFaceTB/SmolLM2-135M-Instruct"
    tok = AutoTokenizer.from_pretrained(name)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(name)
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM", lora_dropout=0.0)).to(device)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)

    def build_prompt(q):
        return tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)

    reward_history = []
    for step in range(cfg.steps):
        model.eval()
        all_seqs, all_comp_masks, all_rewards, all_texts = [], [], [], []
        # ---- ①②③ 采样一组补全 + 打分 + 组内优势 ----
        advantages_list = []
        for q, a in QA:
            enc = tok(build_prompt(q), return_tensors="pt").to(device)
            plen = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=cfg.max_new, do_sample=True,
                                     temperature=1.0, top_k=50, num_return_sequences=cfg.group,
                                     pad_token_id=tok.pad_token_id)
            texts = [tok.decode(g[plen:], skip_special_tokens=True) for g in gen]
            r = reward_fn(texts, [a] * cfg.group)  # (G,) 用该题正确答案打分
            adv = (r - r.mean()) / (r.std() + 1e-8)  # 组内优势
            for g in gen:
                seq = g
                cmask = torch.zeros_like(seq);
                cmask[plen:] = 1  # 标出补全 token
                all_seqs.append(seq);
                all_comp_masks.append(cmask)
            advantages_list.append(adv)
            all_texts += texts
            all_rewards.append(r)
            advantages = torch.cat(advantages_list).to(device)  # (P*G,)
            avg_reward = torch.cat(all_rewards).mean().item()
            reward_history.append(avg_reward)

            # 把所有序列右侧 pad 到同长，堆成一个 batch
            maxlen = max(s.shape[0] for s in all_seqs)
            input_ids = torch.full((len(all_seqs), maxlen), tok.pad_token_id, device=device)
            attn = torch.zeros((len(all_seqs), maxlen), device=device, dtype=torch.long)
            comp = torch.zeros((len(all_seqs), maxlen), device=device, dtype=torch.long)
            for i, (s, c) in enumerate(zip(all_seqs, all_comp_masks)):
                input_ids[i, :s.shape[0]] = s
                attn[i, :s.shape[0]] = 1
                comp[i, :c.shape[0]] = c
                # ---- ④⑤⑥ 策略 logp(带梯度) + 参考 logp(关 LoRA) + loss + 更新 ----
                model.train()
                policy_logp = seq_logprob(model, input_ids, attn, comp)  # 带梯度
                with torch.no_grad():
                    with model.disable_adapter():  # 参考=基座(无 LoRA)
                        ref_logp = seq_logprob(model, input_ids, attn, comp)
                # 策略梯度(组内优势) + KL 惩罚(别跑离基座太远)
                pg_loss = -(advantages * policy_logp).mean()
                kl = (ref_logp - policy_logp).mean()  # 近似 KL(policy||ref)
                loss = pg_loss + cfg.beta * kl
                optimizer.zero_grad();
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()

                if step % 5 == 0 or step == cfg.steps - 1:
                    print(f"  step {step:2}: 答对率={avg_reward * 100:5.1f}%  loss={loss.item():.3f}")

            early = sum(reward_history[:3]) / 3
            late = sum(reward_history[-3:]) / 3
            print(f"\n>>> 答对率：前3步={early * 100:.1f}% → 后3步={late * 100:.1f}%  "
                  f"({'↑ 上升，GRPO 在起作用' if late > early + 0.02 else '波动(RL 噪声大，需更多步/更大模型)'})")
            print(">>> 手写 GRPO 跑通：采样组→打分→组内优势→裁剪/KL→更新 LoRA。生产上云端用 trl.GRPOTrainer。")




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=15)
    p.add_argument("--group", type=int, default=4, help="每个提示采样几个补全(组大小 G)")
    p.add_argument("--max_new", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=0.01)
    args = p.parse_args()
    main(args)