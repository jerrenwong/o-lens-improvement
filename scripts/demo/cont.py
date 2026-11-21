import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False); torch.set_num_threads(48)
N = "Qwen/Qwen3.5-9B"; tok = AutoTokenizer.from_pretrained(N)
hf = AutoModelForCausalLM.from_pretrained(N, dtype=torch.float32, attn_implementation="sdpa").eval()
T = "I am so thristy, the sun is scorching, there seems to be nothing around me, should I bite through these thorns for water?"
S0 = ("Passage: She pulled on her goggles, tightened her cap and dove into the lane while the coach started the clock\nUnstated word: swimming\n\n"
      "Passage: The physicist who wrote that energy equals mass times the speed of light squared worked as a patent clerk when\nUnstated word: Einstein\n\n"
      "Passage: El médico le puso el termómetro bajo la lengua y frunció el ceño al ver\nUnstated word: fever\n\n"
      "Passage: The engine coughed twice, the needle sat on empty, and the car rolled to a\nUnstated word: fuel\n\nPassage: ")
for name, p in [("forced ' c'", S0 + T + "\nUnstated word: c"),
                ("plain question", "Someone says: \"" + T + "\"\nWhat plant are they about to bite? Answer in one word:")]:
    ids = tok(p, add_special_tokens=False).input_ids
    lg = hf(torch.tensor([ids])).logits[0, -1].float().softmax(-1)
    top = lg.topk(5)
    print(name, [(tok.decode([i]), round(float(q), 3)) for q, i in zip(top.values, top.indices.tolist())], flush=True)
ids = tok(S0 + T + "\nUnstated word:", add_special_tokens=False).input_ids
lg = hf(torch.tensor([ids])).logits[0, -1].float().softmax(-1); top = lg.topk(8)
print("unstated-word question, raw top-8", [(tok.decode([i]), round(float(q), 3)) for q, i in zip(top.values, top.indices.tolist())])
print("CONT_DONE")
