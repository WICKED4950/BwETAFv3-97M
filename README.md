# 🧾 BwETAFv3-97M: Model Card


**Boring’s Experimental Transformer for Autoregression (Flax)**
A 97M parameter autoregressive language model built using a custom Flax pipeline, loads of tuning, and a sprinkle of existential dread.

> *Trained on determination, fueled by suffering, powered by free TPUs.*

---
## 📌 Model Summary

* **Model Name:** BwETAFv3-97M
* **Parameters:** 97,609,728
* **Training Tokens:** 3,390,046,208
* **Training Time:** 4.82 TPUv3-8 Hours
* **Framework:** Flax / JAX
* **Max Context Length:** 2048 tokens
* **Tokenizer:** Custom-trained BPE tokenizer (`vocab_size = 16,384`)
* **Dataset Used:** Fineweb (For training and validation) and TinyStories (For training only)

---
## 🧪 Hyperparameters

```json
{
  "vocab_size": 16384,
  "max_len": 2048,
  "dtype": "bfloat16",
  "num_heads": 8,
  "attention_dim": 768,
  "attn_chunks": 1,
  "gqa_repeats": 2,
  "use_flash_attention": false,
  "num_blocks": 12,
  "ff_dim": 2304,
  "dropout_rate": 0.05,
  "emb_init_range": 0.02,
}

```

---
## 🛠 Optimizer Settings

```json
{
  "peaklr":1.5e-3,
  "warmup_percent":0.02,
  "min_value":1.8e-4,
  "training_decay":"cosine",
  "weight_decay": 0.1,
  "min_warmup_value":6e-4,
  "b1": 0.95,
  "b2": 0.98,
  "eps": 1.5e-8,
  "opt_dtype": "bfloat16"
}
```
---
## 📈 Performance

* **Final Validation Loss:** `3.02`

* **Training-Validation loss Graphs:**
![image/png](https://cdn-uploads.huggingface.co/production/uploads/661e235e08dd378c818654ad/cKhmf_vy56WVjh3L-GklJ.png)

* For detailed stats, refer to `stats.json` in the model files.

---
## ⚡ Quickstart


```bash
!pip install BwETAF==0.6
# If you are having any troubles with compatablility issues with the model like jax, hf and BwETAF run
!pip install --upgrade jax jaxlib flax tiktoken jax_cuda12_plugin datasets flash-attention-jax
```

```python
from BwETAF.api.predictv2 import KV_caching
import BwETAF
import jax
from BwETAF.tokenizer.main import load

model = BwETAF.load_hf("WICKED4950/BwETAFv3-97M",jax.numpy.bfloat16)
tokenizer = load("Loaded_model")

thing = KV_caching(model, top_p=0.92,temperature=0.8)
prompt = """The night sky was alive with lightning, each flash revealing the jagged cliffs ahead. I gripped the letter tighter, knowing it held the answer to everything. The wind screamed as I took a step closer to the edge"""
input = tokenizer.encode(prompt)
print("The input is",input)
the_thing = thing(jax.numpy.array(input),max_len=128)
print(prompt,end="")
for i in the_thing:
    print(tokenizer.decode(i), end="")
print()


# To get the params or jax struct
params = model.trainable_variables
structure = model.model_struct
```

> ☁️ *Colab support and examples coming soon!*

---
## 🧭 Intended Use

This model is primarily released for research and experimentation,  
but it can also be fine-tuned and adapted for downstream tasks.  
Users are encouraged to evaluate it carefully for their specific use cases.  

---
## 📚 Paper

This model is described in detail in the accompanying paper:  
📄 [BwETAFv3: Efficient Autoregressive Language Modeling with 97M Parameters](BwETAFv3.pdf)

The paper covers:  
- Architecture overview  
- Training methodology  
- Regularization & stability techniques  
- Scaling considerations  
- Evaluation results  
- Contact info  

---
## 📬 Contact Me


* 📸 Instagram: [boring.\_.wicked](https://www.instagram.com/boring._.wicked/)
* 💬 Discord: `fused_computation.1` *(if you spot me lurking in any AI-related servers)*

---
