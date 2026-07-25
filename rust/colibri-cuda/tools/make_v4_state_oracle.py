#!/usr/bin/env python3
"""Generate the pinned L0 embedding -> mHC-pre -> attn-norm PyTorch oracle."""

import argparse, hashlib, json, struct, warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import safetensors, torch
from safetensors.torch import safe_open

REVISION = "b5968e9190ef611bbf34a7229255be88a0e937c1"
IDS, HIDDEN, HC, MIX = [0, 1, 2, 129279], 7168, 4, 24
MAGIC = b"CV4S0001"
OUT = Path(__file__).resolve().parent.parent / "fixtures/v4_state_pytorch_oracle.json"

def sha(t):
    return hashlib.sha256(bytes(t.contiguous().view(torch.uint8).reshape(-1).tolist())).hexdigest()

def file_sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def sinkhorn(mixes, scale, base):
    pre = torch.sigmoid(mixes[:, :HC] * scale[0] + base[:HC]) + 1e-6
    post = 2 * torch.sigmoid(mixes[:, HC:2 * HC] * scale[1] + base[HC:2 * HC])
    comb = mixes[:, 2 * HC:].reshape(-1, HC, HC) * scale[2] + base[2 * HC:].reshape(HC, HC)
    stages = {"comb_logits_f32": sha(comb)}; trace=[comb.clone()]
    comb = torch.softmax(comb, dim=-1) + 1e-6
    stages["after_softmax_eps_f32"] = sha(comb); trace.append(comb.clone())
    for step in range(20):
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
        stages[f"col_{step:02d}_f32"] = sha(comb); trace.append(comb.clone())
        if step != 19:
            comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
            stages[f"row_{step:02d}_f32"] = sha(comb); trace.append(comb.clone())
    return pre, post, comb, stages, torch.stack(trace,dim=1)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--model-dir",type=Path,required=True); ap.add_argument("--json",type=Path,default=OUT); ap.add_argument("--bin",type=Path); ap.add_argument("--check",action="store_true"); a=ap.parse_args()
    a.bin = a.bin or a.json.with_suffix(".bin")
    if not a.model_dir.is_dir(): ap.error("missing pinned model directory")
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True); torch.set_float32_matmul_precision("highest")
    one, two = a.model_dir / "model-00001-of-00064.safetensors", a.model_dir / "model-00002-of-00064.safetensors"
    with safe_open(one, framework="pt", device="cpu") as f: emb=f.get_slice("embed.weight")[IDS]
    with safe_open(two, framework="pt", device="cpu") as f:
        fn, base, scale = (f.get_tensor(f"layers.0.hc_attn_{x}") for x in ("fn","base","scale"))
        norm=f.get_tensor("layers.0.attn_norm.weight")
    residual=emb[:,None,:].repeat(1,HC,1)
    flat=residual.flatten(1).float(); mixes=torch.nn.functional.linear(flat,fn.float())*torch.rsqrt(flat.square().mean(-1,keepdim=True)+1e-6)
    pre,post,comb,stages,trace=sinkhorn(mixes,scale.float(),base.float())
    collapsed=(pre[:,:,None]*residual.float()).sum(1).to(torch.bfloat16)
    normalized=(collapsed.float()*torch.rsqrt(collapsed.float().square().mean(-1,keepdim=True)+1e-6)*norm.float()).to(torch.bfloat16)
    parts=[emb.view(torch.uint16),collapsed.view(torch.uint16),normalized.view(torch.uint16),post,comb,mixes,pre,trace]
    binary=MAGIC+b"".join(t.contiguous().view(torch.uint8).reshape(-1).tolist() and bytes(t.contiguous().view(torch.uint8).reshape(-1).tolist()) for t in parts)
    sizes=[t.numel()*t.element_size() for t in parts]; offsets=[]; pos=len(MAGIC)
    for size in sizes: offsets.append(pos); pos+=size
    names=["embedding","collapsed","norm","post","comb","mixes","pre","sinkhorn_trace"]
    dtypes=["bf16-le","bf16-le","bf16-le","f32-le","f32-le","f32-le","f32-le","f32-le"]
    shapes=[[4,HIDDEN],[4,HIDDEN],[4,HIDDEN],[4,HC],[4,HC,HC],[4,MIX],[4,HC],[4,41,HC,HC]]
    binary_meta={name:{"offset":offset,"bytes":size,"dtype":dtype,"shape":shape} for name,offset,size,dtype,shape in zip(names,offsets,sizes,dtypes,shapes)}
    data={"schema":"colibri-v4-state-pytorch-oracle-v2","model_revision":REVISION,"ids":IDS,"shape":{"rows":4,"hidden":HIDDEN,"hc":HC,"mix":MIX,"sinkhorn_iters":20,"sinkhorn_trace_stages":41},"semantics":"embed BF16; repeat hc=4; FP32 F.linear(x,hc_attn_fn)*rsqrt(mean(x^2)+1e-6); official 20-step Sinkhorn; pre collapse then BF16; attn RMSNorm FP32 then BF16","official_source":{"model.py_sha256":file_sha(a.model_dir/'inference/model.py'),"kernel.py_sha256":file_sha(a.model_dir/'inference/kernel.py'),"tilelang_executed":False,"reason":"CPU PyTorch formula oracle"},"environment":{"torch":torch.__version__,"safetensors":safetensors.__version__,"device":"cpu"},"source":{"embed":{"shard":one.name,"dtype":"BF16","shape":[129280,HIDDEN]},"hc":{"shard":two.name,"fn":"F32[24,28672]","base":"F32[24]","scale":"F32[3]"},"attn_norm":{"shard":two.name,"dtype":"BF16","shape":[HIDDEN]}},"hashes":{"embedding_bf16":sha(emb),"residual_bf16":sha(residual),"flat_f32":sha(flat),"mixes_f32":sha(mixes),"pre_f32":sha(pre),"post_f32":sha(post),"comb_f32":sha(comb),"collapsed_bf16":sha(collapsed),"norm_bf16":sha(normalized),"sinkhorn_trace_f32":sha(trace)},"sinkhorn_stage_hashes":stages,"binary":{"magic":"CV4S0001","sha256":hashlib.sha256(binary).hexdigest()}|binary_meta}
    rendered=json.dumps(data,indent=2,sort_keys=True)+"\n"
    if a.check:
        if not a.json.is_file() or not a.bin.is_file() or a.json.read_text()!=rendered or a.bin.read_bytes()!=binary: raise SystemExit("fixture differs; regenerate without --check")
        print("PASS: state fixture reproduces exactly")
    else:
        a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(rendered); a.bin.write_bytes(binary); print(f"wrote {a.json} and {a.bin}")
if __name__ == '__main__': main()
