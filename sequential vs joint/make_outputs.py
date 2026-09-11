#!/usr/bin/env python3
"""Build publication-oriented tables, figures, and a provenance manifest."""

import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
R, T, F, A = ROOT/"results", ROOT/"tables", ROOT/"figures", ROOT/"artifacts"
T.mkdir(exist_ok=True); F.mkdir(exist_ok=True)
TARGETS = ("redshift", "stellar_mass", "metallicity", "age", "ssfr")


def load(name): return json.loads((R/name).read_text())
def write_csv(name, rows):
    with (T/name).open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=rows[0]); w.writeheader(); w.writerows(rows)


probe_rows=[]
labels={"S_I":("Sequential","Image"),"S_S":("Sequential","Spectrum"),
        "J_I":("Joint","Image"),"J_S":("Joint","Spectrum")}
for space,(regime,modality) in labels.items():
    d=load(f"probes_{space}.json")["representations"]
    for kind,v in d.items():
        scores={"redshift":v["redshift"]["summary"]["r2"]["mean"]}
        scores.update({k:x["summary"]["r2"]["mean"] for k,x in v["properties"].items()})
        probe_rows.append({"regime":regime,"modality":modality,"representation":kind,
                           **{k:f"{scores[k]:.6f}" for k in TARGETS}})
write_csv("downstream_probes.csv",probe_rows)

retrieval_rows=[]
for model in ("sequential","joint"):
    d=load(f"retrieval_{model}.json")
    for scope,x in d["scopes"].items():
        for direction,v in x.items():
            retrieval_rows.append({"regime":model,"scope":scope,"direction":direction,
                **{k:f"{v[k]:.8f}" for k in ("recall_at_1","recall_at_5","recall_at_10","mrr")},
                "median_rank":v["median_rank"]})
write_csv("retrieval.csv",retrieval_rows)

retention_rows=[]
for name,v in load("retention.json")["comparisons"].items():
    retention_rows.append({"comparison":name,"left":v["states"][0],"right":v["states"][1],
        "linear_cka":f"{v['linear_cka_test']:.6f}",
        "pairwise_cosine_spearman":f"{v['pairwise_cosine_spearman']:.6f}",
        **{f"neighbor_overlap_at_{k}":f"{v['neighbor_overlap'][str(k)]:.6f}" for k in (10,50,100)}})
write_csv("retention.csv",retention_rows)

decomp_rows=[]
for regime in ("independent","sequential","joint"):
    d=load(f"decomposition_{regime}.json")
    for direction,v in d["directions"].items():
        for component,targets in v["probes"].items():
            row={"regime":regime,"direction":direction,"component":component,
                 "cross_modal_global_r2":f"{v['test_global_r2']:.6f}"}
            row.update({t:f"{targets[t]['summary']['r2']['mean']:.6f}" for t in TARGETS})
            decomp_rows.append(row)
write_csv("shared_private.csv",decomp_rows)

rank_rows=[]
for space,(regime,modality) in labels.items():
    d=load(f"rank_geometry_{space}.json")["representations"]
    for kind,v in d.items():
        rank_rows.append({"regime":regime,"modality":modality,"representation":kind,
                          "feature_dim":v["feature_dim"],
                          "effective_rank":f"{v['effective_rank']:.6f}",
                          "participation_ratio":f"{v['participation_ratio']:.6f}",
                          "largest_eigenvalue_share":f"{v['largest_eigenvalue_share']:.6f}"})
write_csv("rank_geometry.csv",rank_rows)

morph_rows=[]
for state in ("U_I","S_I","J_I_raw","J_I"):
    d=load(f"galaxy10_{state}.json")["summary"]["test"]
    morph_rows.append({"state":state,
        "accuracy_mean":f"{d['accuracy']['mean']:.6f}",
        "accuracy_std":f"{d['accuracy']['std']:.6f}",
        "macro_f1_mean":f"{d['macro_f1']['mean']:.6f}",
        "macro_f1_std":f"{d['macro_f1']['std']:.6f}"})
write_csv("galaxy10_morphology.csv",morph_rows)
fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
x=np.arange(4)
ax.bar(x,[100*float(r["macro_f1_mean"]) for r in morph_rows],
       yerr=[100*float(r["macro_f1_std"]) for r in morph_rows],
       color=["#4C78A8","#72B7B2","#F58518","#E45756"],capsize=3)
ax.set_xticks(x,["Independent raw","Sequential aligned","Joint raw","Joint aligned"])
ax.set_ylabel("Galaxy10 macro F1 (%)"); ax.set_ylim(0,80)
fig.savefig(F/"05_morphology_retention.png",dpi=180); plt.close(fig)


# Downstream heatmaps
fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
for ax,kind in zip(axes,("raw","projected")):
    rows=[r for r in probe_rows if r["representation"]==kind]
    matrix=np.array([[float(r[t]) for t in TARGETS] for r in rows])
    im=ax.imshow(matrix,vmin=0,vmax=.9,cmap="viridis",aspect="auto")
    ax.set_xticks(range(5),["z","Mass","Metal.","Age","sSFR"])
    ax.set_yticks(range(4),[f"{r['regime']} {r['modality']}" for r in rows])
    ax.set_title(f"{kind.title()} representation")
    for i in range(4):
        for j in range(5): ax.text(j,i,f"{matrix[i,j]:.2f}",ha="center",va="center",color="white" if matrix[i,j]<.45 else "black",fontsize=8)
fig.colorbar(im,ax=axes,label="Linear probe R2")
fig.savefig(F/"01_downstream_information.png",dpi=180); plt.close(fig)

# Retrieval
fig,ax=plt.subplots(figsize=(7,4.5),constrained_layout=True)
rows=[r for r in retrieval_rows if r["scope"]=="test_29697"]
x=np.arange(2); width=.36
for i,model in enumerate(("sequential","joint")):
    vals=[100*float(next(r for r in rows if r["regime"]==model and r["direction"]==d)["recall_at_10"]) for d in ("image_to_spectrum","spectrum_to_image")]
    ax.bar(x+(i-.5)*width,vals,width,label=model.title())
ax.set_xticks(x,["Image to spectrum","Spectrum to image"]); ax.set_ylabel("Recall@10 (%)"); ax.legend(frameon=False)
fig.savefig(F/"02_pair_retrieval.png",dpi=180); plt.close(fig)

# Retention
fig,ax=plt.subplots(figsize=(10,4.8),constrained_layout=True)
names=[r["comparison"].replace("_"," ") for r in retention_rows]
x=np.arange(len(names))
ax.bar(x-.2,[float(r["linear_cka"]) for r in retention_rows],.4,label="Linear CKA")
ax.bar(x+.2,[float(r["neighbor_overlap_at_100"]) for r in retention_rows],.4,label="kNN overlap@100")
ax.set_xticks(x,names,rotation=24,ha="right"); ax.set_ylim(0,.8); ax.legend(frameon=False)
fig.savefig(F/"03_representation_retention.png",dpi=180); plt.close(fig)

# Shared/private target accessibility, averaged over directions and targets
fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
for i,regime in enumerate(("independent","sequential","joint")):
    subset=[r for r in decomp_rows if r["regime"]==regime]
    vals=[]
    for component in ("total","shared","private"):
        q=[r for r in subset if r["component"]==component]
        vals.append(np.mean([[float(r[t]) for t in TARGETS] for r in q]))
    ax.bar(np.arange(3)+(i-1)*.25,vals,.25,label=regime.title())
ax.set_xticks(range(3),["Total","Cross-predictable","Residual"]); ax.set_ylabel("Mean property probe R2"); ax.legend(frameon=False)
fig.savefig(F/"04_shared_private_accessibility.png",dpi=180); plt.close(fig)

manifest={"study":"sequential versus joint cross-modal JEPA","labels_sha256":hashlib.sha256((A/"labels.npz").read_bytes()).hexdigest(),
          "population":{"rows":168280,"train":138583,"test":29697,"property_test":15993},
          "states":{"U":"independent SSL raw backbone","S":"sequential learned-query pooler","J_raw":"joint backbone output","J":"joint projected output"},
          "warning":"Sequential freezes its pretrained backbones and trains about 2.23M pooler parameters for 10 epochs; joint trains about 392M parameters end-to-end for 50 epochs. This is not an initialization-only controlled comparison."}
(A/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print("tables, figures, and manifest written")

