# -*- coding: utf-8 -*-
"""Árvore completa: equivalente determinístico (PL único) e PDDD por cortes de Benders."""
from pyscipopt import Model, quicksum, SCIP_PARAMSETTING

PARAMETROS_PADRAO = {
    "UHE": {"Vmin": 5754.0, "Vmax": 8795.0, "VI": 8000.0,
            "Prod": 0.235, "Engol": 1628.0, "CustoVert": 0.01},
    "UTE": {
        "Termobahia": {"Capac": 186.0, "Custo": 424080.0},
        "PecemII": {"Capac": 144.0, "Custo": 1764000.0},
    },
    "Carga": {1: 530.0, 2: 508.0, 3: 516.0},
    "CDef": 5969520.0,
    "Afluencias": {"M1": 2156.11, "M2-O": 1838.59, "M2-P": 713.66,
                    "M3-O": 1674.00, "M3-P": 656.21},
    "Probabilidades": {"M2-O": 0.50, "M2-P": 0.50,
                       "M3-O": 0.50, "M3-P": 0.50},
}


def _arvore(p):
    a, pr = p["Afluencias"], p["Probabilidades"]
    return {
        "M1": {"estagio": 1, "pai": None, "afl": a["M1"], "prob": 1.0},
        "M2-O": {"estagio": 2, "pai": "M1", "afl": a["M2-O"], "prob": pr["M2-O"]},
        "M2-P": {"estagio": 2, "pai": "M1", "afl": a["M2-P"], "prob": pr["M2-P"]},
        "M3-OO": {"estagio": 3, "pai": "M2-O", "afl": a["M3-O"], "prob": pr["M2-O"]*pr["M3-O"]},
        "M3-OP": {"estagio": 3, "pai": "M2-O", "afl": a["M3-P"], "prob": pr["M2-O"]*pr["M3-P"]},
        "M3-PO": {"estagio": 3, "pai": "M2-P", "afl": a["M3-O"], "prob": pr["M2-P"]*pr["M3-O"]},
        "M3-PP": {"estagio": 3, "pai": "M2-P", "afl": a["M3-P"], "prob": pr["M2-P"]*pr["M3-P"]},
    }


def _linha(no, d, p, metodo):
    e = d["estagio"]
    return {"Metodo": metodo, "No": no, "Estagio": f"M{e}", "Pai": d["pai"] or "-",
            "Probabilidade": d["prob"], "Afluencia": d["afl"], "VolumeFinal": d["vf"],
            "VolumeTurbinado": d["vt"], "VolumeVertido": d["vv"],
            "GeracaoHidraulica": p["UHE"]["Prod"]*d["vt"],
            "Termobahia": d["gt"]["Termobahia"], "PecemII": d["gt"]["PecemII"],
            "Deficit": d["def"], "Carga": p["Carga"][e], "CustoImediato": d["ci"]}


def resolver_pl_unico(p):
    """Equivalente determinístico: um único PL, uma decisão em cada nó da árvore."""
    arv, uhe, utes = _arvore(p), p["UHE"], p["UTE"]
    m = Model("PL_UNICO_ARVORE_COMPLETA"); m.hideOutput()
    vf={}; vt={}; vv={}; de={}; gt={}
    for no,d in arv.items():
        vf[no]=m.addVar(f"vf_{no}",lb=uhe["Vmin"],ub=uhe["Vmax"])
        vt[no]=m.addVar(f"vt_{no}",lb=0,ub=uhe["Engol"])
        vv[no]=m.addVar(f"vv_{no}",lb=0)
        de[no]=m.addVar(f"def_{no}",lb=0)
        for t,x in utes.items(): gt[t,no]=m.addVar(f"gt_{t}_{no}",lb=0,ub=x["Capac"])
    obj=0
    for no,d in arv.items():
        ci=quicksum(utes[t]["Custo"]*gt[t,no] for t in utes)+p["CDef"]*de[no]+uhe["CustoVert"]*vv[no]
        obj += d["prob"]*ci
        entrada=uhe["VI"] if d["pai"] is None else vf[d["pai"]]
        m.addCons(vf[no]+vt[no]+vv[no]==entrada+d["afl"],name=f"BH_{no}")
        m.addCons(uhe["Prod"]*vt[no]+quicksum(gt[t,no] for t in utes)+de[no]==p["Carga"][d["estagio"]],name=f"AD_{no}")
    m.setObjective(obj,"minimize"); m.optimize()
    if str(m.getStatus()).lower()!="optimal": raise RuntimeError(f"Status SCIP: {m.getStatus()}")
    res=[]
    for no,d in arv.items():
        vals={**d,"vf":m.getVal(vf[no]),"vt":m.getVal(vt[no]),"vv":m.getVal(vv[no]),
              "def":m.getVal(de[no]),"gt":{t:m.getVal(gt[t,no]) for t in utes}}
        vals["ci"]=sum(utes[t]["Custo"]*vals["gt"][t] for t in utes)+p["CDef"]*vals["def"]+uhe["CustoVert"]*vals["vv"]
        res.append(_linha(no,vals,p,"PL Único"))
    return {"CustoTotal":m.getObjVal(),"Resultados":res,"ProbabilidadesUsadas":p["Probabilidades"].copy()}


def _subproblema(p, no, entrada, cortes=None):
    """Resolve um nó. Os cortes aproximam o custo futuro em função do volume final."""
    arv, uhe, utes = _arvore(p), p["UHE"], p["UTE"]
    d=arv[no]; m=Model(f"SP_{no}"); m.hideOutput(); m.setPresolve(SCIP_PARAMSETTING.OFF)
    vf=m.addVar("vf",lb=uhe["Vmin"],ub=uhe["Vmax"]); vt=m.addVar("vt",lb=0,ub=uhe["Engol"])
    vv=m.addVar("vv",lb=0); de=m.addVar("def",lb=0)
    gt={t:m.addVar(f"gt_{t}",lb=0,ub=x["Capac"]) for t,x in utes.items()}
    alpha=m.addVar("alpha",lb=0) if cortes is not None else None
    bh=m.addCons(vf+vt+vv==float(entrada)+d["afl"],name="BH")
    m.addCons(uhe["Prod"]*vt+quicksum(gt[t] for t in utes)+de==p["Carga"][d["estagio"]],name="AD")
    if cortes is not None:
        for k,(a,b) in enumerate(cortes): m.addCons(alpha>=a+b*vf,name=f"corte_{k}")
    ci=quicksum(utes[t]["Custo"]*gt[t] for t in utes)+p["CDef"]*de+uhe["CustoVert"]*vv
    m.setObjective(ci+(alpha if alpha is not None else 0),"minimize"); m.optimize()
    if str(m.getStatus()).lower()!="optimal": raise RuntimeError(f"Status SCIP em {no}: {m.getStatus()}")
    out={**d,"vf":m.getVal(vf),"vt":m.getVal(vt),"vv":m.getVal(vv),"def":m.getVal(de),
         "gt":{t:m.getVal(gt[t]) for t in utes},"valor":m.getObjVal(),"dual_entrada":m.getDualsolLinear(bh)}
    out["ci"]=sum(utes[t]["Custo"]*out["gt"][t] for t in utes)+p["CDef"]*out["def"]+uhe["CustoVert"]*out["vv"]
    return out


def _corte_esperado(p, filhos, x, cortes_filhos=None):
    arv, pr=p and _arvore(p), p["Probabilidades"]
    vals=[]
    for f in filhos:
        v=_subproblema(p,f,x,cortes_filhos.get(f,[]) if cortes_filhos is not None else None)
        cond = pr["M3-O"] if f.endswith("O") else pr["M3-P"]
        vals.append((cond,v))
    q=sum(w*v["valor"] for w,v in vals); b=sum(w*v["dual_entrada"] for w,v in vals)
    return (q-b*x,b), vals


def resolver_pddd(p, tolerancia=1e-3, max_iter=100):
    """PDDD recursiva para a árvore completa, com cortes de Benders e limites ZINF/ZSUp."""
    pr=p["Probabilidades"]
    cortes_m2={"M2-O":[],"M2-P":[]}; cortes_m1=[]; hist=[]
    melhor=None
    for it in range(1,max_iter+1):
        s1=_subproblema(p,"M1",p["UHE"]["VI"],cortes_m1)
        zinf=s1["valor"]
        sol2={}; sol3={}; custo_fut2=0.0
        for n2,w2 in (("M2-O",pr["M2-O"]),("M2-P",pr["M2-P"])):
            s2=_subproblema(p,n2,s1["vf"],cortes_m2[n2]); sol2[n2]=s2
            filhos=("M3-OO","M3-OP") if n2=="M2-O" else ("M3-PO","M3-PP")
            corte3, vals3=_corte_esperado(p,filhos,s2["vf"])
            cortes_m2[n2].append(corte3)
            for f,(w3,v3) in zip(filhos,vals3): sol3[f] = v3
            custo_fut2 += w2*(s2["ci"]+sum(w3*v3["ci"] for w3,v3 in vals3))
        zsup=s1["ci"]+custo_fut2
        # Recalcula os nós M2 com os novos cortes e cria o corte esperado para M1.
        qs=[]
        for n2,w2 in (("M2-O",pr["M2-O"]),("M2-P",pr["M2-P"])):
            v=_subproblema(p,n2,s1["vf"],cortes_m2[n2]); qs.append((w2,v))
        q=sum(w*v["valor"] for w,v in qs); b=sum(w*v["dual_entrada"] for w,v in qs)
        cortes_m1.append((q-b*s1["vf"],b))
        gap=max(0.0,zsup-zinf); hist.append({"Iteracao":it,"ZINF":zinf,"ZSUP":zsup,"Gap":gap})
        melhor=(s1,sol2,sol3,zsup)
        if gap <= tolerancia*max(1.0,abs(zsup)): break
    # Uma passada final com todos os cortes fornece a política convergida.
    s1=_subproblema(p,"M1",p["UHE"]["VI"],cortes_m1); nos={"M1":s1}; custo=s1["ci"]
    for n2,w2 in (("M2-O",pr["M2-O"]),("M2-P",pr["M2-P"])):
        s2=_subproblema(p,n2,s1["vf"],cortes_m2[n2]); nos[n2]=s2; custo += w2*s2["ci"]
        filhos=("M3-OO","M3-OP") if n2=="M2-O" else ("M3-PO","M3-PP")
        for f in filhos:
            s3=_subproblema(p,f,s2["vf"]); nos[f]=s3
            w3=pr["M3-O"] if f.endswith("O") else pr["M3-P"]
            custo += w2*w3*s3["ci"]
    res=[_linha(no,nos[no],p,"PDDD") for no in _arvore(p)]
    return {"CustoTotal":custo,"Resultados":res,"Convergencia":hist,"Iteracoes":len(hist),
            "ZINF":hist[-1]["ZINF"],"ZSUP":hist[-1]["ZSUP"],"ProbabilidadesUsadas":pr.copy()}
