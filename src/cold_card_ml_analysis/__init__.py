"""
Detecção de transações do ataque (ColdCard) com scikit-learn.

Fluxo (segue a seção 2.3 do enunciado):
  1. Lê todos os blocos e monta um dataset: positivo = transação que paga ao
     endereço do roubo (TARGET); negativo = todas as outras.
     NENHUMA feature usa endereços (só tipos de endereço, valores, formato,
     assinatura...), senão o modelo aprenderia o atalho "paga ao TARGET".
  2. Compara vários classificadores (lista do exemplo do professor) com os
     métodos do capítulo 3 do scikit-learn (cross_validate, StratifiedGroupKFold,
     várias métricas, Dummy como piso).
  3. Pontua TODAS as transações fora da amostra de treino (out-of-fold, por
     bloco) e mostra as candidatas a OUTROS ataques, agrupadas por destino.
  4. Opcional (--extras): validation_curve, learning_curve, GridSearchCV,
     TunedThresholdClassifierCV e permutation_test_score na árvore de decisão.

Uso:
    python cold_card_ml.py blocos/
    python cold_card_ml.py blocos/ --extras --neg-sample 100000

Saídas (pasta --outdir, padrão "saida_ml/"):
    resultados_modelos.csv, candidatas.csv, destinos_candidatos.csv,
    arvore.txt, pr_curve.png e, com --extras, validation_curve.png e
    learning_curve.png.
"""

import argparse
import json
import pickle
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só salva PNG (funciona sem servidor gráfico no WSL)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_validate,
    learning_curve,
    permutation_test_score,
    validation_curve,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    import orjson

    loads = orjson.loads
except ImportError:
    loads = json.loads

TARGET = "bc1qnk4zh9qcnap2mycp56qjrgza3cc8ylrh8fecp0"
SAT = 100_000_000
KINDS = ("p2wpkh", "p2wsh", "p2tr", "p2sh", "p2pkh", "none")
META_COLS = {"tx_hash", "block", "time", "label", "outs", "ins"}


# =====================================================================
# 1) CARGA E FEATURES (rodam nos workers, uma linha por transação)
# =====================================================================


def read_witness(hexstr):
    """Divide o witness serializado em itens (bytes). [] se malformado."""
    try:
        b = bytes.fromhex(hexstr)
        i = 0

        def varint():
            nonlocal i
            v = b[i]
            i += 1
            if v == 0xFD:
                v = int.from_bytes(b[i : i + 2], "little")
                i += 2
            return v

        items = []
        for _ in range(varint()):
            n = varint()
            items.append(b[i : i + n])
            i += n
        return items
    except ValueError, IndexError:
        return []


def addr_kind(a):
    if not a:
        return "none"
    if a.startswith("bc1p"):
        return "p2tr"
    if a.startswith("bc1q"):
        return "p2wpkh" if len(a) == 42 else "p2wsh"
    if a.startswith("3"):
        return "p2sh"
    if a.startswith("1"):
        return "p2pkh"
    return "none"


def tx_record(tx, target, block_name):
    inputs = tx.get("inputs", [])
    outs = tx.get("out", [])
    if not inputs or (len(inputs) == 1 and "prev_out" not in inputs[0]):
        return None  # coinbase

    prevs = [i.get("prev_out") for i in inputs]
    in_addr_list = [(p or {}).get("addr") for p in prevs]
    in_addrs = {a for a in in_addr_list if a}
    if target in in_addrs:
        return None  # o próprio TARGET gastando: não é saque nem transação normal
    out_addr_list = [o.get("addr") for o in outs]
    label = int(target in out_addr_list)

    # ---- assinatura / witness ----
    sig_lens, sighashes, low_r, wit_items = [], [], [], []
    multisig = taproot = 0
    for i in inputs:
        w = i.get("witness")
        if not w:
            continue
        items = read_witness(w)
        wit_items.append(len(items))
        if len(items) == 1 and len(items[0]) in (64, 65):
            taproot += 1
        if len(items) >= 2 and len(items[-1]) > 33 and items[-1][-1] == 0xAE:
            multisig += 1  # witnessScript termina em OP_CHECKMULTISIG
        for it in items:
            if len(it) >= 9 and it[0] == 0x30 and it[2] == 0x02:  # DER + sighash
                sig_lens.append(len(it))
                sighashes.append(it[-1])
                low_r.append(it[3] <= 32)  # low-R: r sem byte de padding 0x00

    # ---- valores / formato ----
    seqs = [i.get("sequence", 0xFFFFFFFF) for i in inputs]
    in_vals = [p["value"] for p in prevs if p]
    vals = [o["value"] for o in outs]
    weight = tx.get("weight") or tx.get("size", 0) * 4
    vsize = (weight / 4) or 1.0
    in_sum, out_sum = sum(in_vals), sum(vals)
    fee = tx.get("fee")
    if fee is None:
        fee = max(in_sum - out_sum, 0)
    h = tx.get("block_height") or 0
    lt = tx.get("lock_time", 0)
    in_k = Counter(addr_kind(a) for a in in_addr_list)
    out_k = Counter(addr_kind(a) for a in out_addr_list)

    feats = {
        "ver": tx.get("ver", 0),
        "vin_sz": tx.get("vin_sz", len(inputs)),
        "vout_sz": tx.get("vout_sz", len(outs)),
        "log_weight": np.log1p(weight),
        "log_fee_rate": np.log1p(fee / vsize),
        "locktime_zero": int(lt == 0),
        "locktime_near_height": int(0 < lt < 500_000_000 and 0 <= h - lt <= 100),
        # sequence: indicadores (valores de 32 bits perdem precisão em float32)
        "seq_fffffffd": int(all(s == 0xFFFFFFFD for s in seqs)),
        "seq_fffffffe": int(all(s == 0xFFFFFFFE for s in seqs)),
        "seq_ffffffff": int(all(s == 0xFFFFFFFF for s in seqs)),
        "seq_mixed": int(len(set(seqs)) > 1),
        "seq_other": int(
            any(s not in (0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF) for s in seqs)
        ),
        "n_sigs": len(sig_lens),
        "sig_len_max": max(sig_lens, default=0),
        "sig_len_min": min(sig_lens, default=0),
        "frac_low_r": float(np.mean(low_r)) if low_r else -1.0,
        "all_low_r": int(bool(low_r) and all(low_r)),
        "sighash_all": int(bool(sighashes) and all(s == 1 for s in sighashes)),
        "wit_items_max": max(wit_items, default=0),
        "n_multisig_in": multisig,
        "n_taproot_in": taproot,
        "log_in_sum": np.log1p(in_sum),
        "log_in_max": np.log1p(max(in_vals, default=0)),
        "log_in_min": np.log1p(min(in_vals, default=0)),
        "log_out_sum": np.log1p(out_sum),
        "log_out_max": np.log1p(max(vals, default=0)),
        "out_in_ratio": out_sum / in_sum if in_sum else 0.0,
        "n_in_addrs": len(in_addrs),
        "n_out_addrs": len({a for a in out_addr_list if a}),
        "outs_ascending": int(len(vals) > 1 and vals == sorted(vals)),
        "out_has_round": int(any(v > 0 and v % 10_000 == 0 for v in vals)),
    }
    for k in KINDS:
        feats[f"n_in_{k}"] = in_k.get(k, 0)
        feats[f"n_out_{k}"] = out_k.get(k, 0)

    meta = {
        "tx_hash": tx["hash"],
        "block": block_name,
        "time": tx.get("time", 0),
        "label": label,
        "outs": tuple((o.get("addr"), o["value"]) for o in outs if o.get("addr")),
        "ins": tuple(sorted(in_addrs)),
    }
    return {**meta, **feats}


def scan_block(args):
    path, target = args
    with open(path, "rb") as f:
        block = loads(f.read())
    txs = block["tx"] if isinstance(block, dict) else block
    name = Path(path).stem
    return [r for tx in txs if (r := tx_record(tx, target, name))]


def load_dataset(directory, target, workers, cache, use_cache):
    cache = Path(cache)
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            saved = pickle.load(f)
        if saved["target"] == target:
            print(f"Dataset lido do cache {cache} ({len(saved['df']):,} transações)")
            return saved["df"]
        print("Cache é de outro TARGET; reprocessando os blocos.")

    files = sorted(Path(directory).glob("*"))
    if not files:
        raise SystemExit(f"Nenhum arquivo em {directory}")
    print(f"Lendo {len(files)} blocos...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(scan_block, [(p, target) for p in files], chunksize=4):
            rows.extend(r)
    df = pd.DataFrame(rows)
    with open(cache, "wb") as f:
        pickle.dump({"target": target, "df": df}, f)
    print(f"Cache salvo em {cache}")
    return df


# =====================================================================
# 2) MODELOS (lista do exemplo do professor, adaptada ao tamanho dos dados)
# =====================================================================


def make_models(seed):
    sc = StandardScaler
    return {
        "Dummy (piso)": make_pipeline(sc(), DummyClassifier(strategy="prior")),
        "Decision Tree": make_pipeline(
            sc(),
            DecisionTreeClassifier(
                max_depth=5, class_weight="balanced", random_state=seed
            ),
        ),
        "Random Forest": make_pipeline(
            sc(),
            RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
        ),
        "HistGradientBoosting": make_pipeline(
            sc(),
            HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
        ),
        "AdaBoost": make_pipeline(
            sc(), AdaBoostClassifier(n_estimators=100, random_state=seed)
        ),
        "Naive Bayes": make_pipeline(sc(), GaussianNB()),
        "Logistic Regression": make_pipeline(
            sc(), LogisticRegression(class_weight="balanced", max_iter=2000)
        ),
        "Linear SVM": make_pipeline(
            sc(),
            LinearSVC(
                C=0.025, class_weight="balanced", max_iter=5000, random_state=seed
            ),
        ),
        "Nearest Neighbors": make_pipeline(sc(), KNeighborsClassifier(5, n_jobs=-1)),
        "Neural Net": make_pipeline(
            sc(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1, max_iter=300, random_state=seed
            ),
        ),
        "QDA": make_pipeline(sc(), QuadraticDiscriminantAnalysis(reg_param=0.1)),
    }


SCORING = {
    "ap": "average_precision",
    "roc_auc": "roc_auc",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}


def build_sample(df, neg_sample, seed):
    """Todos os positivos + uma amostra de negativos (para modelos lentos)."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(df["label"].to_numpy() == 1)
    neg = np.flatnonzero(df["label"].to_numpy() == 0)
    if len(neg) > neg_sample:
        neg = rng.choice(neg, neg_sample, replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    return idx


def compare_models(models, X, y, groups, folds, seed, outdir):
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(cv.split(X, y, groups))
    for k, (_, te) in enumerate(splits):
        if y[te].sum() == 0:
            print(
                f"  AVISO: fold {k} sem positivos no teste (poucos blocos com positivos?)"
            )

    rows = []
    for name, model in models.items():
        print(f"  {name}...", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = cross_validate(
                model,
                X,
                y,
                cv=splits,
                scoring=SCORING,
                error_score=np.nan,
                return_train_score=False,
            )
        row = {"modelo": name, "tempo_fit_s": np.nanmean(res["fit_time"])}
        for m in SCORING:
            row[m] = np.nanmean(res[f"test_{m}"])
            row[f"{m}_std"] = np.nanstd(res[f"test_{m}"])
        rows.append(row)
    table = pd.DataFrame(rows).set_index("modelo")
    table.to_csv(Path(outdir) / "resultados_modelos.csv")
    return table, splits


# =====================================================================
# 3) PONTUAR TODAS AS TRANSAÇÕES (out-of-fold, por bloco) E ACHAR CANDIDATAS
# =====================================================================


def score_of(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def oof_scores(model, X, y, groups, folds, neg_train, seed):
    """Score de cada transação por um modelo que NÃO viu o bloco dela."""
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)
    scores = np.zeros(len(y))
    for tr, te in cv.split(X, y, groups):
        pos = tr[y[tr] == 1]
        neg = tr[y[tr] == 0]
        if len(neg) > neg_train:
            neg = rng.choice(neg, neg_train, replace=False)
        idx = np.concatenate([pos, neg])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = clone(model).fit(X[idx], y[idx])
        scores[te] = score_of(m, X[te])
    return scores


def best_f1_threshold(y, scores):
    p, r, t = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    k = int(np.nanargmax(f1))
    return t[k], p[k], r[k], f1[k]


def destination_report(df, scores, thr, max_cand, outdir):
    unl = df["label"].to_numpy() == 0
    cand_idx = np.flatnonzero(unl & (scores >= thr))
    cand_idx = cand_idx[np.argsort(-scores[cand_idx])][:max_cand]

    cand_n = Counter()
    cand_val = defaultdict(int)
    cand_score = defaultdict(float)
    cand_origins = defaultdict(set)
    outs_col = df["outs"].to_numpy()
    ins_col = df["ins"].to_numpy()
    for i in cand_idx:
        for addr, val in outs_col[i]:
            cand_n[addr] += 1
            cand_val[addr] += val
            cand_score[addr] += scores[i]
            cand_origins[addr].update(ins_col[i])

    total_n = Counter()
    for i in np.flatnonzero(unl):
        for addr, _ in outs_col[i]:
            if addr in cand_n:
                total_n[addr] += 1

    rows = [
        {
            "destino": a,
            "txs_candidatas": n,
            "txs_totais_no_destino": total_n[a],
            "frac_candidatas": n / total_n[a],
            "origens_distintas": len(cand_origins[a]),
            "valor_btc": cand_val[a] / SAT,
            "score_medio": cand_score[a] / n,
        }
        for a, n in cand_n.items()
    ]
    dest = (
        pd.DataFrame(rows).sort_values(["txs_candidatas", "valor_btc"], ascending=False)
        if rows
        else pd.DataFrame()
    )
    dest.to_csv(Path(outdir) / "destinos_candidatos.csv", index=False)
    return cand_idx, dest


# =====================================================================
# 4) EXTRAS: métodos do capítulo 3 sobre a árvore de decisão
# =====================================================================


def run_extras(X, y, groups, splits, _feature_names, seed, outdir, best_name, models):
    tree = DecisionTreeClassifier(class_weight="balanced", random_state=seed)

    print("\n[3.5.1] validation_curve em max_depth")
    depths = [1, 2, 3, 4, 5, 6, 8, 10, 15]
    tr, te = validation_curve(
        tree,
        X,
        y,
        param_name="max_depth",
        param_range=depths,
        cv=splits,
        scoring="average_precision",
    )
    for d, a, b in zip(depths, tr.mean(1), te.mean(1)):
        print(f"  max_depth={d:>2}  treino AP={a:.4f}  validação AP={b:.4f}")
    plt.figure(figsize=(6, 4))
    plt.plot(depths, tr.mean(1), "o-", label="treino")
    plt.plot(depths, te.mean(1), "s-", label="validação")
    plt.xlabel("max_depth")
    plt.ylabel("Average precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Curva de validação (Decision Tree)")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "validation_curve.png", dpi=150)
    plt.close()

    print("\n[3.5.2] learning_curve")
    sizes, tr, te = learning_curve(
        DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed),
        X,
        y,
        cv=splits,
        scoring="average_precision",
        train_sizes=np.linspace(0.1, 1.0, 5),
        shuffle=True,
        random_state=seed,
        error_score=np.nan,
    )
    for s, a, b in zip(sizes, np.nanmean(tr, 1), np.nanmean(te, 1)):
        print(f"  n_treino={s:>7}  treino AP={a:.4f}  validação AP={b:.4f}")
    plt.figure(figsize=(6, 4))
    plt.plot(sizes, np.nanmean(tr, 1), "o-", label="treino")
    plt.plot(sizes, np.nanmean(te, 1), "s-", label="validação")
    plt.xlabel("Tamanho do treino")
    plt.ylabel("Average precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.title("Curva de aprendizado (Decision Tree)")
    plt.tight_layout()
    plt.savefig(Path(outdir) / "learning_curve.png", dpi=150)
    plt.close()

    print("\n[3.2.1] GridSearchCV na árvore")
    gs = GridSearchCV(
        tree,
        {
            "max_depth": [3, 5, 8, 12],
            "min_samples_leaf": [1, 5, 20],
            "ccp_alpha": [0.0, 1e-4, 1e-3],
        },
        scoring="average_precision",
        cv=splits,
        n_jobs=-1,
    )
    gs.fit(X, y)
    print(f"  melhores parâmetros: {gs.best_params_}  (AP CV = {gs.best_score_:.4f})")

    print("\n[3.1.5] permutation_test_score (20 permutações)")
    score, perm, pval = permutation_test_score(
        DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=seed),
        X,
        y,
        groups=groups,
        cv=StratifiedGroupKFold(n_splits=len(splits), shuffle=True, random_state=seed),
        scoring="average_precision",
        n_permutations=20,
        random_state=seed,
        n_jobs=-1,
    )
    print(
        f"  AP real = {score:.4f} | AP com rótulos embaralhados = {perm.mean():.4f} "
        f"| p-valor = {pval:.4f}"
    )

    print(f"\n[3.3.1] TunedThresholdClassifierCV no melhor modelo ({best_name})")
    try:
        from sklearn.model_selection import TunedThresholdClassifierCV

        tuned = TunedThresholdClassifierCV(
            clone(models[best_name]), scoring="f1", cv=splits, refit=True, n_jobs=1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tuned.fit(X, y)
        print(
            f"  limiar que maximiza F1 (na amostra de treino): {tuned.best_threshold_:.4f}"
            f" | F1 = {tuned.best_score_:.4f}"
        )
    except ImportError:
        print("  (precisa de scikit-learn >= 1.5)")


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("directory", nargs="?", default="blocos/")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument(
        "--neg-sample",
        type=int,
        default=100_000,
        help="negativos usados no treino/comparação (todos os positivos entram)",
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--models",
        default="",
        help='subconjunto separado por vírgula, ex.: "Decision Tree,Random Forest"',
    )
    ap.add_argument(
        "--final-model",
        default="",
        help="modelo usado para pontuar as candidatas (padrão: melhor AP)",
    )
    ap.add_argument(
        "--top", type=int, default=500, help="candidatas salvas em candidatas.csv"
    )
    ap.add_argument(
        "--recall-target",
        type=float,
        default=0.95,
        help="candidata = não rotulada com score >= ao dos positivos conhecidos "
        "neste quantil de recall (padrão 0,95)",
    )
    ap.add_argument(
        "--extras", action="store_true", help="curvas, grid, permutação, limiar"
    )
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache", default="features_cache.pkl")
    ap.add_argument("--outdir", default="saida_ml")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- dataset ----------------
    df = load_dataset(
        args.directory, args.target, args.workers, args.cache, not args.no_cache
    )
    features = [c for c in df.columns if c not in META_COLS]
    y_all = df["label"].to_numpy()
    print(
        f"\n{len(df):,} transações | positivos (hack): {int(y_all.sum())} | "
        f"negativos: {int((y_all == 0).sum()):,} | features: {len(features)}"
    )
    if y_all.sum() < args.folds:
        raise SystemExit("Positivos insuficientes para a validação cruzada.")

    X_all = df[features].to_numpy(dtype=np.float64)
    groups_all = pd.factorize(df["block"])[0]

    # ---------------- comparação de modelos (amostra) ----------------
    idx = build_sample(df, args.neg_sample, args.seed)
    X, y, groups = X_all[idx], y_all[idx], groups_all[idx]
    print(
        f"\nAmostra de comparação: {len(idx):,} linhas "
        f"({int(y.sum())} positivos, prevalência {y.mean():.3%})"
    )

    models = make_models(args.seed)
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = {
            k: v for k, v in models.items() if k in wanted or k.startswith("Dummy")
        }
    print("\nComparando modelos (StratifiedGroupKFold por bloco):")
    table, splits = compare_models(models, X, y, groups, args.folds, args.seed, outdir)

    show = table[["ap", "roc_auc", "precision", "recall", "f1", "tempo_fit_s"]].round(4)
    print("\n" + show.sort_values("ap", ascending=False).to_string())
    print(
        "\n(Precision/AP aqui usam a prevalência da amostra; os números no dataset "
        "completo, mais abaixo, são os realistas.)"
    )

    # ---------------- árvore de decisão: regras ----------------
    tree = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=5, class_weight="balanced", random_state=args.seed
    )
    tree.fit(X, y)
    rules = export_text(tree, feature_names=features)
    (outdir / "arvore.txt").write_text(rules)
    print("\nRegras da árvore de decisão (max_depth=5), também em arvore.txt:")
    print(rules)
    imp = pd.Series(tree.feature_importances_, index=features).sort_values(
        ascending=False
    )
    print("Features mais importantes na árvore:")
    print(imp.head(8).round(4).to_string())

    # ---------------- escolha do modelo final ----------------
    cand_models = table.drop(index=[i for i in table.index if i.startswith("Dummy")])
    final_name = args.final_model or cand_models["ap"].idxmax()
    print(f"\nModelo final para pontuar o dataset completo: {final_name}")

    # ---------------- pontuação out-of-fold do dataset completo ----------------
    print("Pontuando todas as transações (treino sem o bloco avaliado)...", flush=True)
    scores = oof_scores(
        models[final_name],
        X_all,
        y_all,
        groups_all,
        args.folds,
        args.neg_sample,
        args.seed,
    )
    ap_full = average_precision_score(y_all, scores)
    thr, p, r, f1 = best_f1_threshold(y_all, scores)
    pred = (scores >= thr).astype(int)
    print(f"\nDataset completo (prevalência real {y_all.mean():.4%}):")
    print(f"  average precision = {ap_full:.4f}")
    print(
        f"  limiar de melhor F1 = {thr:.4f} -> precision {p:.4f}, recall {r:.4f}, F1 {f1:.4f}"
    )
    print("  matriz de confusão [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_all, pred))
    print(classification_report(y_all, pred, target_names=["resto", "hack"], digits=4))

    prec, rec, _ = precision_recall_curve(y_all, scores)
    plt.figure(figsize=(6, 4.5))
    plt.plot(rec, prec, lw=2)
    plt.scatter([r], [p], color="red", zorder=3, label=f"melhor F1 ({f1:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title(f"Curva precision-recall — {final_name} (AP = {ap_full:.3f})")
    plt.tight_layout()
    plt.savefig(outdir / "pr_curve.png", dpi=150)
    plt.close()

    # ---------------- candidatas a outros ataques ----------------
    # Atenção: o limiar de melhor F1 acima trata as não rotuladas como "resto" e por
    # isso empurra o limiar para cima justamente das que se parecem com o ataque.
    # Para achar candidatas, usamos o score dos POSITIVOS conhecidos: candidata é toda
    # transação não rotulada que pontua tão alto quanto os positivos (recall alvo).
    thr_cand = float(np.quantile(scores[y_all == 1], 1 - args.recall_target))
    _cand_idx, dest = destination_report(df, scores, thr_cand, 20_000, outdir)
    unl = np.flatnonzero(y_all == 0)
    n_above = int((scores[unl] >= thr_cand).sum())
    print(
        f"Limiar de candidata (recall {args.recall_target:.0%} nos positivos conhecidos): "
        f"{thr_cand:.4f}"
    )
    print(
        f"Transações NÃO rotuladas acima do limiar: {n_above:,} "
        f"(analisadas até 20.000 de maior score)"
    )

    top = df.iloc[unl[np.argsort(-scores[unl])][: args.top]].copy()
    top["score"] = np.sort(scores[unl])[::-1][: args.top]
    top["outs"] = top["outs"].apply(lambda o: ";".join(f"{a}:{v}" for a, v in o))
    top["ins"] = top["ins"].apply(lambda t: ";".join(t))
    top[
        ["tx_hash", "block", "time", "score", "vin_sz", "vout_sz", "outs", "ins"]
    ].to_csv(outdir / "candidatas.csv", index=False)

    if len(dest):
        print(
            "\nDestinos que mais concentram candidatas (um ataque deve aparecer como "
            "muitas candidatas, de origens distintas, convergindo para o mesmo destino):"
        )
        cols = [
            "destino",
            "txs_candidatas",
            "txs_totais_no_destino",
            "frac_candidatas",
            "origens_distintas",
            "valor_btc",
            "score_medio",
        ]
        print(dest[cols].head(15).round(4).to_string(index=False))
    else:
        print("Nenhuma candidata acima do limiar.")

    if args.extras:
        run_extras(
            X, y, groups, splits, features, args.seed, outdir, final_name, models
        )

    print(f"\nArquivos salvos em {outdir}/")
