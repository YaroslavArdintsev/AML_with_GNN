import argparse
import json
import random
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from graph import TemporalGraph
from model import TransactionGNN
from prepare import fingerprint


def to_device(value, device):
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return torch.from_numpy(value).to(device)


def batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


@torch.no_grad()
def predict(model, graph, indices, config, batch_size, device):
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)

    position = 0
    for ids in batches(indices, batch_size):
        batch = graph.batch(ids, config["depth"], config["fanout"])
        batch = to_device(batch, device)
        logits = model(batch).cpu().numpy()

        output[position:position + len(ids)] = logits
        position += len(ids)

    return output


def report(graph, indices, logits, threshold, alert_rate):
    y = np.asarray(graph.y[indices], dtype=np.int64)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Для оценки нужны оба класса")

    predicted = logits >= threshold

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, predicted, average="binary", zero_division=0
    )

    # Отдельная метрика: лучшие K операций КАЖДОГО дня.
    # Это оценка очереди после завершения дня, а не онлайн-порог.
    days = graph.time[indices] // 86400
    reviewed = 0
    found = 0

    for day in np.unique(days):
        rows = np.flatnonzero(days == day)
        count = max(1, int(np.ceil(len(rows) * alert_rate)))
        order = np.argsort(-logits[rows], kind="stable")
        selected = rows[order[:count]]

        reviewed += len(selected)
        found += int(y[selected].sum())

    return {
        "rows": len(indices),
        "positives": int(y.sum()),
        "prevalence": float(y.mean()),
        "AP": float(average_precision_score(y, logits)),
        "ROC_AUC": float(roc_auc_score(y, logits)),
        "threshold_logit": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "F1": float(f1),
        "alerts": int(predicted.sum()),
        "false_positives": int(((y == 0) & predicted).sum()),
        "false_negatives": int(((y == 1) & ~predicted).sum()),
        "daily_budget_fraction": alert_rate,
        "daily_budget_reviews": reviewed,
        "daily_budget_precision": found / max(reviewed, 1),
        "daily_budget_recall": found / int(y.sum()),
    }


def model_config(graph, args):
    pp = graph.preprocessing
    category_sizes = [
        max(pp["vocabularies"][name].values()) + 1
        for name in pp["categories"]
    ]
    return {
        "category_sizes": category_sizes,
        "numeric_size": int(graph.num.shape[1]),
        "hidden_size": args.hidden,
        "embedding_size": 8,
        "depth": args.depth,
        "dropout": 0.1,
    }


def train(args, graph, device):
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Для обучения используйте пустую директорию --out")

    train_ids = np.flatnonzero(graph.split == 0)
    validation_ids = np.flatnonzero(graph.split == 1)

    for indices in [train_ids, validation_ids]:
        if set(np.unique(graph.y[indices])) != {0, 1}:
            raise ValueError("В train и validation должны быть оба класса")

    architecture = model_config(graph, args)
    config = {"depth": args.depth, "fanout": args.fanout}

    model = TransactionGNN(**architecture).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )

    positives = int(graph.y[train_ids].sum())
    negatives = len(train_ids) - positives
    weight = (
        args.pos_weight
        if args.pos_weight is not None
        else negatives / positives
    )

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([weight], device=device)
    )

    rng = np.random.default_rng(args.seed)
    best_ap = -1.0
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = rng.permutation(train_ids)

        total_loss = 0.0
        seen = 0

        for step, ids in enumerate(batches(shuffled, args.batch_size), 1):
            batch = graph.batch(ids, args.depth, args.fanout)
            batch = to_device(batch, device)
            target = torch.tensor(
                graph.y[ids], dtype=torch.float32, device=device
            )

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, target)

            if not torch.isfinite(loss):
                raise RuntimeError("Обнаружен нечисловой loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item() * len(ids)
            seen += len(ids)

            if step % 200 == 0:
                print(
                    f"epoch={epoch} step={step} "
                    f"loss={total_loss / seen:.5f}",
                    flush=True,
                )

        validation_logits = predict(
            model, graph, validation_ids, config, args.batch_size, device
        )
        ap = average_precision_score(
            graph.y[validation_ids], validation_logits
        )

        print(
            f"epoch={epoch} loss={total_loss / seen:.5f} "
            f"validation_AP={ap:.6f}",
            flush=True,
        )

        if ap > best_ap:
            best_ap = float(ap)
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

            # Промежуточный checkpoint на случай прерывания обучения.
            torch.save({
                "model_state_dict": best_state,
                "architecture": architecture,
                "sampling": config,
                "preprocessing": graph.preprocessing,
                "encoder_hash": fingerprint(graph.preprocessing),
                "epoch": best_epoch,
                "validation_AP": best_ap,
            }, output / "best_unfinalized.pt")
        else:
            stale_epochs += 1

        if stale_epochs >= args.patience:
            break

    model.load_state_dict(best_state)

    validation_logits = predict(
        model, graph, validation_ids, config, args.batch_size, device
    )

    # Настройка порога без использования test.
    threshold = float(np.quantile(
        validation_logits,
        1.0 - args.alert_rate,
        method="higher",
    ))

    checkpoint = {
        "model_state_dict": best_state,
        "architecture": architecture,
        "sampling": config,
        "preprocessing": graph.preprocessing,
        "encoder_hash": fingerprint(graph.preprocessing),
        "threshold_logit": threshold,
        "alert_rate": args.alert_rate,
        "positive_weight": float(weight),
        "epoch": best_epoch,
        "validation_AP": best_ap,
        "seed": args.seed,
        "torch_version": str(torch.__version__),
    }

    checkpoint_path = output / "model.pt"
    torch.save(checkpoint, checkpoint_path)

    (output / "preprocess.json").write_text(
        json.dumps(graph.preprocessing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Проверяем, что после сохранения и загрузки предсказания совпадают.
    restored_checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    restored = TransactionGNN(
        **restored_checkpoint["architecture"]
    ).to(device)
    restored.load_state_dict(restored_checkpoint["model_state_dict"])

    check_ids = validation_ids[:args.batch_size]
    original = predict(
        model, graph, check_ids, config, args.batch_size, device
    )
    reloaded = predict(
        restored, graph, check_ids, config, args.batch_size, device
    )
    np.testing.assert_allclose(original, reloaded, rtol=1e-5, atol=1e-6)

    metrics = report(
        graph, validation_ids, validation_logits,
        threshold, args.alert_rate,
    )
    (output / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def evaluate_or_score(args, graph, device):
    if not args.checkpoint:
        raise ValueError("Нужен --checkpoint")

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )

    if fingerprint(graph.preprocessing) != checkpoint["encoder_hash"]:
        raise ValueError("Данные подготовлены другим кодировщиком")

    model = TransactionGNN(**checkpoint["architecture"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    part = 2 if args.mode == "evaluate" else 3
    indices = np.flatnonzero(graph.split == part)
    if len(indices) == 0:
        raise ValueError("Нет операций для выбранного режима")

    logits = predict(
        model,
        graph,
        indices,
        checkpoint["sampling"],
        args.batch_size,
        device,
    )

    threshold = checkpoint["threshold_logit"]
    risk_score = torch.sigmoid(torch.from_numpy(logits)).numpy()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({
        "row_id": graph.row_id[indices],
        "timestamp": graph.time[indices],
        "logit": logits,
        "risk_score": risk_score,
        "alert": logits >= threshold,
    })

    if args.mode == "evaluate":
        predictions["label"] = graph.y[indices]

        metrics = report(
            graph, indices, logits,
            threshold, checkpoint["alert_rate"],
        )
        (output / "test_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2))

        precision, recall, thresholds = precision_recall_curve(
            graph.y[indices], logits
        )
        np.savez(
            output / "pr_curve.npz",
            precision=precision,
            recall=recall,
            thresholds=thresholds,
        )

    predictions.sort_values("row_id").to_csv(
        output / "predictions.csv", index=False
    )
    print(f"Сохранено: {output / 'predictions.csv'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "evaluate", "score"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--depth", type=int, choices=[0, 1, 2], default=2)
    parser.add_argument("--fanout", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pos-weight", type=float)
    parser.add_argument("--alert-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if not 0 < args.alert_rate < 1:
        raise ValueError("--alert-rate должен находиться между 0 и 1")
    if min(args.batch_size, args.fanout, args.epochs, args.patience) < 1:
        raise ValueError("Размеры и число эпох должны быть положительными")
    if args.pos_weight is not None and args.pos_weight <= 0:
        raise ValueError("--pos-weight должен быть положительным")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    graph = TemporalGraph(args.data)
    device = torch.device(args.device)

    if args.mode == "train":
        train(args, graph, device)
    else:
        evaluate_or_score(args, graph, device)


if __name__ == "__main__":
    main()