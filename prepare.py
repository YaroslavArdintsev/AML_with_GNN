import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from pathlib import Path

CATS = ['Payment_currency', 'Received_currency', 'Sender_bank_location',
        'Receiver_bank_location', 'Payment_type']
ACCS = ['Sender_account', 'Receiver_account']
INPUTS = ['Time', 'Date', 'Amount'] + CATS + ACCS
LABEL = 'Is_laundering'
MISSING = '__MISSING__'


def fingerprint(obj):
    text = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            result.update(chunk)

    return result.hexdigest()


def read_csv(path: Path, time_format: str = "%Y-%m-%d %H:%M:%S", require_labels: bool = False):
        wanted = set(INPUTS)

        if require_labels:
                wanted.add(LABEL)

        string_columns = ["Date", "Time"] + ACCS + CATS

        frame = pd.read_csv(
                path,
                usecols=lambda name: name in wanted,
                dtype={name: "string" for name in string_columns}
        )

        missing = set(INPUTS) - set(frame.columns)

        if missing:
                raise ValueError(
                        f"Нет обязательных столбцов: {sorted(missing)}"
                )

        if frame.empty:
                raise ValueError(f"Пустой файл: {path}")

        for name in ACCS + ["Date", "Time"]:
                frame[name] = frame[name].str.strip()

                if frame[name].isna().any() or frame[name].eq("").any():
                        raise ValueError(f"Пропущены значения в {name}")

        for name in CATS:
                frame[name] = (
                        frame[name]
                        .str.strip()
                        .replace("", pd.NA)
                        .fillna(MISSING)
                )

        amount = pd.to_numeric(frame["Amount"], errors="raise")
        if not np.isfinite(amount.to_numpy()).all() or (amount < 0).any():
                raise ValueError(
                        "Amount должен быть конечным неотрицательным числом"
                )

        if require_labels:
                if LABEL not in frame:
                        raise ValueError(f"Для обучения нужен {LABEL}")

                labels = pd.to_numeric(frame[LABEL], errors="raise")

                if not labels.isin([0, 1]).all():
                        raise ValueError("Метки должны быть только 0 или 1")

                frame["_label"] = labels.astype(np.int8)
                frame = frame.drop(columns=[LABEL])

        else:
                frame["_label"] = np.int8(-1)

        timestamp = pd.to_datetime(
                frame["Date"] + " " + frame["Time"],
                format=time_format,
                errors="raise",
        )

        frame["_time"] = timestamp.to_numpy(
                dtype="datetime64[s]"
        ).astype(np.int64)

        frame["_row"] = np.arange(len(frame), dtype=np.int64)

        frame = frame.sort_values(
                "_time",
                kind="stable"
        ).reset_index(drop=True)

        return frame


def numeric_features(frame: pd.DataFrame) -> np.ndarray:
        times = frame['_time'].to_numpy()
        hour = (times % 86400) / 3600
        weekday = (times // 86400 + 3) % 7

        return np.column_stack([
                np.log1p(frame["Amount"].to_numpy(dtype=np.float64)),
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * weekday / 7),
                np.cos(2 * np.pi * weekday / 7),
                (frame['Payment_currency'] != frame['Received_currency']).to_numpy(dtype=np.float32),
                (frame['Sender_bank_location'] != frame['Receiver_bank_location']).to_numpy(dtype=np.float32)
        ]).astype(np.float32)

def save_array(folder: Path, name: str, values: np.ndarray) -> None:
        np.save(folder / f'{name}.npy', values, allow_pickle=False)

def main():
        parser = argparse.ArgumentParser()
        parser.add_argument('--csv', required=True)
        parser.add_argument('--out', required=True)
        parser.add_argument('--encoder')
        parser.add_argument('--history')
        parser.add_argument('--time-format', default="%Y-%m-%d %H:%M:%S")
        args = parser.parse_args()

        output = Path(args.out)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
                raise ValueError('Выходная директория должна быть пустой!')

        applying = args.encoder is not None
        if args.history and not applying:
                raise ValueError('--history используется вместе с --encoder')

        preprocessing = None
        if applying:
                preprocessing = json.loads(Path(args.encoder).read_text(encoding="utf-8"))
                time_format = preprocessing['time_format']
        else:
                time_format = args.time_format

        frame = read_csv(args.csv, time_format, require_labels=(not applying))
        frame['_requested'] = True

        if args.history:
                history = read_csv(args.history, time_format, require_labels=False)
                if history['_time'].max() > frame['_time'].min():
                        raise ValueError('История должна заканчиваться не позже начала текущих данных')
                history['_requested'] = False
                frame = pd.concat([history, frame], ignore_index=True)
                del history

        frame = frame.sort_values('_time', kind="stable").reset_index(drop=True)
        times = frame['_time'].to_numpy(dtype=np.int64)
        labels = frame['_label'].to_numpy(dtype=np.int8)
        num = numeric_features(frame)

        if applying:
                split = np.where(frame['_requested'], 3, -1).astype(np.int8)
        else:
                days = np.unique(times // 86400)
                if len(days) < 7:
                        raise ValueError('Для разбиения нужны все 7 дней недели')
                validation_start = int(days[int(len(days) * 0.70)] * 86400)
                test_start = int(days[int(len(days) * 0.85)] * 86400)

                split = np.where(times < validation_start, 0, np.where(times < test_start, 1, 2)).astype(np.int8)

                for part in 0, 1, 2:
                        if set(np.unique(labels[split == part])) != {0, 1}:
                                raise ValueError(f'В части {part} нет хотя бы одного из классов')

                train = split == 0
                vocabularies = {}

                for name in CATS:
                        values = sorted(set(frame.loc[train, name]) - {MISSING})
                        vocabulary = {MISSING: 1}
                        vocabulary.update({value: i + 2 for i, value in enumerate(values)})
                        vocabularies[name] = vocabulary

                preprocessing = {
                        'schema_version': 1,
                        'time_format': time_format,
                        'categories': CATS,
                        'vocabularies': vocabularies,
                        'numeric_mean': num[train].mean(axis=0).tolist(),
                        'numeric_std': np.maximum(num[train].std(axis=0), 1e-6).tolist(),
                        'validation_start': validation_start,
                        'test_start': test_start
                }

        mean = np.asarray(preprocessing['numeric_mean'], dtype=np.float32)
        std = np.asarray(preprocessing['numeric_std'], dtype=np.float32)
        num = ((num - mean) / std).astype(np.float32)

        if not np.isfinite(num).all():
                raise ValueError('После преобразования получены NaN/inf')

        cat = np.column_stack([
                frame[name].map(preprocessing['vocabularies'][name])
                .fillna(0).to_numpy(dtype=np.int64)
                for name in CATS
        ])

        all_accounts = pd.concat(
                [frame['Sender_account'],
                frame['Receiver_account']],
                ignore_index=True
        )
        codes, account_names = pd.factorize(all_accounts, sort=False)
        count = len(frame)
        src = codes[:count].astype(np.int64)
        dst = codes[count:].astype(np.int64)
        num_nodes = len(account_names)
        del all_accounts, codes

        arrays = {
                'time': times,
                'src': src,
                'dst': dst,
                'num': num,
                'cat': cat,
                'y': labels,
                'split': split,
                'row_id': frame['_row'].to_numpy(dtype=np.int64)
        }

        for name, values in arrays.items():
                save_array(output, name, values)

        for direction, nodes in [('out', src), ('in', dst)]:
                order = np.argsort(nodes, kind='stable')
                counts = np.bincount(nodes, minlength=num_nodes)
                ptr = np.concatenate([[0], counts.cumsum()]).astype(np.int64)

                save_array(output, f'{direction}_order', order)
                save_array(output, f'{direction}_ptr', ptr)
                save_array(output, f'{direction}_time', times[order])

        pd.DataFrame(
                {"node_id": np.arange(num_nodes),
                 "account": account_names,}
        ).to_csv(output / 'accounts.csv', index=False)

        (output / 'preprocess.json').write_text(
                json.dumps(preprocessing, ensure_ascii=False, indent=2),
                encoding="utf-8"
        )

        manifest = {
                "rows": count,
                "nodes": num_nodes,
                "encoder_hash": fingerprint(preprocessing),
                "input_sha256": file_hash(args.csv),
                "history_sha256": file_hash(args.history) if args.history else None,
                "min_timestamp": int(times.min()),
                "max_timestamp": int(times.max()),
                "parts": {},
        }

        for part in np.unique(split):
                mask = split == part
                manifest["parts"][str(int(part))] = {
                        "rows": int(mask.sum()),
                        "positives": int((labels[mask] == 1).sum()),
                }

        (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
        main()