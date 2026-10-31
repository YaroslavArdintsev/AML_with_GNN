import json
import numpy as np
from pathlib import Path


def lower_bounds(values, starts, ends, targets):
    left = starts.copy()
    right = ends.copy()

    while True:
        active = np.flatnonzero(left < right)
        if len(active) == 0:
            return left

        middle = (left[active] + right[active]) // 2
        earlier = values[middle] < targets[active]

        left[active] = np.where(earlier, middle + 1, left[active])
        right[active] = np.where(earlier, right[active], middle)


class TemporalGraph:
    def __init__(self, folder):
        folder = Path(folder)

        names = [
            "time", "src", "dst", "num", "cat", "y", "split", "row_id",
            "in_order", "in_ptr", "in_time",
            "out_order", "out_ptr", "out_time",
        ]

        for name in names:
            setattr(
                self, name,
                np.load(folder / f'{name}.npy', mmap_mode="r")
            )

        self.preprocessing = json.loads(
            (folder / "preprocess.json").read_text(encoding="utf-8")
        )

    def node_tree(self, nodes, cutoffs, depth, fanout):
        valid_node = nodes >= 0
        safe_nodes = np.maximum(nodes, 0)

        node_features = []
        selected_edges = []
        selected_neighbors = []
        selected_masks = []

        for direction in ('in', 'out'):
            ptr = getattr(self, f'{direction}_ptr')
            order = getattr(self, f'{direction}_order')
            timeline = getattr(self, f'{direction}_time')

            starts = ptr[safe_nodes]
            ends = ptr[safe_nodes + 1]

            boundaries = cutoffs[:, None] - np.array(
                [0, 3600, 86400, 7*86400], dtype=np.int64
            )

            positions = lower_bounds(
                timeline,
                np.repeat(starts, 4),
                np.repeat(ends, 4),
                boundaries.reshape(-1)
            ).reshape(-1, 4)

            stop = positions[:, 0]
            counts = np.column_stack([
                stop - starts,
                stop - positions[:, 1],
                stop - positions[:, 2],
                stop - positions[:, 3]
            ])

            has_history = valid_node & (stop > starts)
            last_position = np.clip(stop - 1, 0, len(timeline) - 1)
            age = np.maximum(cutoffs - timeline[last_position], 0)
            age = np.minimum(age, 30*86400) / 3600.0
            age = np.where(has_history, age, 0)

            features = np.column_stack([
                np.log1p(counts),
                np.log1p(age),
                (~has_history).astype(np.float32)
            ]).astype(np.float32)
            features[~valid_node] = 0
            node_features.append(features)

            if depth == 0:
                continue

            positions_to_take = (
                stop[:, None] - np.arange(fanout, 0, -1)[None, :]
            )
            mask = (
            (positions_to_take >= starts[:, None])
                & valid_node[:, None]
            )

            safe_positions = np.clip(
                positions_to_take, 0, len(order) - 1
            )
            edge_ids = order[safe_positions]

            neighbor = (
                self.src[edge_ids]
                if direction == 'in'
                else self.dst[edge_ids]
            )

            selected_edges.append(edge_ids)
            selected_neighbors.append(np.where(mask, neighbor, -1))
            selected_masks.append(mask)

        result = {
            'X': np.concatenate(node_features, axis=1).astype(np.float32)
        }

        if depth == 0:
            return result

        edges = np.concatenate(selected_edges, axis=1)
        neighbors = np.concatenate(selected_neighbors, axis=1)
        masks = np.concatenate(selected_masks, axis=1)

        historical_times = self.time[edges]
        allowed = np.broadcast_to(cutoffs[:, None], edges.shape)

        if not np.all(historical_times[mask] < allowed[mask]):
            raise RuntimeError('В исторический граф попало будущее событие')

        edge_num = np.array(self.num[edges], copy=True)
        edge_cat = np.array(self.cat[edges], copy=True)
        edge_num[~mask] = 0
        edge_cat[~mask] = 0

        age = np.log1p(
            np.maximum(allowed - historical_times, 0) / 3600.0
        ).astype(np.float32)
        age[~mask] = 0

        child_cutoffs = np.where(mask, historical_times, 0)

        result.update({
            'num': edge_num,
            'cat': edge_cat,
            'age': age[..., None],
            'mask': mask,
            'child': self.node_tree(
                neighbors.reshape(-1),
                child_cutoffs.reshape(-1),
                depth - 1,
                fanout
            )
        })
        return result

    def batch(self, event_ids, depth, fanout):
        event_ids = np.asarray(event_ids, dtype=np.int64)

        nodes = np.stack([
            self.src[event_ids],
            self.dst[event_ids]
        ], axis=1).reshape(-1)

        cutoffs = np.repeat(self.time[event_ids], 2)

        return {
            'num': np.array(self.num[event_ids], copy=True),
            'cat': np.array(self.cat[event_ids], copy=True),
            'tree': self.node_tree(nodes, cutoffs, depth, fanout),
        }