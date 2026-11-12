#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from _common import ROOT, json_write, resolve_dataset_path
from cadvmh.baselines.neural import SparseRationalizer
from cadvmh.data.loader import load_dataset
from cadvmh.data.tokenization import special_token_mask
from cadvmh.evaluation.faithfulness import masked_rationale_text
from cadvmh.models import CADVMH
from cadvmh.utils.config import load_config


class RationaleDataset(Dataset):
    def __init__(self, frame, label_to_id):
        self.frame = frame.reset_index(drop=True)
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return {
            "text": str(row.text),
            "label": self.label_to_id[row.label],
            "sample_id": row.sample_id,
        }


def rationale_collator(tokenizer, max_length):
    def collate(rows):
        encoded = tokenizer(
            [row["text"] for row in rows],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        encoded["offsets"] = encoded.pop("offset_mapping")
        encoded["special_token_mask"] = special_token_mask(
            encoded["input_ids"], tokenizer, encoded["attention_mask"]
        )
        encoded["labels"] = torch.tensor([row["label"] for row in rows])
        encoded["texts"] = [row["text"] for row in rows]
        encoded["sample_ids"] = [row["sample_id"] for row in rows]
        return encoded

    return collate


def probability(logits, class_ids):
    return torch.softmax(logits, -1).gather(1, class_ids[:, None]).squeeze(1)


def record_from_output(output, batch, labels, tokenizer):
    final_probabilities = torch.softmax(output.final_logits, -1)
    predictions = final_probabilities.argmax(-1)
    gold = batch["labels"]
    values = {
        "gold_original": probability(output.original_logits, gold),
        "gold_evidence": probability(output.evidence_logits, gold),
        "gold_counterfactual": probability(output.counterfactual_logits, gold),
        "predicted_original": probability(output.original_logits, predictions),
        "predicted_evidence": probability(output.evidence_logits, predictions),
        "predicted_counterfactual": probability(
            output.counterfactual_logits, predictions
        ),
    }
    records = []
    for row in range(gold.size(0)):
        eligible = output.valid_token_mask[row].bool()
        selection = output.selection_mask[row].bool() & eligible
        offsets = batch["offsets"][row].tolist()
        token_records = []
        for index in torch.where(eligible.cpu())[0].tolist():
            left, right = offsets[index]
            if right <= left:
                continue
            token_records.append(
                {
                    "token": tokenizer.convert_ids_to_tokens(
                        int(batch["input_ids"][row, index])
                    ),
                    "text": batch["texts"][row][left:right],
                    "start": int(left),
                    "end": int(right),
                    "probability": float(
                        output.selector_probabilities[row, index].cpu()
                    ),
                    "selected": bool(selection[index].cpu()),
                }
            )
        evidence_text = masked_rationale_text(
            batch["texts"][row], token_records, keep_selected=True
        )
        counterfactual_text = masked_rationale_text(
            batch["texts"][row], token_records, keep_selected=False
        )
        records.append(
            {
                "sample_id": batch["sample_ids"][row],
                "text": batch["texts"][row],
                "true_class": labels[int(gold[row])],
                "predicted_class": labels[int(predictions[row])],
                "correct": bool(gold[row] == predictions[row]),
                "prediction_confidence": float(
                    final_probabilities[row, predictions[row]].cpu()
                ),
                "selected_rationale": " ".join(
                    token["text"] for token in token_records if token["selected"]
                ),
                "evidence_only_text": evidence_text,
                "counterfactual_text": counterfactual_text,
                "selection_ratio": float(
                    selection.sum().cpu() / eligible.sum().clamp_min(1).cpu()
                ),
                "gold_probability_original": float(values["gold_original"][row].cpu()),
                "gold_probability_evidence": float(values["gold_evidence"][row].cpu()),
                "gold_probability_counterfactual": float(
                    values["gold_counterfactual"][row].cpu()
                ),
                "gold_comprehensiveness": float(
                    (
                        values["gold_original"][row]
                        - values["gold_counterfactual"][row]
                    ).cpu()
                ),
                "gold_sufficiency": float(
                    (values["gold_original"][row] - values["gold_evidence"][row]).cpu()
                ),
                "predicted_probability_original": float(
                    values["predicted_original"][row].cpu()
                ),
                "predicted_probability_evidence": float(
                    values["predicted_evidence"][row].cpu()
                ),
                "predicted_probability_counterfactual": float(
                    values["predicted_counterfactual"][row].cpu()
                ),
                "predicted_comprehensiveness": float(
                    (
                        values["predicted_original"][row]
                        - values["predicted_counterfactual"][row]
                    ).cpu()
                ),
                "predicted_sufficiency": float(
                    (
                        values["predicted_original"][row]
                        - values["predicted_evidence"][row]
                    ).cpu()
                ),
                "tokens": token_records,
            }
        )
    return records


def choose_examples(records, count):
    selected, used = [], set()

    def take(candidates, reason, amount=1):
        for record in candidates:
            if len(selected) >= count or amount <= 0:
                break
            if record["sample_id"] not in used:
                selected.append(record | {"selection_reason": reason})
                used.add(record["sample_id"])
                amount -= 1

    correct = [row for row in records if row["correct"]]
    failures = [row for row in records if not row["correct"]]
    take(
        sorted(correct, key=lambda row: row["prediction_confidence"], reverse=True),
        "correct high-confidence",
        2,
    )
    take(
        sorted(correct, key=lambda row: row["prediction_confidence"]),
        "subtle or ambiguous",
        2,
    )
    take(
        [
            row
            for row in records
            if re.search(
                r"\b(no|not|never|without|isn't|don't|can't|cannot|won't)\b",
                row["text"],
                re.I,
            )
        ],
        "negation",
        1,
    )
    take(
        [
            row
            for row in records
            if re.search(
                r"\b(feels? like|as if|drowning|darkness|storm|spiral|weight|war|battle)\b",
                row["text"],
                re.I,
            )
        ],
        "figurative-language candidate",
        1,
    )
    take(
        sorted(failures, key=lambda row: row["prediction_confidence"], reverse=True),
        "failure case",
        1,
    )
    take(
        sorted(
            records,
            key=lambda row: abs(row["predicted_comprehensiveness"]),
            reverse=True,
        ),
        "strong rationale effect",
        count,
    )
    return selected[:count]


def highlighted_html(record):
    pieces, cursor = [], 0
    for token in record["tokens"]:
        pieces.append(html.escape(record["text"][cursor : token["start"]]))
        color = f"rgba(220,55,45,{0.10 + 0.80 * token['probability']:.3f})"
        outline = (
            "font-weight:700;outline:2px solid #7f1d1d;" if token["selected"] else ""
        )
        pieces.append(
            f'<span title="p={token["probability"]:.3f}" style="background:{color};{outline}">{html.escape(record["text"][token["start"] : token["end"]])}</span>'
        )
        cursor = token["end"]
    pieces.append(html.escape(record["text"][cursor:]))
    return "".join(pieces)


def write_qualitative_html(examples, destination):
    rows = [
        f"<tr><td>{html.escape(row['selection_reason'])}</td><td>{html.escape(row['true_class'])}</td><td>{html.escape(row['predicted_class'])}<br><small>{row['prediction_confidence']:.3f}</small></td><td>{highlighted_html(row)}</td></tr>"
        for row in examples
    ]
    document = (
        """<!doctype html><meta charset="utf-8"><title>CADV-MH rationale examples</title>
<style>body{font:15px/1.55 sans-serif;margin:2rem;color:#172033}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd2dc;padding:.7rem;vertical-align:top}th{background:#eef2f7}small{color:#596579}</style>
<h1>CADV-MH qualitative rationales</h1><p>Color intensity is selector probability; outlined bold tokens are deterministic top-rho selections. Examples are heuristic candidates and require human review before publication.</p>
<table><thead><tr><th>Case</th><th>True class</th><th>Prediction</th><th>Original text with selected tokens highlighted</th></tr></thead><tbody>"""
        + "".join(rows)
        + "</tbody></table>"
    )
    destination.write_text(document, encoding="utf-8")


def save_figure(fig, output, name):
    fig.savefig(
        output / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white"
    )
    fig.savefig(
        output / f"{name}.pdf", bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def _clean_display_text(value):
    """Remove C1 controls caused by legacy text encodings from rendered figures."""
    value = str(value)
    if any(marker in value for marker in ("â", "Ã", "Â")):
        try:
            value = value.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return re.sub(r"[\x80-\x9f]", " ", value).replace("\n", " ").strip()


def _word_spans(record, max_characters=220):
    text = _clean_display_text(record["text"])
    if len(text) > max_characters:
        text = text[:max_characters].rsplit(" ", 1)[0] + " …"
    spans = []
    for match in re.finditer(r"\S+", text):
        overlapping = [
            token
            for token in record["tokens"]
            if token["end"] > match.start() and token["start"] < match.end()
        ]
        spans.append(
            (
                match.group(),
                max((token["probability"] for token in overlapping), default=0.0),
                any(token["selected"] for token in overlapping),
            )
        )
    return spans


def _packed_highlighted_text(record, line_width=40):
    lines, line, characters = [], [], 0
    for word, score, selected in _word_spans(record):
        if line and characters + len(word) + 1 > line_width:
            lines.append(line)
            line, characters = [], 0
        style = {
            "fontsize": 7.6,
            "color": "#172033",
            "bbox": {
                "facecolor": (
                    0.73,
                    0.11,
                    0.11,
                    (0.18 + 0.42 * score) if selected else (0.025 + 0.22 * score),
                ),
                "edgecolor": "#991b1b" if selected else "none",
                "linewidth": 0.75,
                "boxstyle": "round,pad=0.08",
            },
        }
        if selected:
            style["fontweight"] = "bold"
        line.append(TextArea(word + " ", textprops=style))
        characters += len(word) + 1
    if line:
        lines.append(line)
    return VPacker(
        children=[HPacker(children=items, align="baseline", pad=0, sep=1) for items in lines],
        align="left",
        pad=0,
        sep=3,
    )


def qualitative_figure(examples, output):
    columns = 2
    rows = int(np.ceil(len(examples) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(7.2, 2.05 * rows))
    axes = np.atleast_1d(axes).ravel()
    for index, (axis, record) in enumerate(zip(axes, examples)):
        axis.set(xticks=[], yticks=[], xlim=(0, 1), ylim=(0, 1))
        for spine in axis.spines.values():
            spine.set_color("#d4d9e2")
            spine.set_linewidth(0.8)
        status = "correct" if record["correct"] else "error"
        axis.text(
            0.025,
            0.95,
            f"{chr(97 + index)})  {record['selection_reason'].capitalize()}",
            weight="semibold",
            fontsize=8.2,
            va="top",
            transform=axis.transAxes,
        )
        axis.text(
            0.025,
            0.80,
            f"True: {record['true_class']}   Pred.: {record['predicted_class']} "
            f"({record['prediction_confidence']:.2f})   [{status}]",
            fontsize=7.5,
            color="#475569",
            va="top",
            transform=axis.transAxes,
        )
        packed = _packed_highlighted_text(record)
        axis.add_artist(
            AnchoredOffsetbox(
                loc="upper left",
                child=packed,
                pad=0,
                frameon=False,
                bbox_to_anchor=(0.02, 0.70),
                bbox_transform=axis.transAxes,
                borderpad=0,
            )
        )
    for axis in axes[len(examples) :]:
        axis.axis("off")
    fig.suptitle(
        "Qualitative rationale examples",
        fontsize=11,
        fontweight="semibold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.006,
        "Darker shading denotes higher selector probability; outlined bold tokens form the deterministic top-ρ rationale.",
        ha="center",
        fontsize=7.2,
        color="#475569",
    )
    fig.subplots_adjust(top=0.94, bottom=0.05, hspace=0.14, wspace=0.08)
    save_figure(fig, output, "figure1_qualitative_rationales")


def probability_figure(examples, output):
    x = np.arange(len(examples))
    width = 0.25
    fig, axis = plt.subplots(figsize=(7.2, 3.45))
    colors = ["#334e68", "#2a9d8f", "#d95d39"]
    axis.bar(
        x - width,
        [row["gold_probability_original"] for row in examples],
        width,
        label="Original",
        color=colors[0],
    )
    axis.bar(
        x,
        [row["gold_probability_evidence"] for row in examples],
        width,
        label="Evidence-only",
        color=colors[1],
    )
    axis.bar(
        x + width,
        [row["gold_probability_counterfactual"] for row in examples],
        width,
        label="Counterfactual",
        color=colors[2],
    )
    axis.set(
        xticks=x,
        xticklabels=[f"E{i + 1}" for i in range(len(examples))],
        ylabel="Gold-class probability",
        ylim=(0, 1),
        xlabel="Qualitative example",
    )
    fig.suptitle(
        "Effect of rationale interventions", fontsize=11, fontweight="semibold", y=0.98
    )
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        fontsize=8,
    )
    axis.grid(axis="y", color="#d9dee7", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(top=0.81, bottom=0.17, left=0.10, right=0.98)
    save_figure(fig, output, "figure2_prediction_views")


def aggregate(records):
    if not records:
        return {
            "comprehensiveness": float("nan"),
            "sufficiency": float("nan"),
            "selection_ratio": float("nan"),
        }
    return {
        "comprehensiveness": float(
            np.mean([row["gold_comprehensiveness"] for row in records])
        ),
        "sufficiency": float(np.mean([row["gold_sufficiency"] for row in records])),
        "selection_ratio": float(np.mean([row["selection_ratio"] for row in records])),
    }


def faithfulness_figure(methods, output, target_ratio):
    metrics = [
        ("comprehensiveness", "Comprehensiveness ↑"),
        ("sufficiency", "Sufficiency ↓"),
        ("selection_ratio", "Selection ratio"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    palette = ["#334e68", "#2a9d8f", "#d95d39"]
    for panel, (axis, (key, title)) in enumerate(zip(axes, metrics)):
        names = list(methods)
        display_names = [
            "Integrated\ngradients" if name == "Integrated gradients" else name
            for name in names
        ]
        values = [methods[name][key] for name in names]
        bars = axis.bar(
            display_names,
            values,
            color=palette[: len(names)],
            width=0.68,
        )
        axis.set_title(f"{chr(97 + panel)}) {title}", loc="left", fontsize=9, fontweight="semibold")
        axis.grid(axis="y", color="#d9dee7", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(axis="x", rotation=0, labelsize=6.8)
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=2, fontsize=7)
        lower = min(0, min(values))
        upper = max(values + ([target_ratio] if key == "selection_ratio" else []))
        margin = max(0.03, (upper - lower) * 0.22)
        axis.set_ylim(lower - (0.02 if lower < 0 else 0), upper + margin)
        if key == "selection_ratio":
            axis.axhline(
                target_ratio,
                color="#b91c1c",
                linestyle="--",
                label=f"target ρ = {target_ratio:.2f}",
            )
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Faithfulness–sparsity comparison", fontsize=11, fontweight="semibold", y=0.98
    )
    fig.subplots_adjust(top=0.76, bottom=0.22, left=0.07, right=0.98, wspace=0.47)
    save_figure(fig, output, "figure3_faithfulness_sparsity")


def classwise_figure(records, labels, output):
    grouped = {
        label: aggregate([row for row in records if row["true_class"] == label])
        for label in labels
    }
    metrics = [
        ("comprehensiveness", "Comprehensiveness ↑", "#334e68"),
        ("sufficiency", "Sufficiency ↓", "#d95d39"),
        ("selection_ratio", "Selection ratio", "#2a9d8f"),
    ]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharex=True)
    for panel, (axis, (key, title, color)) in enumerate(zip(axes, metrics)):
        values = [grouped[label][key] for label in labels]
        bars = axis.bar(x, values, color=color, width=0.68)
        axis.axhline(0, color="#334155", linewidth=0.7)
        axis.set_title(f"{chr(97 + panel)}) {title}", loc="left", fontsize=9, fontweight="semibold")
        axis.set_xticks(x, labels, rotation=28, ha="right", fontsize=7.2)
        axis.grid(axis="y", color="#d9dee7", linewidth=0.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=6.6)
        lower, upper = min(0, min(values)), max(values)
        margin = max(0.035, (upper - lower) * 0.20)
        axis.set_ylim(lower - (0.03 if lower < 0 else 0), upper + margin)
        if key == "selection_ratio":
            axis.axhline(0.20, color="#991b1b", linestyle="--", linewidth=1, label="target ρ = 0.20")
            axis.legend(frameon=False, fontsize=7)
    fig.suptitle("Class-wise rationale behavior", fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, output, "figure4_classwise_rationales")
    return grouped


def explanation_baseline_aggregates(dataset):
    root = ROOT / "experiments" / dataset / "explanation_baselines"
    names = {
        "attention": "Attention",
        "integrated_gradients": "Integrated gradients",
    }
    methods = {}
    for method, label in names.items():
        path = root / method / "seed_42" / "metrics.json"
        if not path.exists():
            continue
        values = json.loads(path.read_text(encoding="utf-8"))
        methods[label] = {
            "comprehensiveness": values["comprehensiveness_mean"],
            "sufficiency": values["sufficiency_mean"],
            "selection_ratio": values["selection_ratio_mean"],
        }
    return methods


def sparse_aggregate(checkpoint, vocabulary_path, frame, labels, max_length, device):
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    model = SparseRationalizer(len(vocabulary) + 2, 200, 128, len(labels)).to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    label_to_id = {label: index for index, label in enumerate(labels)}
    values = {"comprehensiveness": [], "sufficiency": [], "selection_ratio": []}
    with torch.no_grad():
        for start in range(0, len(frame), 32):
            batch = frame.iloc[start : start + 32]
            tokenized = [
                [
                    vocabulary.get(token, 1)
                    for token in str(text).lower().split()[:max_length]
                ]
                or [1]
                for text in batch.text
            ]
            length = max(map(len, tokenized))
            ids = torch.zeros(len(tokenized), length, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids, dtype=torch.float32)
            for index, tokens in enumerate(tokenized):
                ids[index, : len(tokens)] = torch.tensor(tokens, device=device)
                mask[index, : len(tokens)] = 1
            output = model.rationale_views(ids, mask)
            classes = torch.tensor(
                [label_to_id[label] for label in batch.label], device=device
            )
            original = probability(output["original_logits"], classes)
            evidence = probability(output["evidence_logits"], classes)
            counter = probability(output["counterfactual_logits"], classes)
            values["comprehensiveness"].extend((original - counter).cpu().tolist())
            values["sufficiency"].extend((original - evidence).cpu().tolist())
            values["selection_ratio"].extend(
                (output["selection"].sum(1) / mask.sum(1).clamp_min(1)).cpu().tolist()
            )
    return {key: float(np.mean(value)) for key, value in values.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Generate CADV-MH qualitative and quantitative rationale analysis."
    )
    parser.add_argument("--dataset", choices=["dataset_a", "dataset_b"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--examples", type=int, default=8, choices=range(6, 9))
    parser.add_argument(
        "--max-examples",
        type=int,
        help="Bounded validation run; omit for the complete test set",
    )
    parser.add_argument("--sparse-checkpoint")
    args = parser.parse_args()
    checkpoint = (
        Path(args.checkpoint)
        if args.checkpoint
        else ROOT / "experiments" / args.dataset / "cadvmh" / "seed_42" / "best.pt"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "experiments" / args.dataset / "interpretability"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = payload["config"]
    if payload.get("config", {}).get("project", {}).get("seed") != 42:
        raise ValueError("Interpretability analysis requires a seed-42 checkpoint")
    config = load_config(ROOT / "configs" / f"{args.dataset}.yaml")
    config["dataset"]["path"] = str(resolve_dataset_path(config))
    frame, _ = load_dataset(config["dataset"], config["data"].get("encoding", "utf-8"))
    manifest_path = ROOT / "prepared" / args.dataset / "primary_split.csv"
    if (
        payload.get("split_fingerprint")
        != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    ):
        raise ValueError("Checkpoint and immutable split manifest do not match")
    manifest = pd.read_csv(manifest_path)
    frame = frame.merge(
        manifest[["sample_id", "partition"]], on="sample_id", validate="one_to_one"
    )
    test = frame.loc[frame.partition == "test"].reset_index(drop=True)
    if args.max_examples:
        test = test.head(args.max_examples)
    labels = config["dataset"]["labels"]
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_config["model"]["encoder_name"], use_fast=True
    )
    model = CADVMH(
        len(labels),
        checkpoint_config["model"]["encoder_name"],
        selector_hidden_dim=checkpoint_config["model"]["selector_hidden_dim"],
        interaction_dim=checkpoint_config["model"]["interaction_dim"],
        dropout=checkpoint_config["model"]["selector_dropout"],
    )
    model.load_state_dict(payload["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    loader = DataLoader(
        RationaleDataset(test, {label: index for index, label in enumerate(labels)}),
        batch_size=config["training"]["microbatch_size"],
        shuffle=False,
        collate_fn=rationale_collator(tokenizer, config["model"]["max_length"]),
    )
    records = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                **{
                    key: batch[key].to(device)
                    for key in ("input_ids", "attention_mask", "special_token_mask")
                },
                mask_token_id=tokenizer.mask_token_id,
                stage=int(payload["stage"]),
                selection_mode="top_rho",
                selection_ratio=config["model"]["selection_ratio"],
                enable_evidence=True,
                return_auxiliary=True,
                fusion_mode="full",
            )
            batch["labels"] = batch["labels"].to(device)
            records.extend(record_from_output(output, batch, labels, tokenizer))
    with (output_dir / "rationale_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    examples = choose_examples(records, min(args.examples, len(records)))
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if key != "tokens"}
            for row in examples
        ]
    ).to_csv(output_dir / "qualitative_examples.csv", index=False)
    write_qualitative_html(examples, output_dir / "figure1_qualitative_rationales.html")
    qualitative_figure(examples, output_dir)
    probability_figure(examples, output_dir)
    methods = {"CADV-MH": aggregate(records)}
    methods.update(explanation_baseline_aggregates(args.dataset))
    sparse_checkpoint = (
        Path(args.sparse_checkpoint)
        if args.sparse_checkpoint
        else ROOT
        / "experiments"
        / args.dataset
        / "extended_baselines"
        / "sparse_rationalizer"
        / "seed_42"
        / "model.pt"
    )
    if sparse_checkpoint.exists():
        methods["Sparse rationalizer"] = sparse_aggregate(
            sparse_checkpoint,
            sparse_checkpoint.with_name("vocabulary.json"),
            test,
            labels,
            config["model"]["max_length"],
            device,
        )
    faithfulness_figure(methods, output_dir, config["model"]["selection_ratio"])
    grouped = classwise_figure(records, labels, output_dir)
    json_write(
        output_dir / "interpretability_summary.json",
        {
            "dataset": args.dataset,
            "checkpoint": str(checkpoint),
            "examples_evaluated": len(records),
            "selection_mode": "top_rho",
            "target_selection_ratio": config["model"]["selection_ratio"],
            "methods": methods,
            "classwise": grouped,
            "qualitative_examples": [row["sample_id"] for row in examples],
            "sparse_comparison_available": "Sparse rationalizer" in methods,
        },
    )
    print(
        f"Interpretability report created from {len(records):,} examples: {output_dir}"
    )
    if "Sparse rationalizer" not in methods:
        print(
            "Sparse-rationalizer checkpoint not found; Figure 3 uses the available "
            "attention/gradient explanation baselines instead."
        )


if __name__ == "__main__":
    main()
