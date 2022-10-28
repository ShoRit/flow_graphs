from collections import Counter, defaultdict
from datetime import time
from glob import glob
import json
import pprint


def get_chemu_rel(arg1_lbl, arg2_lbl):
    """Narrow the scope of ChEMU relations."""
    assert arg1_lbl in ["REACTION_STEP", "WORKUP"]

    if arg2_lbl in ["TIME", "TEMPERATURE", "REAGENT_CATALYST"]:
        arg2_lbl = "RXN_CONDITION"

    if arg2_lbl in ["YIELD_PERCENT", "YIELD_OTHER"]:
        arg2_lbl = "AMOUNT_OF"

    return arg2_lbl


def process_chemu_rels(ann_fname):
    """process_rels for ChEMU"""
    ann_data, rel_data = [], []
    ann_dict = {}
    ann_arr_idx = 0

    for line in open(ann_fname):
        if line.startswith("T"):
            index, info, word = line.strip().split("\t")
            label, start, end = info.split()
            start, end = int(start), int(end)

            if start == end or word.strip() == "":
                continue

            ann_data.append(
                {
                    "_id": index,
                    "start": start,
                    "end": end,
                    "word": word.strip(),
                    "label": label,
                }
            )
            ann_dict[index] = ann_arr_idx
            ann_arr_idx += 1

    for line in open(ann_fname):
        if line.startswith("R"):
            index, arg_lbl, arg1, arg2 = line.strip().split()
            if arg_lbl is None:
                continue

            arg1 = arg1.split(":")[-1]
            arg2 = arg2.split(":")[-1]
            arg1_idx = ann_dict[arg1]
            arg2_idx = ann_dict[arg2]

            arg1_lbl, arg2_lbl = (
                ann_data[arg1_idx]["label"],
                ann_data[arg2_idx]["label"],
            )
            arg_lbl = get_chemu_rel(arg1_lbl, arg2_lbl)

            rel_data.append(
                {
                    "_id": index,
                    "arg1_start": ann_data[arg1_idx]["start"],
                    "arg1_end": ann_data[arg1_idx]["end"],
                    "arg1_word": ann_data[arg1_idx]["word"],
                    "arg1_label": ann_data[arg1_idx]["label"],
                    "arg2_start": ann_data[arg2_idx]["start"],
                    "arg2_end": ann_data[arg2_idx]["end"],
                    "arg2_word": ann_data[arg2_idx]["word"],
                    "arg2_label": ann_data[arg2_idx]["label"],
                    "arg_label": arg_lbl,
                }
            )

    return ann_data, rel_data


def create_chemu():
    """Standardize format for ChEMU"""
    data_dir = "/data/flow_graphs/chemu/"
    data_dict = defaultdict(list)
    ent_lbls, rel_lbls = [], []
    count = 0

    for split in ["train", "dev", "test"]:
        ann_files = glob(f"{data_dir}/{split}/*.ann")
        for ann_fname in ann_files:
            ann_file = open(ann_fname)
            txt_fname = ann_fname.replace(".ann", ".txt")
            doc_id = ann_fname.split("/")[-1][:-4]
            try:
                text = open(txt_fname).read()
            except Exception as e:
                continue

            anns, rels = process_chemu_rels(ann_fname)

            data_dict[split].append(
                {
                    "_id": doc_id,
                    "text": text,
                    "anns": anns,
                    "rels": rels,
                }
            )

            for elem in anns:
                ent_lbls.append(elem["label"])
            for elem in rels:
                rel_lbls.append(elem["arg_label"])

            count += 1
            if count % 100 == 0:
                print(
                    "Completed {}, {}".format(
                        count,
                        time.strftime("%d_%m_%Y") + "_" + time.strftime("%H:%M:%S"),
                    )
                )

    out_dir = f"/projects/flow_graphs/data/chemu/"
    for split in ["train", "dev", "test"]:
        with open(f"{out_dir}/{split}.json", "w") as json_file:
            json.dump(data_dict[split], json_file, indent=4)

    pprint(Counter(ent_lbls))
    pprint(Counter(rel_lbls))

    json.dump(Counter(ent_lbls), open(f"{out_dir}/ent_lbl.json", "w"), indent=4)
    json.dump(Counter(rel_lbls), open(f"{out_dir}/rel_lbl.json", "w"), indent=4)
