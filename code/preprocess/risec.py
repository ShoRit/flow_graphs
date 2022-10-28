from collections import Counter, defaultdict
from datetime import time
from glob import glob
import json

SUBSTRING_TO_LABEL = {
    "_PPT": "Arg_PPT",
    "_GOL": "Arg_GOL",
    "_DIR": "Arg_DIR",
    "_PRD": "Arg_PRD",
    "_PAG": "Arg_PAG",
    "_MNR": "ArgM_MNR",
    "_PRP": "ArgM_PRP",
    "_LOC": "ArgM_LOC",
    "_TMP": "ArgM_TMP",
    "_MEANS": "ArgM_INT",
    "Simultaneous": "ArgM_SIM",
}


def conv_rel_map(rel):
    for substring, label in SUBSTRING_TO_LABEL.items():
        if substring in rel:
            return label
    return None


def processes_rels(ann_fname):
    """Preprocess datasets into standard json format"""
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
            arg_lbl = conv_rel_map(arg_lbl)

            if arg_lbl is None:
                continue

            arg1 = arg1.split(":")[-1]
            arg2 = arg2.split(":")[-1]
            arg1_idx = ann_dict[arg1]
            arg2_idx = ann_dict[arg2]

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


def standardize_risec():
    """Standardize the format of RISeC dataset"""
    data_dir = "/data/flow_graphs/COOKING/RISEC/data/"
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
                # text 			 = preprocess_texts(text)
            except Exception as e:
                print(e)
                continue

            anns, rels = processes_rels(ann_fname)

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

    out_dir = f"/projects/flow_graphs/data/risec/"
    for split in ["train", "dev", "test"]:
        with open(f"{out_dir}/{split}.json", "w") as json_file:
            json.dump(data_dict[split], json_file, indent=4)

    json.dump(Counter(ent_lbls), open(f"{out_dir}/ent_lbl.json", "w"), indent=4)
    json.dump(Counter(rel_lbls), open(f"{out_dir}/rel_lbl.json", "w"), indent=4)
