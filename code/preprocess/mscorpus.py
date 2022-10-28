from collections import Counter, defaultdict
import time
import json
import pprint


def normalize_relation(rel):
    if rel in [
        "Property_Of",
        "Amount_Of",
        "Brand_Of",
        "Descriptor_Of",
        "Apparatus_Attr_Of",
    ]:
        rel = "Information_Of"
    elif rel == "Atmospheric_Material":
        rel = "Participant_Material"

    return rel


def process_mscorpus_rels(ann_fname):
    """process_rels for mscorpus"""
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
        if line.startswith("E"):
            elems = line.strip().split()
            arg1_lbl, arg1 = elems[1].split(":")
            arg1_idx = ann_dict[arg1]
            ann_dict[elems[0]] = ann_dict[arg1]

            for idx in range(2, len(elems)):
                arg2_lbl, arg2 = elems[idx].split(":")
                arg2_idx = ann_dict[arg2]

                rel_data.append(
                    {
                        "_id": f"{elems[0]}-{idx}",
                        "arg1_start": ann_data[arg1_idx]["start"],
                        "arg1_end": ann_data[arg1_idx]["end"],
                        "arg1_word": ann_data[arg1_idx]["word"],
                        "arg1_label": ann_data[arg1_idx]["label"],
                        "arg2_start": ann_data[arg2_idx]["start"],
                        "arg2_end": ann_data[arg2_idx]["end"],
                        "arg2_word": ann_data[arg2_idx]["word"],
                        "arg2_label": ann_data[arg2_idx]["label"],
                        "arg_label": normalize_relation(arg2_lbl),
                    }
                )

    for line in open(ann_fname):
        if line.startswith("R"):
            index, arg_lbl, arg1, arg2 = line.strip().split()
            # arg_lbl 					= conv_rel_map(arg_lbl)

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
                    "arg_label": normalize_relation(arg_lbl),
                }
            )

    return ann_data, rel_data


def standardize_mscorpus():
    """Standardize MSCorpus format"""
    data_dir = "/data/flow_graphs/MSPT/"
    data_dict = defaultdict(list)
    split_dict = defaultdict(list)
    ent_names, rel_names = defaultdict(lambda: defaultdict(int)), defaultdict(
        lambda: defaultdict(int)
    )
    ent_lbls, rel_lbls = [], []
    count = 0

    for split in ["train", "dev", "test"]:
        split_file = open(f"{data_dir}/sfex-{split}-fnames.txt")
        for line in split_file.readlines():
            split_dict[split].append(line.strip())

    for split in split_dict:
        for file in split_dict[split]:
            ann_fname = f"{data_dir}/data/{file}.ann"
            txt_fname = ann_fname.replace(".ann", ".txt")
            doc_id = ann_fname.split("/")[-1][:-4]
            try:
                text = open(txt_fname).read()
            except Exception as e:
                continue

            anns, rels = process_mscorpus_rels(ann_fname)

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

    out_dir = f"/projects/flow_graphs/data/mscorpus/"
    for split in ["train", "dev", "test"]:
        with open(f"{out_dir}/{split}.json", "w") as json_file:
            json.dump(data_dict[split], json_file, indent=4)

    pprint(Counter(ent_lbls))
    pprint(Counter(rel_lbls))

    json.dump(Counter(ent_lbls), open(f"{out_dir}/ent_lbl.json", "w"), indent=4)
    json.dump(Counter(rel_lbls), open(f"{out_dir}/rel_lbl.json", "w"), indent=4)
