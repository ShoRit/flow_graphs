from collections import Counter, defaultdict
from datetime import time
from glob import glob
import json

from helper import punct_dict


def conv_rel_map(rel):
    if rel == "t":
        rel = "targ"
    elif rel == "a":
        rel = "agent"
    elif rel == "o":
        rel = "other-mod"
    elif rel == "d":
        rel = "dest"
    elif rel == "v":
        rel = "v-tm"
    elif rel == "s":
        rel = "targ"
    return rel


def process_lines(entfile, relfile):
    """Process EFGC dataset, which comes in two files, analogous to preprocess_rels"""
    text = ""
    curr_offset = 0
    curr_label, curr_id = "O", "1_1_1"
    ents = []
    ents_dict = {}
    anns, rels = [], []

    for line in open(entfile):
        try:
            proc_num, sent_num, char_num, word, _, label = line.strip().split(" ")
        except Exception as e:
            import pdb

            pdb.set_trace()

        curr_offset = len(text)
        if f"{proc_num}_{sent_num}_{char_num}" != curr_id:
            curr_id = f"{proc_num}_{sent_num}_{char_num}"
        if word not in punct_dict:
            text = text + word + " "
        else:
            text = text[:-1] + word + " "
            curr_offset -= 1
        ents.append((curr_offset, word, label, curr_id))

    prev_start, prev_end, prev_label, prev_id = None, None, None, None
    for i in range(0, len(ents)):
        curr_start, curr_word, curr_label, curr_id = (
            ents[i][0],
            ents[i][1],
            ents[i][2],
            ents[i][3],
        )
        if curr_label[-1] == "B" or curr_label == "O":
            if prev_label is not None and prev_label != "O":
                ents_dict[prev_id] = (
                    prev_start,
                    prev_end,
                    prev_label[:-2],
                    text[prev_start:prev_end],
                )
                anns.append(
                    {
                        "_id": prev_id,
                        "start": prev_start,
                        "end": prev_end,
                        "word": text[prev_start:prev_end],
                        "label": prev_label[:-2],
                    }
                )

            prev_start, prev_end, prev_label, prev_id = (
                curr_start,
                len(curr_word) + curr_start,
                curr_label,
                curr_id,
            )
        elif curr_label == prev_label[0:-1] + "I":
            prev_end = len(curr_word) + curr_start

    if prev_label[:1] != "O":
        ents_dict[prev_id] = (
            prev_start,
            prev_end,
            prev_label[:-2],
            text[prev_start:prev_end],
        )
        anns.append(
            {
                "_id": prev_id,
                "start": prev_start,
                "end": prev_end,
                "word": text[prev_start:prev_end],
                "label": prev_label[:-2],
            }
        )

    for line in open(relfile, encoding="unicode_escape"):
        if line.startswith("#"):
            continue
        info = line.strip().split()
        if len(info) == 7:
            p1, s1, c1, rel, p2, s2, c2 = info
        else:
            continue
        if rel == "-":
            continue
        arg1_idx, arg2_idx = f"{p1}_{s1}_{c1}", f"{p2}_{s2}_{c2}"
        rels.append(
            {
                "_id": f"{arg1_idx}-{arg2_idx}",
                "arg1_start": ents_dict[arg1_idx][0],
                "arg1_end": ents_dict[arg1_idx][1],
                "arg1_word": ents_dict[arg1_idx][3],
                "arg1_label": ents_dict[arg1_idx][2],
                "arg2_start": ents_dict[arg2_idx][0],
                "arg2_end": ents_dict[arg2_idx][1],
                "arg2_word": ents_dict[arg2_idx][3],
                "arg2_label": ents_dict[arg2_idx][2],
                "arg_label": conv_rel_map(rel),
            }
        )

    return anns, rels, text


def create_japflow():
    """Standardize the EFCG dataset format"""
    data_dir = "/data/flow_graphs/COOKING/FlowGraph/all"
    data_dict = {"all": []}
    ent_lbls, rel_lbls = [], []
    count = 0

    for entfile in glob(f"{data_dir}/*.list"):
        relfile = entfile[:-5] + ".flow"
        doc_id = entfile.split("/")[-1][:-5]
        try:
            anns, rels, text = process_lines(entfile, relfile)
        except Exception as e:
            print(e)
            import pdb

            pdb.set_trace()
            print(relfile)

        data_dict["all"].append({"_id": doc_id, "text": text, "anns": anns, "rels": rels})
        for elem in anns:
            ent_lbls.append(elem["label"])
        for elem in rels:
            rel_lbls.append(elem["arg_label"])
        count += 1
        if count % 100 == 0:
            print(
                "Completed {}, {}".format(
                    count, time.strftime("%d_%m_%Y") + "_" + time.strftime("%H:%M:%S")
                )
            )

    out_dir = f"/projects/flow_graphs/data/japflow/"
    for split in ["all"]:
        with open(f"{out_dir}/{split}.json", "w") as json_file:
            json.dump(data_dict[split], json_file, indent=4)

    json.dump(Counter(ent_lbls), open(f"{out_dir}/ent_lbl.json", "w"), indent=4)
    json.dump(Counter(rel_lbls), open(f"{out_dir}/rel_lbl.json", "w"), indent=4)
