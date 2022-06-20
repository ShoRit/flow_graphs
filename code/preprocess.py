from helper import *
from intervals import *
import spacy
from torch.nn.utils.rnn import pad_sequence
import stanza


def generate_reldesc():
    from sentence_transformers import SentenceTransformer

    rel2desc_emb = {}
    rel2desc = json.load(open(f"../data/rel2desc.json"))
    encoder = SentenceTransformer("bert-base-nli-mean-tokens")
    rel2id, id2rel = {}, {}

    for i, rel in enumerate(rel2desc):
        rel2id[rel] = i
        id2rel[i] = rel
        rel2desc_emb[rel] = encoder.encode(rel2desc[rel])

    return rel2desc, rel2id, id2rel, rel2desc_emb


def conv_rel_map(rel):
    if args.dataset == "risec":
        if "_PPT" in rel:
            rel = "Arg_PPT"
        elif "_GOL" in rel:
            rel = "Arg_GOL"
        elif "_DIR" in rel:
            rel = "Arg_DIR"
        elif "_PRD" in rel:
            rel = "Arg_PRD"
        elif "_PAG" in rel:
            rel = "Arg_PAG"
        elif "_MNR" in rel:
            rel = "ArgM_MNR"
        elif "_PRP" in rel:
            rel = "ArgM_PRP"
        elif "_LOC" in rel:
            rel = "ArgM_LOC"
        elif "_TMP" in rel:
            rel = "ArgM_TMP"
        elif "_MEANS" in rel:
            rel = "ArgM_INT"
        elif "Simultaneous" in rel:
            rel = "ArgM_SIM"

        else:
            rel = None

    if args.dataset == "japflow":
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

    if args.dataset == "mscorpus":
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


def get_chemu_rel(arg1_lbl, arg2_lbl):

    assert arg1_lbl in ["REACTION_STEP", "WORKUP"]

    if arg2_lbl in ["TIME", "TEMPERATURE", "REAGENT_CATALYST"]:
        arg2_lbl = "RXN_CONDITION"

    if arg2_lbl in ["YIELD_PERCENT", "YIELD_OTHER"]:
        arg2_lbl = "AMOUNT_OF"

    return arg2_lbl


def processes_rels(ann_fname):
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


def preprocess_texts(text):
    # text= text.replace('\n','\u00A0')
    text = text.replace("\n", " ")
    return text


def create_risec():
    data_dir = "/data/flow_graphs/COOKING/RISEC/data/"
    data_dict = ddict(list)
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

    out_dir = f"../data/risec/"
    for split in ["train", "dev", "test"]:
        with open(f"{out_dir}/{split}.json", "w") as json_file:
            json.dump(data_dict[split], json_file, indent=4)

    json.dump(Counter(ent_lbls), open(f"{out_dir}/ent_lbl.json", "w"), indent=4)
    json.dump(Counter(rel_lbls), open(f"{out_dir}/rel_lbl.json", "w"), indent=4)


def process_lines(entfile, relfile):
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

    out_dir = f"../data/japflow/"
    for split in ["all"]:
        with open(f"{out_dir}/{split}.json", "w") as json_file:
            json.dump(data_dict[split], json_file, indent=4)

    json.dump(Counter(ent_lbls), open(f"{out_dir}/ent_lbl.json", "w"), indent=4)
    json.dump(Counter(rel_lbls), open(f"{out_dir}/rel_lbl.json", "w"), indent=4)


def process_mscorpus_rels(ann_fname):
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
                        "arg_label": conv_rel_map(arg2_lbl),
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
                    "arg_label": conv_rel_map(arg_lbl),
                }
            )

    return ann_data, rel_data


def create_mscorpus():
    data_dir = "/data/flow_graphs/MSPT/"
    data_dict = ddict(list)
    split_dict = ddict(list)
    ent_names, rel_names = ddict(lambda: ddict(int)), ddict(lambda: ddict(int))
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

    out_dir = f"../data/mscorpus/"
    for split in ["train", "dev", "test"]:
        with open(f"{out_dir}/{split}.json", "w") as json_file:
            json.dump(data_dict[split], json_file, indent=4)

    pprint(Counter(ent_lbls))
    pprint(Counter(rel_lbls))

    json.dump(Counter(ent_lbls), open(f"{out_dir}/ent_lbl.json", "w"), indent=4)
    json.dump(Counter(rel_lbls), open(f"{out_dir}/rel_lbl.json", "w"), indent=4)


def process_chemu_rels(ann_fname):
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
    data_dir = "/data/flow_graphs/chemu/"
    data_dict = ddict(list)
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

    out_dir = f"../data/chemu/"
    for split in ["train", "dev", "test"]:
        with open(f"{out_dir}/{split}.json", "w") as json_file:
            json.dump(data_dict[split], json_file, indent=4)

    pprint(Counter(ent_lbls))
    pprint(Counter(rel_lbls))

    json.dump(Counter(ent_lbls), open(f"{out_dir}/ent_lbl.json", "w"), indent=4)
    json.dump(Counter(rel_lbls), open(f"{out_dir}/rel_lbl.json", "w"), indent=4)


def get_srl_parses(srls=None):
    if srls is None:
        srls = {
            "verbs": [
                {
                    "verb": "Did",
                    "description": "[do.01: Did] Uriah honestly think he could beat the game in under three hours ?",
                    "tags": [
                        "B-V",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                    ],
                    "frame": "do.01",
                    "frame_score": 0.9999996423721313,
                    "lemma": "do",
                },
                {
                    "verb": "think",
                    "description": "Did [ARG0: Uriah] [ARGM-ADV: honestly] [think.01: think] [ARG1: he could beat the game in under three hours] ?",
                    "tags": [
                        "O",
                        "B-ARG0",
                        "B-ARGM-ADV",
                        "B-V",
                        "B-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "I-ARG1",
                        "O",
                    ],
                    "frame": "think.01",
                    "frame_score": 1.0,
                    "lemma": "think",
                },
                {
                    "verb": "could",
                    "description": "Did Uriah honestly think he [go.04: could] beat the game in under three hours ?",
                    "tags": [
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-V",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                        "O",
                    ],
                    "frame": "go.04",
                    "frame_score": 0.10186540335416794,
                    "lemma": "could",
                },
                {
                    "verb": "beat",
                    "description": "Did Uriah honestly think [ARG0: he] [ARGM-MOD: could] [beat.03: beat] [ARG1: the game] [ARGM-TMP: in under three hours] ?",
                    "tags": [
                        "O",
                        "O",
                        "O",
                        "O",
                        "B-ARG0",
                        "B-ARGM-MOD",
                        "B-V",
                        "B-ARG1",
                        "I-ARG1",
                        "B-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "I-ARGM-TMP",
                        "O",
                    ],
                    "frame": "beat.03",
                    "frame_score": 0.9999936819076538,
                    "lemma": "beat",
                },
            ],
            "words": [
                "Did",
                "Uriah",
                "honestly",
                "think",
                "he",
                "could",
                "beat",
                "the",
                "game",
                "in",
                "under",
                "three",
                "hours",
                "?",
            ],
        }
    words = srls["words"]
    frames = srls["verbs"]
    srl_graph = []

    for frame in frames:
        verb, tags, = (
            frame["verb"],
            frame["tags"],
        )
        try:
            root_start_idx = tags.index("B-V")
        except Exception as e:
            root_start_idx = -1

        if root_start_idx == -1:
            continue

        root_verb = ""
        for idx in range(root_start_idx, len(tags)):
            if tags[idx][2:] == "V":
                root_verb += words[root_start_idx]
                root_end_idx = idx
            else:
                break
        root_end_idx += 1
        curr_tag, dep_word, start_idx = None, "", -1
        for idx, tag in enumerate(tags):
            if curr_tag is not None:
                if tag == "O" or tag[2:] == "V":
                    curr_dict = {
                        "root_verb": root_verb,
                        "word": dep_word,
                        "dep_tag": curr_tag,
                        "root_start_idx": root_start_idx,
                        "root_end_idx": root_end_idx,
                        "start_idx": start_idx,
                        "end_idx": idx,
                    }
                    srl_graph.append(curr_dict)
                    dep_word, curr_tag = "", None
                elif tag[2:] == curr_tag:
                    dep_word += " " + words[idx]
                elif tag[2:] != curr_tag:
                    curr_dict = {
                        "root_verb": root_verb,
                        "word": dep_word,
                        "dep_tag": curr_tag,
                        "root_start_idx": root_start_idx,
                        "root_end_idx": root_end_idx,
                        "start_idx": start_idx,
                        "end_idx": idx,
                    }
                    srl_graph.append(curr_dict)
                    curr_tag, start_idx, dep_word = tag[2:], idx, words[idx]
            else:
                if tag != "O" and tag[2:] != "V":
                    curr_tag, start_idx, dep_word = tag[2:], idx, words[idx]

    verb_start_idxs = list(set([elem["root_start_idx"] for elem in srl_graph]))

    corr_frames = []
    for frame in srls["verbs"]:
        try:
            root_idx = frame["tags"].index("B-V")
        except Exception as e:
            root_idx = -1
        if root_idx not in verb_start_idxs or root_idx == -1:
            corr_frames.append(0)
        else:
            corr_frames.append(1)

    verb_names = []
    verb_end_idxs = []

    for i, frame in enumerate(srls["verbs"]):
        if corr_frames[i] == 1:
            end_idx = -1
            for idx, tag in enumerate(frame["tags"]):
                if tag[2:] == "V":
                    end_idx = idx
            verb_end_idxs.append(end_idx + 1)
            verb_names.append(frame["verb"])

    new_srls = {"ins": [], "del": []}
    for i in range(0, len(verb_names)):
        vn_i, start_i, end_i = verb_names[i], verb_start_idxs[i], verb_end_idxs[i]
        for j in range(0, len(verb_names)):
            if i == j:
                continue
            vn_j, start_j, end_j = verb_names[j], verb_start_idxs[j], verb_end_idxs[j]
            srl_elems = [elem for elem in srl_graph if elem["root_start_idx"] == start_i]
            for elem in srl_elems:
                if elem["start_idx"] <= start_j and elem["end_idx"] > end_j:
                    curr_dict = {
                        "root_verb": vn_i,
                        "word": vn_j,
                        "dep_tag": elem["dep_tag"],
                        "root_start_idx": start_i,
                        "root_end_idx": end_i,
                        "start_idx": start_j,
                        "end_idx": end_j,
                    }
                    new_srls["ins"].append(curr_dict)
                    new_srls["del"].append(elem)

    # print(srl_graph)

    for elem in new_srls["del"]:
        if elem in srl_graph:
            srl_graph.remove(elem)

    for elem in new_srls["ins"]:
        srl_graph.append(elem)

    edges = [
        (
            (elem["root_start_idx"], elem["root_end_idx"]),
            (elem["start_idx"], elem["end_idx"]),
            elem["dep_tag"],
        )
        for elem in srl_graph
    ]
    node_idxs = {}
    for edge in edges:
        n1, n2 = edge[0], edge[1]
        if n1 not in node_idxs:
            node_idxs[n1] = len(node_idxs)
        if n2 not in node_idxs:
            node_idxs[n2] = len(node_idxs)

    nodes = {}
    for nix in node_idxs:
        nodes[node_idxs[nix]] = words[nix[0] : nix[1]]

    edge_index = [[], []]
    for edge in edges:
        edge_index[0].append(node_idxs[edge[0]])
        edge_index[1].append(node_idxs[edge[1]])

    edge_type = [elem[2] for elem in edges]

    return {"nodes": nodes, "edge_index": edge_index, "edge_type": edge_type}


def dump_srls(data_dir, splits, text_tokenizer="scispacy"):
    bad_counts = 0
    # from transformer_srl import dataset_readers, models, predictors
    if text_tokenizer == "scispacy":
        nlp = spacy.load("en_core_sci_md")
    sent_srl_dict = {}
    srl_sents = set()
    for split in splits:
        docs = json.load(open(f"{data_dir}/{split}.json"))
        for count, doc in enumerate(docs):
            print(f"Done for {count}/{len(docs)}", end="\r")
            rel_map = {}
            lbl_cnt = ddict(int)
            interdict = ddict(list)
            parses_arr = []
            srl_dict = {}

            if text_tokenizer == "scispacy":
                sents = [x for x in nlp(doc["text"]).sents]

            for rel in doc["rels"]:
                start_span = min(rel["arg1_start"], rel["arg2_start"])
                end_span = max(rel["arg1_end"], rel["arg2_end"])
                rel_map[(start_span, end_span)] = (
                    rel["arg1_start"],
                    rel["arg1_end"],
                    rel["arg1_word"],
                    rel["arg1_label"],
                    rel["arg2_start"],
                    rel["arg2_end"],
                    rel["arg2_word"],
                    rel["arg2_label"],
                    rel["arg_label"],
                )

            # sent_idxs = [0]+[max([tok.idx for tok in sent])+1 for sent in sents]
            sent_idxs = [0] + [sent.end_char for sent in sents]
            sent_ints = [(sent_idxs[i], sent_idxs[i + 1]) for i in range(0, len(sent_idxs) - 1)]
            rel_ints = sorted(rel_map)

            for rel_start, rel_end in rel_ints:
                for sent_cnt, sent in enumerate(sent_ints):
                    sent_start, sent_end = sent_ints[sent_cnt]
                    if rel_start > sent_end:
                        continue
                    if rel_end < sent_start:
                        break
                    interdict[(rel_start, rel_end)].append(
                        (sents[sent_cnt], sent_ints[sent_cnt][0])
                    )

            for rel_int in interdict:
                (
                    arg1_start,
                    arg1_end,
                    arg1_word,
                    arg1_label,
                    arg2_start,
                    arg2_end,
                    arg2_word,
                    arg2_label,
                    arg_lbl,
                ) = rel_map[rel_int]
                arg1_ann_map, arg2_ann_map, bw_arg_ann_map = (
                    IntervalMapping(),
                    IntervalMapping(),
                    IntervalMapping(),
                )

                arg1_ann_map[arg1_start:arg1_end] = (arg1_start, arg1_end, arg1_label)
                arg2_ann_map[arg2_start:arg2_end] = (arg2_start, arg2_end, arg2_label)
                bw_arg_ann_map[min(arg1_start, arg2_start) : max(arg2_end, arg1_end)]

                sent_str, sent_start = "", None

                for elem in interdict[rel_int]:
                    sent, sent_start_pos = elem
                    if sent_start is None:
                        sent_start = sent_start_pos
                    sent_str += str(sent)
                    if text_tokenizer == "scispacy":
                        sent_toks = sent

                    srl_sents.add(sent_str)

    print(f"Total number of sentences = {len(srl_sents)}")

    # predictor = predictors.SrlTransformersPredictor.from_path("/projects/flow_graphs/models/srl_bert_base_conll2012.tar.gz", "transformer_srl")
    # for sent in tqdm(list(srl_sents)):
    # 	srls = predictor.predict(sentence= sent)
    # 	srl_dict[sent] = get_srl_parses(srls)

    # dump_pickle(srl_dict, f'{data_dir}/srl_sents.pkl')


def dump_amrs(data_dir, splits, bert_model, text_tokenizer="scispacy"):
    from transformers import AutoTokenizer, AutoModel

    bad_counts = 0
    import amrlib, spacy
    from amrlib.graph_processing.annotator import add_lemmas
    from amrlib.alignments.rbw_aligner import RBWAligner

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model)

    amrlib.setup_spacy_extension()

    if text_tokenizer == "scispacy":
        nlp = spacy.load("en_core_sci_md")
    elif text_tokenizer == "spacy":
        nlp = spacy.load("en_core_web_sm")

    bad_amr_sents = open(f"{data_dir}/bad_amr_sents.log", "w")
    sent_amr_dict = {}
    amr_sents = set()
    for split in splits:
        docs = json.load(open(f"{data_dir}/{split}.json"))
        for count, doc in enumerate(docs):
            print(f"Done for {count}/{len(docs)}", end="\r")
            rel_map = {}
            lbl_cnt = ddict(int)
            interdict = ddict(list)
            parses_arr = []

            sents = [x for x in nlp(doc["text"]).sents]

            for rel in doc["rels"]:
                start_span = min(rel["arg1_start"], rel["arg2_start"])
                end_span = max(rel["arg1_end"], rel["arg2_end"])
                rel_map[(start_span, end_span)] = (
                    rel["arg1_start"],
                    rel["arg1_end"],
                    rel["arg1_word"],
                    rel["arg1_label"],
                    rel["arg2_start"],
                    rel["arg2_end"],
                    rel["arg2_word"],
                    rel["arg2_label"],
                    rel["arg_label"],
                )

            # sent_idxs = [0]+[max([tok.idx for tok in sent])+1 for sent in sents]
            sent_idxs = [0] + [sent.end_char for sent in sents]
            sent_ints = [(sent_idxs[i], sent_idxs[i + 1]) for i in range(0, len(sent_idxs) - 1)]
            rel_ints = sorted(rel_map)

            for rel_start, rel_end in rel_ints:
                for sent_cnt, sent in enumerate(sent_ints):
                    sent_start, sent_end = sent_ints[sent_cnt]
                    if rel_start > sent_end:
                        continue
                    if rel_end < sent_start:
                        break
                    interdict[(rel_start, rel_end)].append(
                        (sents[sent_cnt], sent_ints[sent_cnt][0])
                    )

            for rel_int in interdict:
                (
                    arg1_start,
                    arg1_end,
                    arg1_word,
                    arg1_label,
                    arg2_start,
                    arg2_end,
                    arg2_word,
                    arg2_label,
                    arg_lbl,
                ) = rel_map[rel_int]
                arg1_ann_map, arg2_ann_map, bw_arg_ann_map = (
                    IntervalMapping(),
                    IntervalMapping(),
                    IntervalMapping(),
                )

                arg1_ann_map[arg1_start:arg1_end] = (arg1_start, arg1_end, arg1_label)
                arg2_ann_map[arg2_start:arg2_end] = (arg2_start, arg2_end, arg2_label)
                bw_arg_ann_map[min(arg1_start, arg2_start) : max(arg2_end, arg1_end)]

                sent_str, sent_start = "", None

                for elem in interdict[rel_int]:
                    sent, sent_start_pos = elem
                    if sent_start is None:
                        sent_start = sent_start_pos
                    # sent_str 						+=str(sent.strip())
                    sent_str += str(sent)
                    sent_toks = sent
                    amr_sents.add(sent_str)

    print(f"Total number of sentences = {len(amr_sents)}")

    bad_counts = 0
    amr_dict = ddict(dict)
    for sent in tqdm(list(amr_sents)):
        try:
            doc = nlp(sent)
            bert_toks = tokenizer(sent, return_tensors="pt")
            bert_embs = model(**bert_toks)["last_hidden_state"][0]
            input_ids = list(np.array(bert_toks["input_ids"].squeeze(dim=0).cpu()))
            node_dict = {}
            edges_arr = []
            node2str = {}
            node2embs = {}

            start_tok_idx, end_tok_idx = 1, 1
            tok_pos = []

            for count, span in enumerate(doc.sents):
                graph_string = span._.to_amr()
                penman_graph = add_lemmas(graph_string[0], snt_key="snt")
                aligner = RBWAligner.from_penman_w_json(
                    penman_graph
                )  # use this with an annotated penman graph object
                aligned_string = aligner.get_graph_string()
                alignments = (
                    aligned_string.split("::")[4]
                    .split("\n")[0]
                    .replace("alignments", "")
                    .strip()
                    .split()
                )
                tokens = ast.literal_eval(
                    aligned_string.split("::")[2].split("\n")[0][len("tokens") + 1 :]
                )

                for tok_idx, tok_str in enumerate(tokens):
                    curr_bert_toks = tokenizer.encode(tok_str, add_special_tokens=False)
                    end_tok_idx = start_tok_idx + len(curr_bert_toks)
                    tok_pos.append((tok_str, tok_idx, start_tok_idx, end_tok_idx))
                    start_tok_idx = end_tok_idx
                    # for i in range(start_tok_idx, len(input_ids)):
                    # 	if input_ids[i] 			== curr_bert_toks[0]:
                    # 		start_tok_idx 			= i
                    # 		for j in range(i, i+len(curr_bert_toks)):
                    # 			assert curr_bert_toks[j-i]== input_ids[j]
                    # 		end_tok_idx				= j
                    # 		tok_pos.append((tok_str, tok_idx, start_tok_idx, end_tok_idx))
                    # 		start_tok_idx 			= end_tok_idx
                    # 		break

                # triples 							= penman_graph.triples
                edges = penman_graph.edges()

                for n1, rel, n2 in edges:
                    n1 = f"{count}-{n1}"
                    n2 = f"{count}-{n2}"
                    if n1 not in node_dict:
                        node_dict[n1] = len(node_dict)
                    if n2 not in node_dict:
                        node_dict[n2] = len(node_dict)

                for edge in edges:
                    src, rel, tgt = edge.source, edge.role, edge.target
                    edges_arr.append((f"{count}-{src}", rel, f"{count}-{tgt}"))

                last_aligned_str = aligned_string.split("::")[4]
                aligned_idx = last_aligned_str.index("\n") + 1
                aligned_graph_strs = last_aligned_str[aligned_idx:].strip().split("\n")

                last_node = None
                for aligned_node in aligned_graph_strs:
                    e_match = re.search(r"~e\.\d+", aligned_node)
                    z_match = re.search(r"\(z\d+ \/", aligned_node)
                    if e_match is not None and z_match is not None:
                        e_tok = int(e_match[0].split(".")[1])
                        z_node = z_match[0][1:].split("/")[0]
                        node2str[f"{count}-{z_node}"] = tok_pos[e_tok]
                    elif z_match is not None:
                        z_node = z_match[0][1:].split("/")[0]
                    elif e_match is not None:
                        e_tok = int(e_match[0].split(".")[1])
                        node2str[f"{count}-{last_node}"] = tok_pos[e_tok]
                for node in node2str:
                    start_idx, end_idx = node2str[node][2], node2str[node][3]
                    node2embs[node] = torch.max(bert_embs[start_idx:end_idx], dim=0)[0]

            amr_dict[sent]["node2str"] = node2str
            amr_dict[sent]["node2embs"] = node2embs
            amr_dict[sent]["edges_arr"] = edges_arr
            amr_dict[sent]["node_dict"] = node_dict
            amr_dict[sent]["triples"] = penman_graph.triples
            amr_dict[sent]["tok_pos"] = tok_pos
            amr_dict[sent]["aligned_graph"] = aligned_graph_strs

        except Exception as e:
            print(e)
            bad_amr_sents.write(sent + "\n")
            # bad_amr_sents.write(e)

    print(bad_counts)
    dump_pickle(amr_dict, f"{data_dir}/amr_sents.pkl")


"""
def get_amrs(sent, bert_model, text_tokenizer='scispacy'):
    from transformers import AutoTokenizer, AutoModel
    import amrlib, spacy	
    from amrlib.graph_processing.annotator import add_lemmas
    from amrlib.alignments.rbw_aligner import RBWAligner
    
    tokenizer         	= 	AutoTokenizer.from_pretrained(bert_model)
    model 				= 	AutoModel.from_pretrained(bert_model)
 
    amrlib.setup_spacy_extension()
 
    if text_tokenizer == 'scispacy': nlp = spacy.load("en_core_sci_md")
    elif text_tokenizer == 'spacy':  nlp = spacy.load("en_core_web_sm")
 
    bad_amr_sents 				=  open(f'{data_dir}/bad_amr_sents.log','w') 
    sent_amr_dict 				=  {}
    amr_sents 		 			=  set()
    for split in splits:
        docs 					= json.load(open(f'{data_dir}/{split}.json'))
        for count, doc in enumerate(docs):
            print(f'Done for {count}/{len(docs)}', end='\r')
            rel_map 			= {}
            lbl_cnt 			= ddict(int)
            interdict	   		= ddict(list)
            parses_arr 			= []

        
            sents = [x for x in nlp(doc['text']).sents]

            for rel in doc['rels']:
                start_span 	= min(rel['arg1_start'], rel['arg2_start'])
                end_span	= max(rel['arg1_end'], rel['arg2_end'])
                rel_map[(start_span, end_span)] = (rel['arg1_start'], rel['arg1_end'], rel['arg1_word'], rel['arg1_label'], \
                                                rel['arg2_start'], rel['arg2_end'], rel['arg2_word'], rel['arg2_label'], rel['arg_label'])

            # sent_idxs = [0]+[max([tok.idx for tok in sent])+1 for sent in sents]
            sent_idxs = [0]+[sent.end_char for sent in sents]
            sent_ints = [(sent_idxs[i], sent_idxs[i+1]) for i in range(0,len(sent_idxs)-1)]
            rel_ints  = sorted(rel_map)

            for rel_start, rel_end in rel_ints:
                for sent_cnt, sent in enumerate(sent_ints):
                    sent_start, sent_end = sent_ints[sent_cnt]
                    if rel_start 	> sent_end:		 continue
                    if rel_end 		< sent_start:	 break
                    interdict[(rel_start, rel_end)].append((sents[sent_cnt], sent_ints[sent_cnt][0]))

            for rel_int in interdict:
                arg1_start, arg1_end, arg1_word, arg1_label,  arg2_start, arg2_end, arg2_word, arg2_label, arg_lbl = rel_map[rel_int]
                arg1_ann_map, arg2_ann_map, bw_arg_ann_map 				= IntervalMapping(), IntervalMapping(), IntervalMapping()
                
                arg1_ann_map[arg1_start:arg1_end] = (arg1_start ,arg1_end, arg1_label)
                arg2_ann_map[arg2_start:arg2_end] = (arg2_start ,arg2_end, arg2_label)
                bw_arg_ann_map[min(arg1_start, arg2_start): max(arg2_end, arg1_end)]

                sent_str, sent_start 				= "", None

                for elem  in interdict[rel_int]:
                    sent, sent_start_pos 			= elem
                    if sent_start is None:
                        sent_start 					= sent_start_pos
                    # sent_str 						+=str(sent.strip())
                    sent_str 						+= str(sent)
                    sent_toks 						= sent
                    amr_sents.add(sent_str)
    
    print(f"Total number of sentences = {len(amr_sents)}")
 
    bad_counts 										= 0
    amr_dict 		    							= ddict(dict)
    for sent in tqdm(list(amr_sents)):
        try:
            doc 									= nlp(sent)	
            bert_toks 								= tokenizer(sent, return_tensors='pt')
            bert_embs 								= model(**bert_toks)['last_hidden_state'][0]
            input_ids 								= list(np.array(bert_toks['input_ids'].squeeze(dim=0).cpu()))
            node_dict 								= {}
            edges_arr 								= []
            node2str 								= {}
            node2embs								= {}
    
            start_tok_idx, end_tok_idx 				= 1,1
            tok_pos 								= []
    
            for count, span in enumerate(doc.sents):
                graph_string 						= span._.to_amr()
                penman_graph 						= add_lemmas(graph_string[0], snt_key='snt')
                aligner 							= RBWAligner.from_penman_w_json(penman_graph)    # use this with an annotated penman graph object
                aligned_string  					= aligner.get_graph_string()   
                alignments 							= aligned_string.split('::')[4].split('\n')[0].replace('alignments','').strip().split()
                tokens                              = ast.literal_eval(aligned_string.split('::')[2].split('\n')[0][len('tokens')+1:])

                for tok_idx, tok_str in enumerate(tokens):
                    curr_bert_toks 					= tokenizer.encode(tok_str, add_special_tokens=False)
                    end_tok_idx						= start_tok_idx + len(curr_bert_toks)
                    tok_pos.append((tok_str, tok_idx, start_tok_idx, end_tok_idx))
                    start_tok_idx 					= end_tok_idx
                    # for i in range(start_tok_idx, len(input_ids)):
                    # 	if input_ids[i] 			== curr_bert_toks[0]:
                    # 		start_tok_idx 			= i
                    # 		for j in range(i, i+len(curr_bert_toks)):
                    # 			assert curr_bert_toks[j-i]== input_ids[j]
                    # 		end_tok_idx				= j
                    # 		tok_pos.append((tok_str, tok_idx, start_tok_idx, end_tok_idx))
                    # 		start_tok_idx 			= end_tok_idx
                    # 		break
    
                # triples 							= penman_graph.triples
                edges 								= penman_graph.edges()

                for n1,rel,n2 in edges:
                    n1 = f'{count}-{n1}' 
                    n2 = f'{count}-{n2}'
                    if n1 not in node_dict: node_dict[n1] 				= len(node_dict)
                    if n2 not in node_dict: node_dict[n2] 				= len(node_dict)
                
                for edge in edges:
                    src, rel, tgt 										= edge.source, edge.role, edge.target 
                    edges_arr.append((f'{count}-{src}', rel, f'{count}-{tgt}'))

                last_aligned_str 					= aligned_string.split('::')[4]
                aligned_idx 						= last_aligned_str.index('\n') +1
                aligned_graph_strs					= last_aligned_str[aligned_idx:].strip().split('\n')
                
                last_node 							= None
                for aligned_node in aligned_graph_strs:
                    e_match = re.search(r'~e\.\d+',aligned_node)
                    z_match = re.search(r'\(z\d+ \/',aligned_node)
                    if e_match is not None and z_match is not None:
                        e_tok  = int(e_match[0].split('.')[1])
                        z_node = z_match[0][1:].split('/')[0]
                        node2str[f'{count}-{z_node}'] = tok_pos[e_tok]
                    elif z_match is not None:
                        z_node = z_match[0][1:].split('/')[0]
                    elif e_match is not None:
                        e_tok  = int(e_match[0].split('.')[1])
                        node2str[f'{count}-{last_node}'] = tok_pos[e_tok]
     
     
                for node in node2str:
                    start_idx, end_idx 			= node2str[node][2], node2str[node][3]
                    node2embs[node]				= torch.max(bert_embs[start_idx:end_idx],dim=0)[0]

            amr_dict[sent]['node2str']		= node2str
            amr_dict[sent]['node2embs']		= node2embs
            amr_dict[sent]['edges_arr']		= edges_arr
            amr_dict[sent]['node_dict']		= node_dict
            amr_dict[sent]['triples']		= penman_graph.triples
            amr_dict[sent]['tok_pos']		= tok_pos
            amr_dict[sent]['aligned_graph'] = aligned_graph_strs

        
        except Exception as e:
            print(e)
            bad_amr_sents.write(sent+'\n')
            # bad_amr_sents.write(e)
            
    print(bad_counts)
    dump_pickle(amr_dict, f'{data_dir}/amr_sents2.pkl')
            

"""


def add_parses(data_dir, splits, text_tokenizer="scispacy"):
    from transformer_srl import dataset_readers, models, predictors

    stanza_nlp = stanza.Pipeline(lang="en")
    data = ddict(lambda: ddict(list))
    labels = {}
    if text_tokenizer == "scispacy":
        nlp = spacy.load("en_core_sci_md")
    bad_counts = 0
    for split in splits:
        docs = json.load(open(f"{data_dir}/{split}.json"))

        for count, doc in enumerate(docs):
            print(f"Done for {count}/{len(docs)}", end="\r")
            rel_map = {}
            lbl_cnt = ddict(int)
            interdict = ddict(list)
            parses_arr = []

            srl_dict = {}

            if text_tokenizer == "scispacy":
                sents = [x for x in nlp(doc["text"]).sents]

            for rel in doc["rels"]:
                start_span = min(rel["arg1_start"], rel["arg2_start"])
                end_span = max(rel["arg1_end"], rel["arg2_end"])
                rel_map[(start_span, end_span)] = (
                    rel["arg1_start"],
                    rel["arg1_end"],
                    rel["arg1_word"],
                    rel["arg1_label"],
                    rel["arg2_start"],
                    rel["arg2_end"],
                    rel["arg2_word"],
                    rel["arg2_label"],
                    rel["arg_label"],
                )

            # sent_idxs = [0]+[max([tok.idx for tok in sent])+1 for sent in sents]
            sent_idxs = [0] + [sent.end_char for sent in sents]
            sent_ints = [(sent_idxs[i], sent_idxs[i + 1]) for i in range(0, len(sent_idxs) - 1)]
            rel_ints = sorted(rel_map)

            for rel_start, rel_end in rel_ints:
                for sent_cnt, sent in enumerate(sent_ints):
                    sent_start, sent_end = sent_ints[sent_cnt]
                    if rel_start > sent_end:
                        continue
                    if rel_end < sent_start:
                        break
                    interdict[(rel_start, rel_end)].append(
                        (sents[sent_cnt], sent_ints[sent_cnt][0])
                    )

            for rel_int in interdict:
                (
                    arg1_start,
                    arg1_end,
                    arg1_word,
                    arg1_label,
                    arg2_start,
                    arg2_end,
                    arg2_word,
                    arg2_label,
                    arg_lbl,
                ) = rel_map[rel_int]
                arg1_ann_map, arg2_ann_map, bw_arg_ann_map = (
                    IntervalMapping(),
                    IntervalMapping(),
                    IntervalMapping(),
                )

                arg1_ann_map[arg1_start:arg1_end] = (arg1_start, arg1_end, arg1_label)
                arg2_ann_map[arg2_start:arg2_end] = (arg2_start, arg2_end, arg2_label)
                bw_arg_ann_map[min(arg1_start, arg2_start) : max(arg2_end, arg1_end)]

                sent_str, sent_start = "", None

                for elem in interdict[rel_int]:
                    sent, sent_start_pos = elem
                    if sent_start is None:
                        sent_start = sent_start_pos
                    sent_str += str(sent)
                    if text_tokenizer == "scispacy":
                        sent_toks = sent

                arg1_start_idx, arg2_start_idx, arg1_end_idx, arg2_end_idx = (
                    arg1_start - sent_start,
                    arg2_start - sent_start,
                    arg1_end - sent_start,
                    arg2_end - sent_start,
                )

                try:
                    assert sent_str[arg1_start_idx:arg1_end_idx] == arg1_word
                    assert sent_str[arg2_start_idx:arg2_end_idx] == arg2_word
                except Exception as e:
                    try:
                        arg1_start_idx, arg2_start_idx = sent_str.index(arg1_word), sent_str.index(
                            arg2_word
                        )
                        arg1_end_idx, arg2_end_idx = arg1_start_idx + len(
                            arg1_word
                        ), arg2_start_idx + len(arg2_word)
                        assert sent_str[arg1_start_idx:arg1_end_idx] == arg1_word
                        assert sent_str[arg2_start_idx:arg2_end_idx] == arg2_word
                    except Exception as e:
                        bad_counts += 1

                dep_arr = []
                dep_doc = stanza_nlp(sent_str)
                for sent in dep_doc.sentences:
                    for word in sent.words:
                        word_id, word_text, word_head, word_deprel = (
                            word.id,
                            word.text,
                            word.head,
                            word.deprel,
                        )

                        if word.start_char >= arg1_start_idx and word.end_char <= arg1_end_idx:
                            dep_val = 1
                        elif word.start_char >= arg2_start_idx and word.end_char <= arg2_end_idx:
                            dep_val = 2
                        else:
                            dep_val = 0
                        dep_arr.append((word_id, word_head, word_text, word_deprel, dep_val))

                predictor = predictors.SrlTransformersPredictor.from_path(
                    "/projects/flow_graphs/models/srl_bert_base_conll2012.tar.gz",
                    "transformer_srl",
                )

                if sent_str not in srl_dict:
                    srls = predictor.predict(sentence=sent_str)
                    srl_dict[sent_str] = get_srl_parses(srls)

                parses_arr.append(
                    {
                        "file": doc["_id"],
                        "arg_label": arg_lbl,
                        "sent": sent_str,
                        "deps": dep_arr,
                        "srls": srl_dict[sent_str],
                        "arg1_start": arg1_start,
                        "arg1_end": arg1_end,
                        "arg1_word": arg1_word,
                        "arg1_label": arg1_label,
                        "arg2_start": arg2_start,
                        "arg2_end": arg2_end,
                        "arg2_word": arg2_word,
                        "arg2_label": arg2_label,
                    }
                )

            doc["parses"] = parses_arr

        with open(f"{data_dir}/parses_{split}.json", "w") as json_file:
            json.dump(docs, json_file, indent=4)

        print(bad_counts)
    return


def add_info():
    pass


def create_datafield(
    data_dir,
    splits,
    bert_model="bert-base-uncased",
    text_tokenizer="scispacy",
    omit_rels=[],
    dep_flag=True,
):
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model)
    from torch_geometric.data import Data

    stanza_nlp = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
    deprel_dict = load_deprels(enhanced=False)

    rel2desc, all_rel2id, id2all_rel, rel2desc_emb = generate_reldesc()
    data = ddict(lambda: ddict(list))
    labels = {}

    if text_tokenizer == "scispacy":
        nlp = spacy.load("en_core_sci_md")
    elif text_tokenizer == "spacy":
        nlp = spacy.load("en_core_web_sm")

    bad_sents = []
    for split in splits:
        docs = json.load(open(f"{data_dir}/{split}.json"))
        # docs 		= json.load(open(f'{data_dir}/parses_{split}.json'))
        for count, doc in enumerate(docs):
            print(f"Done for {count}/{len(docs)}", end="\r")
            rel_map = {}
            lbl_cnt = ddict(int)
            interdict = ddict(list)
            sents = [x for x in nlp(doc["text"]).sents]

            for rel in doc["rels"]:
                start_span = min(rel["arg1_start"], rel["arg2_start"])
                end_span = max(rel["arg1_end"], rel["arg2_end"])
                rel_map[(start_span, end_span)] = (
                    rel["arg1_start"],
                    rel["arg1_end"],
                    rel["arg1_word"],
                    rel["arg1_label"],
                    rel["arg2_start"],
                    rel["arg2_end"],
                    rel["arg2_word"],
                    rel["arg2_label"],
                    rel["arg_label"],
                )

            sent_idxs = [0] + [max([tok.idx for tok in sent]) for sent in sents]
            sent_ints = [(sent_idxs[i], sent_idxs[i + 1]) for i in range(0, len(sent_idxs) - 1)]
            rel_ints = sorted(rel_map)

            for rel_start, rel_end in rel_ints:
                for sent_cnt, sent in enumerate(sent_ints):
                    sent_start, sent_end = sent_ints[sent_cnt]
                    if rel_start > sent_end:
                        continue
                    if rel_end < sent_start:
                        break
                    interdict[(rel_start, rel_end)].append(sents[sent_cnt])

            ### creates each separate instance for each relation.

            for rel_int in interdict:
                (
                    arg1_start,
                    arg1_end,
                    arg1_word,
                    arg1_label,
                    arg2_start,
                    arg2_end,
                    arg2_word,
                    arg2_label,
                    arg_lbl,
                ) = rel_map[rel_int]

                # if arg_lbl in omit_rels: continue

                tokens, tok_range, org_toks, arg1_tokens, arg2_tokens = (
                    [],
                    [],
                    [],
                    [],
                    [],
                )
                arg1_ann_map, arg2_ann_map, bw_arg_ann_map = (
                    IntervalMapping(),
                    IntervalMapping(),
                    IntervalMapping(),
                )

                sent_str = ""
                sent_start = None
                sent_end = None
                for sent in interdict[rel_int]:
                    if sent_start is None:
                        sent_start = sent.start_char
                    sent_end = sent.end_char
                    # if   text_tokenizer == 'scispacy':
                    # sent_toks = sent

                    # for tok in sent_toks:
                    # 	# if   text_tokenizer == 'scispacy':
                    # 	start, end = tok.idx, tok.idx + len(tok)
                    # 	bert_toks = tokenizer.encode_plus(tok.text,  add_special_tokens=False)['input_ids']
                    # 	if len(bert_toks) != 0:
                    # 		tokens    += bert_toks
                    # 		tok_range.append((start-sent_start , end-sent_start))
                    # 		org_toks.append(tok.text)

                    # 	if arg1_ann_map.contains(start,end):	arg1_tokens += [1]*len(bert_toks)
                    # 	else:									arg1_tokens += [0]*len(bert_toks)

                    # 	if arg2_ann_map.contains(start,end): 	arg2_tokens += [1]*len(bert_toks)
                    # 	else:								 	arg2_tokens += [0]*len(bert_toks)

                sent_str = doc["text"][sent_start:sent_end]
                ## obtained the sentence boundary for the two relations in question.
                assert sent_start <= min(arg1_start, arg2_start) and sent_end >= max(
                    arg2_end, arg1_end
                )

                e1_start, e1_end, e2_start, e2_end = (
                    arg1_start - sent_start,
                    arg1_end - sent_start,
                    arg2_start - sent_start,
                    arg2_end - sent_start,
                )
                arg1_ann_map[e1_start:e1_end] = (e1_start, e1_end, arg1_label)
                arg2_ann_map[e2_start:e2_end] = (e2_start, e2_end, arg2_label)
                # bw_arg_ann_map[min(e1_start, e2_start): max(e2_end, e1_end)]

                sent_toks = tokenizer(sent_str, return_offsets_mapping=True, max_length=512)
                bert_toks = sent_toks["input_ids"]
                tok_range = sent_toks["offset_mapping"]

                e1_toks = (
                    [0]
                    + [
                        1 if arg1_ann_map.contains(elem[0], elem[1]) else 0
                        for elem in tok_range[1:-1]
                    ]
                    + [0]
                )
                e2_toks = (
                    [0]
                    + [
                        1 if arg2_ann_map.contains(elem[0], elem[1]) else 0
                        for elem in tok_range[1:-1]
                    ]
                    + [0]
                )

                assert 1 in e1_toks
                assert 1 in e2_toks
                # spans for e1_toks, e2_toks

                node_dict = {}
                node_idx_dict = {}
                node_mask_dict = {}
                edge_arr = []
                dep_arr = []

                # Specifically for the root that is attached to the main verb
                # STAR NODE
                node_dict[(-1, -1)] = 0
                node_idx_dict[(-1, -1)] = (1, len(bert_toks) - 1)
                node_mask_dict[(-1, -1)] = 0

                dep_doc = stanza_nlp(sent_str)
                num_sents = len(dep_doc.sentences)

                for sent_cnt, dep_sent in enumerate(dep_doc.sentences):
                    for word in dep_sent.words:
                        if arg1_ann_map.contains(
                            word.start_char, word.end_char
                        ) and arg2_ann_map.contains(word.start_char, word.end_char):
                            dep_val = 3
                        elif arg2_ann_map.contains(word.start_char, word.end_char):
                            dep_val = 2
                        elif arg1_ann_map.contains(word.start_char, word.end_char):
                            dep_val = 1
                        else:
                            dep_val = 0

                        # if 		word.start_char	>= e1_start  and word.end_char <= e1_end : dep_val = 1
                        # elif	word.start_char >= e2_start  and word.end_char <= e2_end : dep_val = 2
                        # else:   dep_val = 0
                        dep_arr.append(
                            (
                                (sent_cnt, word.id),
                                word.text,
                                (sent_cnt, word.head),
                                word.deprel,
                                word.start_char,
                                word.end_char,
                                dep_val,
                            )
                        )

                last_six, last_eix = 1, 1

                for elem in dep_arr:
                    (
                        start_idx,
                        word,
                        end_idx,
                        deprel,
                        start_char,
                        end_char,
                        mask_val,
                    ) = elem
                    curr_map = IntervalMapping()
                    curr_map[start_char:end_char] = 1
                    if start_idx not in node_dict:
                        node_dict[start_idx] = len(node_dict)
                    if end_idx not in node_dict:
                        node_dict[end_idx] = len(node_dict)

                    start_flag = False
                    start_tok_idx = 1
                    end_tok_idx = 1
                    for idx in range(start_tok_idx, len(tok_range) - 1):
                        curr_start, curr_end = tok_range[idx][0], tok_range[idx][1]
                        if curr_map.contains(curr_start, curr_end):
                            if start_flag is False:
                                start_tok_idx = idx
                                start_flag = True
                            end_tok_idx = idx + 1

                        # if 		curr_end 	== 0  and idx ==len(tok_range)-1 : end_tok_idx =idx; break
                        # elif 	curr_end 	<= start_char	: start_tok_idx = idx +1; continue
                        # elif 	curr_end	<= end_char		: continue
                        # elif	curr_start 	>= end_char		: end_tok_idx = idx; break

                    # if 	idx == len(tok_range) -2		: end_tok_idx = idx+1
                    # if 	idx == len(tok_range) -1		: end_tok_idx = idx+1
                    if start_tok_idx == 1 and end_tok_idx == 1:
                        start_tok_idx, end_tok_idx = last_six, last_eix
                    node_idx_dict[start_idx] = (start_tok_idx, end_tok_idx)

                    if ":" in deprel:
                        deprel = deprel.split(":")[0]
                    # edge_index[0].append(start_idx)
                    # edge_index[1].append(end_idx)
                    # edge_type.append(deprel_dict[deprel])
                    edge_arr.append((start_idx, end_idx, deprel_dict[deprel]))
                    node_mask_dict[start_idx] = mask_val
                    last_six, last_eix = start_tok_idx, end_tok_idx

                for sent_num in range(num_sents):
                    tok_idxs = [
                        node_idx_dict[elem] for elem in node_idx_dict if elem[0] == sent_num
                    ]
                    min_tok_idx = min([tok_idx[0] for tok_idx in tok_idxs])
                    max_tok_idx = max([tok_idx[1] for tok_idx in tok_idxs])

                    node_idx_dict[(sent_num, 0)] = (min_tok_idx, max_tok_idx)
                    node_mask_dict[(sent_num, 0)] = 0

                x, edge_index, edge_type, n1_mask, n2_mask = [], [[], []], [], [], []
                for node in node_dict:
                    six, eix = node_idx_dict[node]
                    temp_ones = torch.ones((512,)) * -torch.inf

                    try:
                        assert six < eix
                    except Exception as e:
                        import pdb

                        pdb.set_trace()
                    temp_ones[six:eix] = 0
                    x.append(temp_ones)

                    mask = node_mask_dict[node]
                    if mask == 0:
                        n1_mask.append(0)
                        n2_mask.append(0)
                    if mask == 1:
                        n1_mask.append(1)
                        n2_mask.append(0)
                    if mask == 2:
                        n1_mask.append(0)
                        n2_mask.append(1)
                    if mask == 3:
                        n1_mask.append(1)
                        n2_mask.append(1)

                for edge in edge_arr:
                    n1, n2, rel_idx = edge
                    edge_index[0].append(node_dict[n1])
                    edge_index[1].append(node_dict[n2])
                    edge_type.append(rel_idx)

                for sent_num in range(num_sents):
                    edge_index[0].append(node_dict[(sent_num, 0)])
                    edge_index[1].append(node_dict[(-1, -1)])
                    edge_type.append(deprel_dict["STAR"])

                try:
                    assert sum(n1_mask) > 0 and sum(n2_mask) > 0
                except Exception as e:
                    import pdb

                    pdb.set_trace()

                try:
                    x, edge_index, edge_type, n1_mask, n2_mask = (
                        torch.stack(x, dim=0),
                        torch.LongTensor(edge_index),
                        torch.LongTensor(edge_type),
                        torch.LongTensor(n1_mask),
                        torch.LongTensor(n2_mask),
                    )
                    dep_data = Data(
                        x=x,
                        edge_index=edge_index,
                        edge_type=edge_type,
                        n1_mask=n1_mask,
                        n2_mask=n2_mask,
                    )
                except Exception as e:
                    import pdb

                    pdb.set_trace()

                # import pdb; pdb.set_trace()

                data[split]["rels"].append(
                    {
                        "tokens": bert_toks,
                        "file": doc["_id"],
                        "tok_range": tok_range,
                        "org_toks": org_toks,
                        "arg1_ids": e1_toks,
                        "arg2_ids": e2_toks,
                        "desc_emb": rel2desc_emb[arg_lbl],
                        "span_info": rel_map[rel_int],
                        "label": arg_lbl,
                        "sent": sent_str,
                        "sent_start": sent_start,
                        "dep_data": dep_data,
                        "amr_data": [],
                    }
                )

        if "japflow" in data_dir:
            rels_data = list(data[split]["rels"])
            random.shuffle(rels_data)
            data["train"]["rels"] = rels_data[0 : int(len(rels_data) * 0.8)]
            data["dev"]["rels"] = rels_data[int(len(rels_data) * 0.8) : int(len(rels_data) * 0.9)]
            data["test"]["rels"] = rels_data[int(len(rels_data) * 0.9) :]

    return data


def create_mini_batch(samples):
    from torch_geometric.loader import DataLoader

    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    marked_e1 = [s[2] for s in samples]
    marked_e2 = [s[3] for s in samples]
    relation_emb = [s[4] for s in samples]

    if samples[0][5] is not None:
        label_ids = torch.stack([s[5] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    marked_e1 = pad_sequence(marked_e1, batch_first=True)
    marked_e2 = pad_sequence(marked_e2, batch_first=True)
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    relation_emb = torch.tensor(relation_emb)

    graph_list = [s[6] for s in samples]
    graph_loader = DataLoader(graph_list, batch_size=len(graph_list))
    graph_tensors = [elem for elem in graph_loader][0]

    return (
        tokens_tensors,
        segments_tensors,
        marked_e1,
        marked_e2,
        masks_tensors,
        relation_emb,
        label_ids,
        graph_tensors,
    )


def create_mini_batch_orig(samples):
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    marked_e1 = [s[2] for s in samples]
    marked_e2 = [s[3] for s in samples]
    relation_emb = [s[4] for s in samples]

    if samples[0][5] is not None:
        label_ids = torch.stack([s[5] for s in samples])
    else:
        label_ids = None

    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    marked_e1 = pad_sequence(marked_e1, batch_first=True)
    marked_e2 = pad_sequence(marked_e2, batch_first=True)
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    relation_emb = torch.tensor(relation_emb)

    # graph_list 						= [s[6] for s in samples]
    # graph_loader 					= DataLoader(graph_list, batch_size=len(graph_list))
    # graph_tensors 					= [elem for elem in  graph_loader][0]

    return (
        tokens_tensors,
        segments_tensors,
        marked_e1,
        marked_e2,
        masks_tensors,
        relation_emb,
        label_ids,
    )


def create_fewshot_data():
    for dataset in ["risec", "japflow", "mscorpus", "chemu"]:
        data = load_dill(f"../data/{dataset}/data.dill")
        rels_data = list(data["train"]["rels"])
        random.shuffle(rels_data)
        for fewshot in [0.01, 0.05, 0.1, 0.2, 0.5]:
            data["train"]["rels"] = rels_data[0 : int(len(rels_data) * fewshot)]
            dump_dill(data, f"../data/{dataset}/data-{fewshot}.dill")


def create_dataset():
    if args.dataset == "risec":
        create_risec()
    elif args.dataset == "japflow":
        create_japflow()
    elif args.dataset == "mscorpus":
        create_mscorpus()
    elif args.dataset == "chemu":
        create_chemu()
    # elif    args.dataset == 'curd': 	create_curd()


def create_parses():
    if args.dataset == "risec":
        if args.parse == "srl":
            dump_srls("../data/risec/", ["train", "dev", "test"], text_tokenizer="scispacy")
        elif args.parse == "amr":
            dump_amrs(
                "../data/risec/",
                ["train", "dev", "test"],
                bert_model="bert-base-uncased",
                text_tokenizer="scispacy",
            )
        # dataset 	= add_parses('../data/risec/',['train','dev','test'])
        # dump_dill(dataset, f'../data/risec/parses.dill')


def load_dataset():
    # if  args.dataset == 'risec':

    # dataset 	= create_datafield(f'../data/{args.dataset}/',['train','dev','test'], bert_model ='bert-base-uncased', text_tokenizer='scispacy')
    # dump_dill(dataset, f'../data/{args.dataset}/data.dill')

    if args.dataset == "japflow":
        dataset = create_datafield(
            f"../data/{args.dataset}/",
            ["all"],
            bert_model="bert-base-uncased",
            text_tokenizer="scispacy",
        )
        dump_dill(dataset, f"../data/{args.dataset}/data.dill")

    # elif args.dataset == 'japflow': 	create_japflow()
    # elif    args.dataset == 'curd': 	create_curd()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for analysis")
    parser.add_argument("--step", required=True, help="Which function to call")
    parser.add_argument("--dataset", default="risec", help="Which dataset")
    parser.add_argument("--parse", default="amr", help="Type of parse afforded by the model")
    args = parser.parse_args()

    if args.step == "create":
        create_dataset()
    elif args.step == "load":
        load_dataset()
    elif args.step == "fewshot":
        create_fewshot_data()
    elif args.step == "rel_emb":
        generate_reldesc()
    elif args.step == "parse":
        create_parses()
    elif args.step == "srl":
        create_srl_tags()
