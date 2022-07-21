from tqdm import tqdm


def load_glove(file="/data/glove_vector/glove.6B.100d.txt"):
    glove_dict = {}
    lines = open(file).readlines()
    for line in tqdm(lines):
        data = line.strip().split()
        glove_dict[data[0]] = [float(data[i]) for i in range(1, len(data))]

    return glove_dict


def load_deprels(file="/projects/flow_graphs/data/enh_dep_rel.txt", enhanced=False):
    dep_dict = {}
    lines = open(file).readlines()
    for line in tqdm(lines):
        data = line.strip().split()
        rel_name = data[0]
        if enhanced:
            if rel_name.endswith(":"):
                rel_name = rel_name[:-1]
        else:
            rel_name = rel_name.split(":")[0]
        if rel_name not in dep_dict:
            dep_dict[rel_name] = len(dep_dict)

    if "STAR" not in dep_dict:
        dep_dict["STAR"] = len(dep_dict)

    return dep_dict


punct_dict = {
    "<": "_lt_",
    ">": "_gt_",
    "+": "_plus_",
    "?": "_question_",
    "&": "_amp_",
    ":": "_colon_",
    ".": "_period_",
    "!": "_exclamation_",
    ",": "_comma",
}


