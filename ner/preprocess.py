import argparse
from glob import glob
import os
from tqdm import tqdm

def processes_ans(ann_fname):
    ann_data = []
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
                    "label": 'ENT',
                }
            )
            ann_dict[index] = ann_arr_idx
            ann_arr_idx += 1

    return ann_data


def create_risec():
    data_dir  = "/data/flow_graphs/COOKING/RISEC/data"
    out_dir   = '/projects/flow_graphs/data/ner/risec/'

    for split in ["train", "dev", "test"]:
        if split =='dev':
            split2 = 'valid'
        else:
            split2 = split
        
        try:
            os.mkdir(f'{out_dir}/{split2}')
        except Exception as e:
            print("Directory exists")
        
        
        ann_files = glob(f"{data_dir}/{split}/*.ann")
        for ann_fname in tqdm(ann_files):
            ann_file = open(ann_fname)
            txt_fname = ann_fname.replace(".ann", ".txt")
            doc_id = ann_fname.split("/")[-1][:-4]
            
            out_ann   = open(f'{out_dir}/{split2}/{doc_id}.ann','w')
            out_txt   = open(f'{out_dir}/{split2}/{doc_id}.txt','w')
            
            try:
                text = open(txt_fname).read()
                # text 			 = preprocess_texts(text)
            except Exception as e:
                print(e)
                continue

            anns      = processes_ans(ann_fname)

            for ann in anns:
                out_ann.write(f'{ann["_id"]}\t{ann["label"]} {ann["start"]} {ann["end"]}\t{ann["word"]}\n')
            
            out_txt.write(text)
            
            out_ann.close()
            out_txt.close()


create_risec()