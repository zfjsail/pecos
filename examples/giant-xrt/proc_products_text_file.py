import os
import gzip
import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp

id_to_text = {}
with open("/home/zfj/scp/Amazon-3M.raw/trn.json") as rf:
    for i, line in enumerate(rf):
        if i % 100000 == 0:
            logger.info("line %d", i)
        cur_dict = json.loads(line)
        id_to_text[cur_dict["uid"]] = cur_dict["title"]

out_dir = "/home/zfj/projects/network-embedding/pecos-mine-1/examples/giant-xrt/proc_data_xrt/ogbn-products/"
os.makedirs(out_dir, exist_ok=True)
wf = open(os.path.join(out_dir, "X.all.txt"), "w")
with gzip.open("/home/zfj/projects/network-embedding/ogb/examples/nodeproppred/products/dataset/ogbn_products/mapping/nodeidx2asin.csv.gz") as fin:
    for i, line in enumerate(fin):
        if i % 100000 == 0:
            logger.info("write line %d", i)
            wf.flush()
        if i == 0:
            continue
        items = line.decode('utf8').strip().split(",")
        idx = int(items[0])
        mid = int(items[1])
        cur_text = id_to_text.get(mid, "").strip()
        wf.write(cur_text + "\n")
wf.close()
logger.info("done")
