import os
import gzip
import transformers
import logging

# logging.getLogger(transformers.__name__).setLevel(logging.WARNING)
# LOGGER = logging.getLogger(__name__)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')  # include timestamp


mid_to_title = {}
with open("/sas/zfj/scp/paperinfo/idx_title.tsv") as rf:
    for i, line in enumerate(rf):
        if i % 100000 == 0:
            logger.info("line %d", i)
        items = line.strip().split("\t")
        mid_to_title[int(items[0])] = items[1]
        # if i > 10000:
        #     break

out_dir = "/sas/zfj/projects/network-embedding/pecos-mine-1/examples/giant-xrt/proc_data_xrt/ogbn-papers100M/"
os.makedirs(out_dir, exist_ok=True)
wf = open(os.path.join(out_dir, "X.all.txt"), "w")
with gzip.open("/home/zfj/projects/network-embedding/ogb/examples/nodeproppred/papers100M/dataset/ogbn_papers100M/mapping/nodeidx2paperid.csv.gz", "rb") as fin:
    for i, line in enumerate(fin):
        if i % 100000 == 0:
            logger.info("write line %d", i)
            wf.flush()
        if i == 0:
            continue
        items = line.decode('utf8').strip().split(",")
        idx = int(items[0])
        mid = int(items[1])
        cur_text = mid_to_title.get(mid, "")
        wf.write(cur_text + "\n")
wf.close()
logger.info("done")
