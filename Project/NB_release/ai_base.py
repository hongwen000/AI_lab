import csv
from tqdm import tnrange, tqdm_notebook
def List2CSV(filen, data):
    with open(filen, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data)

def CSV2List2(filen):
    ret = []
    with open(filen, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            ret.append(list(row))
        return ret

from multiprocessing import Pool
from typing import List, NoReturn, Callable
def list_multiprocess(lst: List, func: Callable[[List],List], n: int)-> List:
    if len(lst) < n:
        return func(lst)
    p = Pool(n)
    lists = []
    psize = int(len(lst) / n)
    for i in range(n - 1):
        lists.append(lst[i * psize: (i+1) * psize])
    lists.append(lst[(n-1) * psize:])
    ret = []
    for i in range(n):
        ret.append(p.apply_async(func, args=(lists[i],)))
    p.close()
    p.join()
    last = []
    for i in ret:
        last += i.get()
    return last
