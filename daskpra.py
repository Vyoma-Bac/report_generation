from dask.distributed import Client, progress
import dask.dataframe as dd
import time
import random
client = Client(processes=False, threads_per_worker=2,
                 n_workers=2, memory_limit='200mb')
print(client)


def costly_simulation(list_param):
    time.sleep(random.random())
    return sum(list_param)


import pandas as pd
import numpy as np

input_params = pd.DataFrame(np.random.random(size=(500, 4)),
                            columns=['param_a', 'param_b', 'param_c', 'param_d'])
results=[]
for parameters in input_params.values[:10]:
    result = costly_simulation(parameters)
    results.append(result)
print(results)
# df = dd.read_csv(r'D:\python_training\prof\marksheet.csv', blocksize="1kB")
# print(df.head())
# df.compute()
# print(df.npartitions)
# print(df.describe().compute())
# client.close()
