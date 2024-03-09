# import dask.array as da
# x = da.random.random((10000, 100), chunks=(1000, 1000))
# print(x)
# from dask.distributed import Client, progress
# client = Client(processes=False, threads_per_worker=2,
#                  n_workers=2, memory_limit='2GB')
# print(client)
# client.close()
class Solution:
     def majorityElement():
        nums = [3,1,-2,-5,2,-4]
        b=[k for k in nums if k<0]
        a=[j for j in nums if j>=0]
        f=[]
        for i in range(0,len(a)):
            f.append(a[i])
            f.append(b[i])
        print(f)
     majorityElement()
        