import redis
import torch
import pickle

r = redis.StrictRedis(host='localhost', port=6379, db=0)

inputt=pickle.dumps(torch.tensor([[1.1,2.2,3.3],[1.2,4,6]]))
print(r.set("0",inputt))
res=pickle.loads(r.get("0"))
print(res.__class__,res)