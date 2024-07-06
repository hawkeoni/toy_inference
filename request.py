from argparse import ArgumentParser
from typing import List
import random
import time

import requests
from multiprocessing import Pool


VEC_SIZE = 2 ** 16
class ServerClient:
    
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port
        self.addr = host + ":" + port
    
    def infer(self, data: List[int], timeout: int) -> float:
        payload = " ".join(map(str, data)) + "\nEOS\n"
        req = requests.post(f"http://{self.addr}", data=payload, timeout=timeout)
        req.raise_for_status()
        return float(req.content.decode())

def shoot(client: ServerClient, timeout, duration, start_time):
    s, f = 0, 0
    payload = [random.random() for _ in range(VEC_SIZE)]
    print("start shooting")
    while time.time() - start_time < duration:
        try:
            res = client.infer(payload, timeout)
            s += 1
        except:
            f += 1
    print("finish shooting")
    return s, f

def main(args):
    timeout = 0.5
    n_procs = 10
    duration = 10
    start_time = time.time()
    pool = Pool(n_procs)
    client = ServerClient("localhost", "7878")
    if args.action == "loadtest":
        results = pool.starmap(shoot, [[client, timeout, duration, start_time]] * 10)
        s = sum([r[0] for r in results])
        f = sum([r[1] for r in results])
        print(s, f)
    elif args.action == "request":
        random.seed(1)
        payload = [random.random() for _ in range(VEC_SIZE)]
        print(client.infer(payload, timeout))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", "-a", choices=["loadtest", "request"], required=True)
    args = parser.parse_args()
    main(args)
