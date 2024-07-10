from argparse import ArgumentParser
from typing import List, Union, Any
import random
import time

import requests
from multiprocessing import Pool


VEC_SIZE = 1024

def generate_random_payload() -> List[float]:
    return [random.random() for _ in range(VEC_SIZE)]


class ServerClient:
    
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port
        self.addr = host + ":" + port
    
    def infer(self, data: List[Union[int, float]], timeout: int) -> List[Any]:
        payload = " ".join(map(str, data)) + "\n\n"
        req = requests.post(f"http://{self.addr}", data=payload, timeout=timeout)
        req.raise_for_status()
        return req.content.decode()

def shoot(client: ServerClient, timeout, duration, start_time):
    s, f = 0, 0
    payload = generate_random_payload()
    while time.time() - start_time < duration:
        try:
            _ = client.infer(payload, timeout)
            s += 1
        except:
            f += 1
    return s, f

def main(args):
    timeout = 2
    n_procs = 10
    duration = 10
    start_time = time.time()
    pool = Pool(n_procs)
    client = ServerClient("localhost", "7878")
    if args.action == "loadtest":
        results = pool.starmap(shoot, [[client, timeout, duration, start_time]] * 10)
        s = sum([r[0] for r in results])
        f = sum([r[1] for r in results])
        print(f"Successes: {s}, Failures: {f}, RPS is {s / duration}")
    elif args.action == "request":
        random.seed(1)
        payload = generate_random_payload()
        print(client.infer(payload, timeout))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", "-a", choices=["loadtest", "request"], required=True)
    args = parser.parse_args()
    main(args)
