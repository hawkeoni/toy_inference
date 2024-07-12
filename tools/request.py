from argparse import ArgumentParser
from typing import List, Union, Any
import random
import time

from multiprocessing import Pool
from ll_server_pb2_grpc import LLServiceStub
from ll_server_pb2 import LLRequest, LLResponse
import grpc


VEC_SIZE = 1024

def generate_random_payload() -> List[float]:
    return [random.random() for _ in range(VEC_SIZE)]


class ServerClient:
    
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port
        self.addr = host + ":" + port
        self.channel = grpc.insecure_channel(self.addr)
        self.stub = LLServiceStub(self.channel)
    
    def infer(self, data: List[Union[int, float]], timeout: int) -> List[Any]:
        resp = self.stub.LLDot(LLRequest(x=data))
        return resp.output

def shoot(timeout, duration, start_time):
    s, f = 0, 0
    client = ServerClient("localhost", "7878")
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
    if args.action == "loadtest":
        results = pool.starmap(shoot, [[timeout, duration, start_time]] * 10)
        s = sum([r[0] for r in results])
        f = sum([r[1] for r in results])
        print(f"Successes: {s}, Failures: {f}, RPS is {s / duration}")
    elif args.action == "request":
        client = ServerClient("localhost", "7878")
        random.seed(1)
        payload = generate_random_payload()
        print(client.infer(payload, timeout))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", "-a", choices=["loadtest", "request"], required=True)
    args = parser.parse_args()
    main(args)
