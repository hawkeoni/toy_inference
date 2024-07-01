from typing import List

import requests


class ServerClient:
    
    def __init__(self, host: str, port: str):
        self.host = host
        self.port = port
        self.addr = host + ":" + port
    
    def infer(self, data: List[int]) -> float:
        payload = " ".join(map(str, data)) + "\nEOS\n"
        req = requests.post(f"http://{self.addr}", data=payload)
        req.raise_for_status()
        return float(req.content.decode())


client = ServerClient("localhost", "7878")

print(client.infer([1, 2, 3, 4, 6]))