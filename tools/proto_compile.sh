#!/bin/bash

PROTOS_DIR="../proto/"
python3 -m grpc_tools.protoc -I${PROTOS_DIR} --python_out=. --pyi_out=. --grpc_python_out=. ll_server.proto
