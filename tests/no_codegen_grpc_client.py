import logging

import grpc
import grpc.experimental

# NOTE: The path to the .proto file must be reachable from an entry
# on sys.path. Use sys.path.insert or set the $PYTHONPATH variable to
# import from files located elsewhere on the filesystem.

if __name__ == "__main__":
    protos = grpc.protos("index.proto")
    services = grpc.services("index.proto")

    response = services.Hello.SayHello(
        protos.HelloRequest(name="dhfiudshidsu", age=23), "localhost:50051", insecure=True
    )
    print("Hello client received: " + response.message)