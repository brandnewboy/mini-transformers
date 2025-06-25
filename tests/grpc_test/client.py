import grpc
import hello_pb2
import hello_pb2_grpc

def run_hello_client():
    conn = grpc.insecure_channel('localhost:9003')
    client = hello_pb2_grpc.HelloServiceStub(conn)
    request = hello_pb2.HelloRequest(name="John", age=30)
    response = client.SayHello(request)
    print(response)


if __name__ == "__main__":
    run_hello_client()