import zmq
import sys
from io import StringIO

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:5555")

    while True:
        # Wait for request
        pde_code = socket.recv_string()

        # Execute the received code to solve the PDE and capture the output
        stdout = sys.stdout
        sys.stdout = StringIO()  # Redirect stdout to capture output
        try:
            exec(pde_code)
            result = sys.stdout.getvalue()
        finally:
            sys.stdout = stdout  # Restore original stdout

        # Send the result back to the client
        socket.send_string(result)

if __name__ == "__main__":
    main()
