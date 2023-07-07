import json
import subprocess

import numpy as np


class WeightedTree:

    def __init__(self, dimension: int):
        assert 1 < dimension < 8, "dimension is outside of acceptable values"
        self.dim = dimension
        self.executable = './ParGeoWeightedTree/build/example/TreeClient'
        # self.executable = './ParGeoCtl/pargeoctl'
        # error value for calculated coreset
        self.proc = subprocess.Popen(
            [self.executable, str(dimension)],
            # [self.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        self.points = np.array([])

    def send_message(self, message_str):
        self.proc.stdin.write(bytes(message_str + "\n", 'utf-8'))
        self.proc.stdin.flush()
        response_str = self.proc.stdout.readline().decode()
        response_str = response_str[:-1]  # strip the newline
        return json.loads(response_str)

    def construct_tree(self, points):
        message_json = {
            "type": "build-datastructure",
            "dimension": self.dim,
            "points": points.tolist()}

        message_str = json.dumps(message_json)
        response = self.send_message(message_str)
        # TODO: check response and handle if errors occurred
        return response

    def run_query(self, radius, weights):
        message_json = {
            "type": "run-query",
            "radius": radius,
            "weights": weights.tolist()}

        message_str = json.dumps(message_json)

        response_json = self.send_message(message_str)
        # TODO: check response and handle if errors occurred

        return response_json

    def delete_tree(self):
        message = {
            "type": "exit"}

        json_message = json.dumps(message)

        self.proc.stdin.write(bytes(json_message + "\n", 'utf-8'))
        self.proc.stdin.flush()


# Example usage.
if __name__ == "__main__":

    # Initialize the tree and the subprocess.
    tree = WeightedTree(3)

    # Construct the tree on the data.
    tree_construction_response = tree.construct_tree(np.array([0.0, 0.0, 0.0,
                                  1.0, 2.0, 3.0,
                                  1.0, 2.0, 3.0,
                                  1.0, 2.0, 3.0,
                                  1.0, 2.0, 3.0,
                                  1.0, 2.0, 3.0,
                                  1.0, 2.0, 3.0,
                                  1.0, 2.0, 3.0]))

    print(f"Constructed a tree: {tree_construction_response}")

    # Run a query on the tree.

    result1 = tree.run_query(1, np.array([1, 1, 1]))
    print(f"Query 1 Result: {result1}")

    # Run a query.

    result2 = tree.run_query(.5, np.array([1, 5, 10]))
    print(f"Query 2 Result: {result2}")

    # Delete the tree. Ensures the process exits correctly
    tree.delete_tree()
