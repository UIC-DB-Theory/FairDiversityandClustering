import json
import subprocess

import numpy as np


class WeightedTree:

    def __init__(self, dimension: int):
        """
        This is the constructor for the ParGeoWeightedKDtree, this object is bound to a process that allows the user to
        use the ParGeoWeightedKdtree via json strings sent over pipe.
        :param dimension: the dimensions for the tree.
        """
        assert 1 < dimension < 8, "dimension is outside of acceptable values"
        self.dim = dimension
        self.executable = './ParGeoWeightedTree/build/example/TreeClient'
        # self.executable = './ParGeoCtl/pargeoctl'
        # error value for calculated coreset
        self.proc = subprocess.Popen(
            [self.executable, str(dimension)],
            # [self.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
        )

        self.points = np.array([])
        self.N = None

    def _send_message(self, message_str):
        """
        "Private method" used to actually send the message to the C++ application.
        :param message_str: The message string to send.
        :return: the response json dict.
        """
        self.proc.stdin.write(bytes(message_str + "\n", 'utf-8'))
        self.proc.stdin.flush()
        response_str = self.proc.stdout.readline().decode()
        response_str = response_str[:-1]  # strip the newline
        return json.loads(response_str)

    def construct_tree(self, points: np.ndarray):
        """
        creates a data structure with the given points, returns the time taken to do the action
        :param points: the points to create the kd tree
        :return: the time it took to create the kd tree
        """
        message_json = {
            "type": "build-datastructure",
            "dimension": self.dim,
            "points": points.flatten().tolist()
        }

        self.N = len(points)

        message_str = json.dumps(message_json)
        response = self._send_message(message_str)
        # TODO: check response and handle if errors occurred
        assert response['status'] == 'OK', "construct tree response returned an error"
        return response['time']

    def run_query(self, radius, weights):
        """
        creates a json message for the data structure to run a radius query with the given weights and radius.
        :param radius: The radius for the query
        :param weights: a np array of weights
        :return: a tuple of the time recorded to perform the action and an np array of weights.
        """
        assert (self.N == len(weights.flatten()))

        message_json = {
            "type": "run-query",
            "radius": radius,
            "weights": weights.flatten().tolist()}

        message_str = json.dumps(message_json)

        response_json = self._send_message(message_str)
        # TODO: check response and handle if errors occurred
        assert response_json['status'] == 'OK', "query response returned an error"

        return response_json['time'], np.array(response_json['result'])

    def delete_tree(self):
        """
        This will delete the data structure and close the data structure program
        :return: void
        """
        message = {"type": "exit"}

        json_message = json.dumps(message)

        self.proc.stdin.write(bytes(json_message + "\n", 'utf-8'))
        self.proc.stdin.flush()
