import subprocess
import numpy as np
import json

# proc = subprocess.Popen(
#     ['./ParGeoCtl/pargeoctl'],
#     stdin=subprocess.PIPE,
#     stdout=subprocess.PIPE
# )

# proc.stdin.write(bytes('abc def ghi\n', 'utf-8'))
# proc.stdin.flush()
# a = proc.stdout.readline()
# print(a)

# proc.stdin.write(bytes('123 456 789\n', 'utf-8'))
# proc.stdin.flush()
# a = proc.stdout.readline()
# print(a)

# proc.stdin.write(bytes('123 456 789\n', 'utf-8'))
# proc.stdin.flush()
# a = proc.stdout.readline()
# print(a)


class WeightedTree:

    def __init__(self):

        self.executable = './ParGeoCtl/pargeoctl'
        self.cat = 'cat'
        # error value for calculated coreset
        self.proc = subprocess.Popen(
            [self.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE
        )

        self.points = np.array([])
    
    def construct_tree(self, points):
        message_json = {
            "type": "build-datastructure", 
            "dimension": 3, 
            "points": points.tolist()}
        
        message_str = json.dumps(message_json)

        self.proc.stdin.write(bytes(message_str + "\n", 'utf-8'))
        self.proc.stdin.flush()
        response_str = self.proc.stdout.readline().decode()
        response_str = response_str[:-1] #strip the newline

        response_json = json.loads(response_str)
        # TODO: check response and handle if errors occured


    def run_query(self, radius, weights):
        message_json = {
            "type" : "run-query", 
            "radius" : radius, 
            "weights" : weights.tolist()}
        
        message_str = json.dumps(message_json)

        self.proc.stdin.write(bytes(message_str + "\n", 'utf-8'))
        self.proc.stdin.flush()
        response_str = self.proc.stdout.readline().decode()
        response_str = response_str[:-1] #strip the newline

        response_json = json.loads(response_str)
        # TODO: check response and handle if errors occured

        return (np.array(response_json["result"]),response_json["time"])

    def delete_tree(self):
        message = {
            "type": "exit"}
        
        json_message= json.dumps(message)

        self.proc.stdin.write(bytes(json_message + "\n", 'utf-8'))
        self.proc.stdin.flush()
            

# Example usage.
if __name__ == "__main__":
    
    # Initialize the tree and the subprocess.
    tree = WeightedTree()

    # COnstruct the tree on the data.
    tree.construct_tree(np.array([[0.0, 0.0 ,0.0],[1.0, 2.0 ,3.0]]))

    # Sample computation.
    for i in range(0,5):
        print(i)
    
    # Run a query on the tree.
    result, time = tree.run_query(1, np.array([1, 2]))
    print(f'Result: {result}')
    print(f'Time(s): {time}')

    # More sample computation.
    for i in range(0,5):
        print(i)
    
    # Run a query.
    result, time = tree.run_query(1, np.array([1, 2]))
    print(f'Result: {result}')
    print(f'Time(s): {time}')


    # Delete the tree. Ensures the process exits correctly
    tree.delete_tree()




