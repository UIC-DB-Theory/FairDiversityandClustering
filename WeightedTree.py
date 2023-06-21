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
        message = {
            "type": "build-datastructure", 
            "dimension": 3, 
            "points": points.tolist()}
        
        json_message= json.dumps(message)

        self.proc.stdin.write(bytes(json_message + "\n", 'utf-8'))
        self.proc.stdin.flush()
        a = self.proc.stdout.readline()
        print(a)


    def run_query(self, radius, weights):
        pass
    
    def delete_tree(self):
        message = {
            "type": "exit"}
        
        json_message= json.dumps(message)

        self.proc.stdin.write(bytes(json_message + "\n", 'utf-8'))
        self.proc.stdin.flush()
            

tree = WeightedTree()
tree.construct_tree(np.array([[0.0, 0.0 ,0.0],[1.0, 2.0 ,3.0]]))
for i in range(0,5):
    print(i)
tree.construct_tree(np.array([[0.0, 0.0 ,0.0],[1.0, 2.0 ,3.0]]))
for i in range(0,5):
    print(i)
tree.delete_tree()




