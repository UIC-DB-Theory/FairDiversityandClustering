import subprocess
import sys

executable = 'cat'

proc = subprocess.Popen(
    ['cat'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)

proc.stdin.write(bytes('abc def ghi\n', 'utf-8'))
proc.stdin.write(bytes('123 456 789\n', 'utf-8'))
proc.stdin.flush()

a = proc.stdout.readline()
print(a)

a = proc.stdout.readline()
print(a)




