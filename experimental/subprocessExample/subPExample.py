import asyncio
import sys

async def cat_proc():

    # Create the subprocess; redirect the standard output
    # into a pipe.
    proc = await asyncio.create_subprocess_exec(
        "cat",
        stdout=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE)

    proc.stdin.write(bytes("some input\n","utf-8"))

    data = await proc.stdout.readline()

    print(data)
    #proc.terminate()
    # Read one line of output.
    #data = await proc.stdout.readline()
    #line = data.decode('ascii').rstrip()

    # Wait for the subprocess exit.
    #await proc.wait()
    #return line




if __name__ == '__main__':
    date = asyncio.run(cat_proc())


    # proc = subprocess.Popen("cat", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # outs, errs = proc.communicate(input=bytes("This is input","utf-8"))
    # print("after cat")
    # print(outs)
