import asyncio
import sys

ParGeoCtl = './ParGeoCtl/pargeoctl'

async def pargeo_proc():

    # Create the subprocess; redirect the standard output
    # into a pipe.
    proc = await asyncio.create_subprocess_exec(
        ParGeoCtl,
        stdout=asyncio.subprocess.PIPE,
        stdin=asyncio.subprocess.PIPE)

    proc.stdin.write(bytes('{"type" : "run-query", "radius" : 1, "weights" : [1, 2]}',"utf-8"))

    data = await proc.stdout.readline()

    print(data)

if __name__ == '__main__':
    asyncio.run(pargeo_proc())
