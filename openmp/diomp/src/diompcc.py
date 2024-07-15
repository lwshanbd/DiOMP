#!/usr/bin/env python3

import sys
import subprocess
import re

def main():
    if len(sys.argv) < 2:
        print("Usage: diompCC <source_file> [-conduit=X] [additional_args]...")
        return 1
    args = sys.argv[1:]

    conduit = "ibv"
    for arg in args:
        if arg.startswith("-conduit="):
            conduit = arg.split("=")[1]
            args.remove(arg)
            break

    base_compile_cmd = f"clang++ -ldiomp -lgasnet-{conduit}-par -libverbs -lrt -pthread -fopenmp -lmpi -lhwloc"
    compile_cmd = f"{base_compile_cmd} {' '.join(args)}"
    subprocess.check_call(compile_cmd, shell=True)

if __name__ == "__main__":
    sys.exit(main())