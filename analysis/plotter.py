from matplotlib import pyplot as plt
import pandas as pd
import sys
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="plotter")
    parser.add_argument("path")
    parser.add_argument("--title", default=None)
    parser.add_argument("--skip", default=25, type=int)
    parser.add_argument("--head", default=1, type=int)
    parser.add_argument("--x", default="epoch")
    parser.add_argument("--y1", default="train_acc")
    parser.add_argument("--y2", default="test_acc")

    args = parser.parse_args(sys.argv[1:])
    if args.title is None:
        args.title = args.path
    print(args)

    data = pd.read_csv(args.path, header=args.head, skiprows=args.skip)
    ax = data.plot(x=args.x, y=args.y1, color="Blue", title=args.title)
    data.plot(x=args.x, y=args.y2, color="Red", ax=ax)
    plt.grid()
    plt.show()
