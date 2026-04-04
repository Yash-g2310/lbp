import re, sys, os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse(path):
    train = defaultdict(list)
    val = defaultdict(list)
    pattern = re.compile(r'\[(train|val)\]\[epoch=(\d+)/\d+\].*?loss=([0-9]+(?:\.[0-9]+)?(?:e[-+]?\d+)?)', re.IGNORECASE)
    with open(path, 'r') as f:
        for line in f:
            m = pattern.search(line)
            if m:
                which = m.group(1).lower()
                e = int(m.group(2))
                try:
                    loss = float(m.group(3))
                except:
                    continue
                if which == 'train':
                    train[e].append(loss)
                else:
                    val[e].append(loss)
    return train, val


def avg_map(d):
    items = sorted(d.items())
    epochs = [k for k, _ in items]
    vals = [sum(vs) / len(vs) for _, vs in items]
    return epochs, vals


def main():
    if len(sys.argv) < 2:
        print('Usage: {} path/to/log'.format(sys.argv[0]))
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print('File not found:', path)
        sys.exit(2)

    train, val = parse(path)
    te, tl = avg_map(train) if train else ([], [])
    ve, vl = avg_map(val) if val else ([], [])

    if not te and not ve:
        print('No train/val loss entries found in', path)
        sys.exit(3)

    plt.figure(figsize=(8,5))
    if te:
        plt.plot(te, tl, marker='o', label='train')
    if ve:
        plt.plot(ve, vl, marker='o', label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(os.path.basename(path))
    plt.legend()
    plt.grid(True)

    out = os.path.splitext(os.path.basename(path))[0] + '_loss_curve.png'
    plt.tight_layout()
    plt.savefig(out)
    print('Saved', out)


if __name__ == '__main__':
    main()
