import sys
import random
from pathlib import Path


if __name__ == '__main__':
    args = sys.argv
    num_line = int(args[1])
    output_path = Path(args[2])

    with output_path.open(mode='w') as wf:
        for i in range(0, num_line):
            a = random.randint(1, 999)
            b = random.randint(1, 999)
            former = max(a, b)
            latter = min(a, b)

            line = f"{former}-{latter}"
            line += ' ' * (7 - len(line))
            line += f"_{former - latter}"
            line += ' ' * (11 - len(line))
            line += '\n'

            wf.write(line)
