import sys

def hexa(s, alpha):
    if s[0] == '#':
        s = s[1:]
    r = int(s[:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    w = 255 * (1 - alpha)
    return "#%02x%02x%02x" % (int(w + r * alpha), int(w + g * alpha), int(w + b * alpha))

if __name__ == "__main__":
    print(hexa(sys.argv[1], float(sys.argv[2])))