import json, sys
d = json.load(open(sys.argv[1]))
for k, v in d["readers"].items():
    print(k.ljust(52), " ".join("L%s:%s/%s" % (l, v[l]["rank"]["cactus"], v[l]["rank"]["desert"]) for l in v))
