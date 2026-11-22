import torch, sys
for p in sys.argv[1:]:
    A = torch.load(p, map_location="cpu", mmap=True)
    print(p, type(A))
    if isinstance(A, dict):
        for k, v in A.items():
            if torch.is_tensor(v): print("  ", k, tuple(v.shape), v.dtype)
            elif isinstance(v, dict): print("  ", k, "dict", list(v.keys())[:10], [tuple(x.shape) for x in list(v.values())[:2] if torch.is_tensor(x)])
            else: print("  ", k, type(v))
    elif torch.is_tensor(A): print("  ", tuple(A.shape), A.dtype)
