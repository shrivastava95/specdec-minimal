import time
class notetime:
    def __init__(self, desc="Event"): self.desc = desc
    def __enter__(self):
        self.t = time.time()
        # print(f"[start] {self.desc}: {self.t}")
    def __exit__(self, *args):
        t2 = time.time()
        print(f"[end]   {self.desc}: (+{t2 - self.t:.4f}s)")