#!/usr/bin/env python3
"""Compare two checkpoints tensor by tensor: compare_states.py <ours.pt> <agent.pt> [label]."""
import sys

import torch

ours = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
agent = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
label = sys.argv[3] if len(sys.argv) > 3 else sys.argv[1]
a, b = ours["state_dict"], agent["state_dict"]
if a.keys() != b.keys():
    print(f"{label}: DIFFERENT keys")
    sys.exit(1)
differing = [k for k in a if not torch.equal(a[k], b[k])]
largest = max((float((a[k].double() - b[k].double()).abs().max()) for k in differing), default=0.0)
print(f"{label}: epoch {ours['epoch']} vs {agent['epoch']}; {len(a)} tensors, {len(differing)} differ, "
      f"largest difference {largest:.3g}")
sys.exit(1 if differing or ours["epoch"] != agent["epoch"] else 0)
