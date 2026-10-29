import hashlib, torch
torch.manual_seed(42); torch.cuda.manual_seed_all(42)
p = torch.cuda.get_device_properties(0)
a = torch.rand(1_000_000, device="cuda").cpu().numpy().tobytes()
d = torch.nn.functional.dropout(torch.ones(1_000_000, device="cuda"), 0.1).cpu().numpy().tobytes()
r = torch.randperm(256, device="cuda").cpu().numpy().tobytes()
print(p.name, "SMs", p.multi_processor_count, "rand", hashlib.md5(a).hexdigest()[:10], "dropout", hashlib.md5(d).hexdigest()[:10], "randperm", hashlib.md5(r).hexdigest()[:10])
