# 저자 체크포인트(PL 1.1)의 옵티마이저·스케줄러 상태 확인
# 사용: python3 tools/inspect_author_ckpt.py <ckpt> <train_rows>
import sys, types, pickle, math, torch


class Dummy:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): self.state = s
    def __call__(self, *a, **k): return None
    def __repr__(self): return f'Dummy({getattr(self, "state", None)})'


class U(pickle.Unpickler):
    def find_class(self, mod, name):
        try:
            return super().find_class(mod, name)
        except Exception:
            return Dummy


def load(path):
    pm = types.ModuleType('stubpickle')
    pm.Unpickler = U
    pm.load = lambda f, **k: U(f, **k).load()
    return torch.load(path, map_location='cpu', pickle_module=pm, weights_only=False)


if __name__ == '__main__':
    ck = load(sys.argv[1])
    rows = int(sys.argv[2])
    print('epoch', ck['epoch'], 'global_step', ck['global_step'])
    for s in ck['lr_schedulers']:
        print('scheduler:', {k: (str(v)[:200] if k == 'lr_lambdas' else v) for k, v in s.items()})
    o = ck['optimizer_states'][0]
    print('param_groups:', [{k: v for k, v in g.items() if k != 'params'} for g in o['param_groups']])
    st = o['state']
    k0 = next(iter(st))
    print('state keys:', list(st[k0].keys()), 'step:', st[k0].get('step'))
    T = int(rows / 64 * 10)
    W = int(T * 0.1)
    s = ck['global_step']
    lr = 3e-5 * 0.5 * (1 + math.cos(math.pi * (s - W) / (T - W)))
    print(f'우리 계산: num_train_steps={T} warmup={W} step {s} cosine lr={lr:.4e}')
