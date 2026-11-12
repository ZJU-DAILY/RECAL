"""Deterministic average-node transfer used by the paper's zero-shot setting."""


def transfer_state(source, template):
    out, changed, total = {}, 0, 0
    for key, tmpl in template.items():
        x = source[key]
        total += tmpl.numel()
        if x.shape != tmpl.shape:
            if x.dim() != tmpl.dim():
                raise ValueError(f"Rank mismatch for {key}")
            for axis in range(x.dim()):
                if x.shape[axis] != tmpl.shape[axis]:
                    x = x.float().mean(dim=axis, keepdim=True)
                    x = x.expand(
                        *[tmpl.shape[i] if i == axis else -1 for i in range(x.dim())]
                    )
            changed += tmpl.numel()
        if x.shape != tmpl.shape:
            raise ValueError(f"Shape mismatch for {key}")
        out[key] = x.to(tmpl.dtype).contiguous().clone()
    return out, changed / max(total, 1)
