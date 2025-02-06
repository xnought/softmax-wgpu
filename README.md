# softmax-wgpu

(educational) fastest softmax in webgpu over a matrix across the final dimension.

GPU API from myself at https://github.com/xnought/webgpu-compute

**Roadmap**

- [x] Naive implementation without max
- [x] With max
- [ ] Online max calculation
- [ ] Online sum calculation
- [ ] Optimized shared memory and reductions

## Dev

```bash
python3 -m http.server 3000
```

Then go to http://localhost:3000/. Must refresh manually to get code changes (if you make them).