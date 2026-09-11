# Upstream integration: 2026-09-11

Base: upstream `5f764400059eb5a2182468bd92b9c13688c875ab`.
Branch: `integrate/upstream-20260911-wsl2`.
Ported patch: local `a0cb5679bf7e56bb0a0f449be7fbe1d59bfba93d`.

The original checkout, `master`, installed symlink and binary remain unchanged.
Original binary SHA256:
`1b01bd22e75f0cad335f1ecf124a5e3772cfe952d03f79ffaef88c75dbef9683`.
The separate `add-torchnep` worktree is not included or modified.

## Patch

Keep upstream's independent `NEP_Extrapolation` object and its B-projection buffer.
Replace managed-memory gamma, ASI matrices and cuBLAS pointer arrays with device
allocations and explicit host staging. Copy gamma to its host mirror before CPU
threshold checks and extxyz output. The pre-CUDA-12 GEMV branch reads pointers from
host storage and uses the new upstream B-projection accessor.

## Build and validation

From `src`:

```sh
make -j2 gpumd CC=/usr/local/cuda-12.6/bin/nvcc CUDA_ARCH=-arch=sm_86
```

Build passed (exit 0). Build log: `build-gpumd.log`.
Target is the local RTX 3080 (sm_86). `ldd src/gpumd` resolves all dependencies.
New binary SHA256:
`74676e389c4847801aeb74fe603cc9dc9aed77e2b8b66ab1cca9b2bb6698f00a`.
No CUDA toolkit or driver installation is performed.
LSP cannot inspect this worktree because it lies outside the session cwd;
the CUDA compiler is the source validation surface.

GPU runtime qualification is pending. The GPU had only 30 MiB free after building
so no competing GPU job was stopped and no smoke job was launched.
`integration-smoke/run.in` is a five-step NVE smoke using the new input syntax.
Its model, initial structure and ASI links refer to an existing local fixture at
`/tmp/gpumd_traj_x4ac4pe_`; preserve these inputs before deleting that temporary
fixture. This is an integration smoke, not material qualification.

## Deployment blockers

Do not replace the installed binary before adapting and testing nepstill:

- `compute_extrapolation` now requires `nep_file nep.txt`.
- `dump_exyz` is removed. Use `dump_xyz <stride> dump.xyz velocity force potential`.
- Check the new extxyz output schema against nepstill's actual reader, including
  velocity units, physical energy/force labels and unwrapped coordinates.

The existing calls in nepstill's `active_learning/gpumd_protocol.py` still use
the old syntax. They have deliberately not been changed in this GPUMD-only port.
No build of `nep`/`gnep`, multi-GPU training, or TorchNEP integration is claimed.
