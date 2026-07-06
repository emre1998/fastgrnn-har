# Importable CCS GRU Project Design

## Goal

Convert `msp/ccs_gru_har` into a self-contained Code Composer Studio project. The user imports the existing folder, builds it, and runs it on an MSP430G2553 without manually creating or reconfiguring a CCS project.

## Project structure

Keep the existing GRU sources in place and add the minimum CCS-managed project files:

- `.project`, `.cproject`, and `.ccsproject`
- `lnk_msp430g2553.cmd`
- `system_pre_init.c`
- `targetConfigs/MSP430G2553.ccxml`
- import and run instructions in `README.md`

Generated `Debug` and `Release` directories remain ignored.

## Build configuration

- Project name: `GRU HAR MSP430G2553`
- Device: `MSP430G2553`
- Compiler: TI MSP430 compiler 21.6.1 LTS
- Optimization: `-O3`
- Hardware multiplier: disabled (`--use_hw_mpy=none`)
- Heap: 0 bytes
- Stack: 256 bytes
- Runtime model: ROM
- Active target configuration: `targetConfigs/MSP430G2553.ccxml`

The build must compile `gru.cpp`, `main.cpp`, and `system_pre_init.c`, then link against the local linker command file and TI runtime library. No absolute path to the old FastGRNN workspace may remain in the project metadata.

## Import flow

In CCS, use **File > Import > Code Composer Studio > CCS Projects**, select `msp/ccs_gru_har`, then finish the import. The project should be immediately buildable through **Build Project** and runnable through **Run/Debug** with the connected MSP430G2553 LaunchPad.

## Verification

Run a clean command-line build with the installed CCS toolchain. Verification passes only if:

1. Compilation and linking exit with code 0.
2. The linker includes `gru.obj`, not a stale `fastgrnn.obj`.
3. A new `.out` and `.map` are produced.
4. Flash usage is below 16 KB and RAM usage is below 512 bytes.
5. The repository has no generated build outputs staged for commit.

## Scope

This change packages the existing GRU latency/energy firmware. It does not add live MPU6050 input, alter inference mathematics, or modify model weights.
