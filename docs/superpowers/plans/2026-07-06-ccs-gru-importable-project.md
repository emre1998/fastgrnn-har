# Importable CCS GRU Project Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `msp/ccs_gru_har` a self-contained CCS project that can be imported, built, and run on an MSP430G2553 without manual project setup.

**Architecture:** Package the existing GRU firmware with CCS managed-build metadata, the MSP430G2553 linker/runtime support files, and a target configuration. Derive the metadata from the already verified local CCS project, remove old-workspace absolute paths, rename the project, and prove correctness with a clean TI `cl430` build.

**Tech Stack:** Code Composer Studio 20.5.1, TI MSP430 compiler 21.6.1 LTS, GNU Make supplied by CCS, MSP430G2553.

## Global Constraints

- Project name is exactly `GRU HAR MSP430G2553`.
- Device is exactly `MSP430G2553`.
- Compiler settings are `-O3`, `--use_hw_mpy=none`, heap 0 bytes, stack 256 bytes, ROM runtime model.
- Build inputs are `gru.cpp`, `main.cpp`, `system_pre_init.c`, and `lnk_msp430g2553.cmd`.
- No absolute reference to `workspace_ccstheia` may remain.
- Generated `Debug` and `Release` outputs must remain untracked.

---

### Task 1: CCS managed-project metadata

**Files:**
- Create: `msp/ccs_gru_har/.project`
- Create: `msp/ccs_gru_har/.cproject`
- Create: `msp/ccs_gru_har/.ccsproject`

**Interfaces:**
- Consumes: Existing GRU sources in `msp/ccs_gru_har` and the verified metadata template in `C:/Users/EMRE CAN/workspace_ccstheia/Msp430 Fastgrnn Project Experiment`.
- Produces: A CCS-discoverable project named `GRU HAR MSP430G2553` with Debug and Release managed builds.

- [ ] **Step 1: Verify the project is not currently importable**

Run:

```powershell
$p = 'msp/ccs_gru_har'
@('.project','.cproject','.ccsproject') | ForEach-Object {
    if (-not (Test-Path (Join-Path $p $_))) { Write-Output "MISSING $_" }
}
```

Expected before implementation: three `MISSING` lines.

- [ ] **Step 2: Add `.project`**

Create the Eclipse/CCS project descriptor with this project identity and managed-build natures:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<projectDescription>
  <name>GRU HAR MSP430G2553</name>
  <comment></comment>
  <projects></projects>
  <buildSpec>
    <buildCommand><name>org.eclipse.cdt.managedbuilder.core.genmakebuilder</name><arguments></arguments></buildCommand>
    <buildCommand><name>org.eclipse.cdt.managedbuilder.core.ScannerConfigBuilder</name><arguments></arguments></buildCommand>
  </buildSpec>
  <natures>
    <nature>com.ti.ccstudio.core.ccsNature</nature>
    <nature>org.eclipse.cdt.core.cnature</nature>
    <nature>org.eclipse.cdt.managedbuilder.core.managedBuildNature</nature>
    <nature>org.eclipse.cdt.core.ccnature</nature>
    <nature>org.eclipse.cdt.managedbuilder.core.ScannerConfigNature</nature>
  </natures>
</projectDescription>
```

- [ ] **Step 3: Add `.ccsproject` and `.cproject` from the verified template**

Use the working CCS project metadata as the compatibility baseline. Preserve both Debug and Release configurations, compiler version 21.6.1, `DEVICE_CONFIGURATION_ID=MSP430G2553`, and `targetConfigs/MSP430G2553.ccxml`. Remove all include entries containing `workspace_ccstheia`; local headers resolve from the project directory. Confirm these exact settings exist in the resulting `.cproject`:

```xml
<listOptionValue value="DEVICE_CONFIGURATION_ID=MSP430G2553"/>
<option value="3" valueType="enumerated"/>
<option value="none" valueType="enumerated"/>
<option value="0" valueType="string"/>
<option value="256" valueType="string"/>
```

The option IDs and surrounding XML must remain those from the verified TI-generated template so CCS recognizes them.

- [ ] **Step 4: Validate metadata invariants**

Run:

```powershell
$p = 'msp/ccs_gru_har'
$all = Get-Content -Raw "$p/.project","$p/.cproject","$p/.ccsproject"
if ($all -notmatch 'GRU HAR MSP430G2553') { throw 'Project name missing' }
if ($all -notmatch 'MSP430G2553') { throw 'Device missing' }
if ($all -match 'workspace_ccstheia') { throw 'Old absolute workspace path remains' }
'metadata: PASS'
```

Expected: `metadata: PASS`.

- [ ] **Step 5: Commit project metadata**

```powershell
git add -f msp/ccs_gru_har/.project msp/ccs_gru_har/.cproject msp/ccs_gru_har/.ccsproject
git commit -m "build: add importable CCS GRU project metadata"
```

### Task 2: MSP430G2553 linker, startup, and target support

**Files:**
- Create: `msp/ccs_gru_har/lnk_msp430g2553.cmd`
- Create: `msp/ccs_gru_har/system_pre_init.c`
- Create: `msp/ccs_gru_har/targetConfigs/MSP430G2553.ccxml`
- Create: `msp/ccs_gru_har/targetConfigs/readme.txt`

**Interfaces:**
- Consumes: TI-provided MSP430G2553 support files from the verified local project.
- Produces: Linkable memory layout, runtime pre-initialization, and an MSP430G2553 debug connection definition.

- [ ] **Step 1: Verify required support files are absent**

Run:

```powershell
$p = 'msp/ccs_gru_har'
@('lnk_msp430g2553.cmd','system_pre_init.c','targetConfigs/MSP430G2553.ccxml') | ForEach-Object {
    if (-not (Test-Path (Join-Path $p $_))) { Write-Output "MISSING $_" }
}
```

Expected before implementation: three `MISSING` lines.

- [ ] **Step 2: Add the verified TI support files without semantic changes**

Copy the exact contents of these working files into the matching paths under `msp/ccs_gru_har`:

```text
C:/Users/EMRE CAN/workspace_ccstheia/Msp430 Fastgrnn Project Experiment/lnk_msp430g2553.cmd
C:/Users/EMRE CAN/workspace_ccstheia/Msp430 Fastgrnn Project Experiment/system_pre_init.c
C:/Users/EMRE CAN/workspace_ccstheia/Msp430 Fastgrnn Project Experiment/targetConfigs/MSP430G2553.ccxml
C:/Users/EMRE CAN/workspace_ccstheia/Msp430 Fastgrnn Project Experiment/targetConfigs/readme.txt
```

Do not copy `Debug`, `.theia`, `test_data.h`, or any FastGRNN source/object file.

- [ ] **Step 3: Validate the linker memory model and target**

Run:

```powershell
$p = 'msp/ccs_gru_har'
$link = Get-Content -Raw "$p/lnk_msp430g2553.cmd"
$target = Get-Content -Raw "$p/targetConfigs/MSP430G2553.ccxml"
if ($link -notmatch 'RAM' -or $link -notmatch 'FLASH') { throw 'Linker memory regions missing' }
if ($target -notmatch 'MSP430G2553') { throw 'Target device missing' }
'target support: PASS'
```

Expected: `target support: PASS`.

- [ ] **Step 4: Commit target support**

```powershell
git add msp/ccs_gru_har/lnk_msp430g2553.cmd msp/ccs_gru_har/system_pre_init.c msp/ccs_gru_har/targetConfigs
git commit -m "build: add MSP430G2553 CCS target support"
```

### Task 3: Import documentation and clean build proof

**Files:**
- Create: `msp/ccs_gru_har/README.md`
- Verify: `msp/ccs_gru_har/Debug/GRU HAR MSP430G2553.map`
- Verify: `msp/ccs_gru_har/Debug/GRU HAR MSP430G2553.out`

**Interfaces:**
- Consumes: The complete project from Tasks 1 and 2 and installed CCS 20.5.1 tools.
- Produces: User-facing import/run instructions plus evidence that the exact packaged project builds within MSP430G2553 memory limits.

- [ ] **Step 1: Write the import and run instructions**

Document this exact flow:

```text
1. File > Import > Code Composer Studio > CCS Projects.
2. Browse to msp/ccs_gru_har and finish the import.
3. Confirm target MSP430G2553 and active configuration Debug.
4. Connect the LaunchPad, then choose Build Project.
5. Choose Run > Debug; after programming completes, resume execution.
6. Open a 9600-baud serial terminal to read GRU latency output.
```

Also document that `TEST_MODE=1` is latency mode and `TEST_MODE=3` is the INA226 energy benchmark.

- [ ] **Step 2: Generate a clean managed build**

Import the project once in CCS so it generates `Debug/makefile`, then run:

```powershell
$gmake = 'C:/ti/ccs2051/ccs/utils/bin/gmake.exe'
& $gmake -C 'msp/ccs_gru_har/Debug' clean
if ($LASTEXITCODE -ne 0) { throw 'Clean failed' }
& $gmake -C 'msp/ccs_gru_har/Debug' all
if ($LASTEXITCODE -ne 0) { throw 'Build failed' }
```

Expected final lines include `Finished building target` and exit code 0.

- [ ] **Step 3: Prove the linker used GRU and respected memory limits**

Run:

```powershell
$map = Get-ChildItem 'msp/ccs_gru_har/Debug' -Filter '*.map' | Select-Object -First 1
$text = Get-Content -Raw $map.FullName
if ($text -notmatch 'gru\.obj') { throw 'GRU object missing' }
if ($text -match 'fastgrnn\.obj') { throw 'Stale FastGRNN object linked' }
if ($text -notmatch 'RAM\s+00000200\s+00000200') { throw 'Unexpected RAM layout' }
if ($text -notmatch 'FLASH\s+0000c000\s+00003fde') { throw 'Unexpected Flash layout' }
'link verification: PASS'
```

Expected: `link verification: PASS`. Inspect the `used` columns and confirm RAM is below `0x200` bytes and Flash is below `0x3fde` bytes.

- [ ] **Step 4: Confirm generated outputs are ignored**

Run:

```powershell
git status --short -- msp/ccs_gru_har
```

Expected: no `Debug/`, `Release/`, `.out`, `.map`, `.obj`, or `.d` entries.

- [ ] **Step 5: Commit documentation**

```powershell
git add msp/ccs_gru_har/README.md
git commit -m "docs: add CCS GRU import and run guide"
```

### Task 4: Final verification

**Files:**
- Verify: `msp/ccs_gru_har/*`

**Interfaces:**
- Consumes: All prior tasks.
- Produces: A final pass/fail report for importability, build correctness, and repository cleanliness.

- [ ] **Step 1: Run all static project checks**

```powershell
$p = 'msp/ccs_gru_har'
$required = @('.project','.cproject','.ccsproject','lnk_msp430g2553.cmd','system_pre_init.c','targetConfigs/MSP430G2553.ccxml','gru.cpp','gru.h','main.cpp','model_weights.h','lut.h','README.md')
$missing = $required | Where-Object { -not (Test-Path (Join-Path $p $_)) }
if ($missing) { throw "Missing: $($missing -join ', ')" }
$metadata = Get-Content -Raw "$p/.project","$p/.cproject","$p/.ccsproject"
if ($metadata -match 'workspace_ccstheia|fastgrnn\.obj') { throw 'Stale project reference found' }
'project package: PASS'
```

Expected: `project package: PASS`.

- [ ] **Step 2: Run a fresh clean build and record memory usage**

Repeat Task 3 Steps 2 and 3 after deleting no source files and changing no settings. Expected: exit code 0, a newly timestamped `.out`, `gru.obj` in the map, RAM below 512 bytes, Flash below 16 KB.

- [ ] **Step 3: Review the final diff**

```powershell
git diff --check HEAD~3..HEAD
git status --short
```

Expected: no whitespace errors; unrelated pre-existing user changes remain untouched.
