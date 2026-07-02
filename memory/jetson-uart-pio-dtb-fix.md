---
name: jetson-uart-pio-dtb-fix
description: This AGX Orin's ttyTHS1 RX was broken by a JetPack 6.2.2 DMA regression; fixed with a custom PIO DTB + extlinux entry
metadata:
  type: project
---

The competition AGX Orin Dev Kit (board `p3737-0000+p3701-0005`) running **L4T R36.5.0 / JetPack 6.2.2** (kernel 5.15.185-tegra) has a `serial-tegra` regression: `serial@3100000` (`/dev/ttyTHS1`) was switched to GPCDMA RX but is missing the `iommus = <&smmu_niso0 TEGRA234_SID_GPCDMA>` property, so the SMMU faults the DMA writes. Result: **UART TX works, RX returns the correct byte count but with the leading (DMA-portion) bytes zeroed and only the trailing PIO/FIFO bytes correct** (e.g. a 4x4 transform read back as all-zeros except a stray 0.5). Baud-independent. A second Jetson on R36.4.7 (5.15.148) was unaffected because UART-A defaulted to PIO there.

**Fix applied 2026-06-19 (this machine only):** forced ttyTHS1 to PIO by removing `dmas`/`dma-names` from the `serial@3100000` node.
- Custom DTB: `/boot/dtb/kernel_tegra234-p3737-0000+p3701-0005-nv-pio.dtb` (source/scratch in `~/uart_fix/`).
- `/boot/extlinux/extlinux.conf` has a new `LABEL pio` (`FDT` -> the custom DTB) set as `DEFAULT`; `LABEL primary` is untouched as a fallback. Backup at `extlinux.conf.bak-*`.
- Verify: `dmesg | grep 3100000` shows `RX in PIO mode` / `TX in PIO mode`.

**Caveats:** a JetPack/L4T update may overwrite the DTB or extlinux and silently revert to broken DMA — re-check `dmesg` after any system update. NVIDIA's alternative permanent fix is adding the `iommus` property (keeps DMA). Only ttyTHS1 was patched; other THS UARTs would need the same. The `serial-tegra` driver only uses DMA for transfers >= 16 bytes, which is why the 73-byte transform message in [[]] `src/subsystems/embedded_communicator.py` (`EmbeddedTransformationMessage`) hit the bug. Note that file's stale 115200/69-byte comments — runtime baud is 460800 and the struct is 73 bytes.
