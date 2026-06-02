from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


OUT = Path(__file__).with_name("us_rbc_snr_derivation.pdf")
PAGE_W, PAGE_H = 8.5, 11.0
LEFT, RIGHT, TOP, BOTTOM = 0.75, 0.75, 0.65, 0.65
BODY_SIZE = 10.0
SMALL_SIZE = 8.5
TITLE_SIZE = 18.0
SECTION_SIZE = 13.5
LINE = 0.19


def new_page(pdf: PdfPages):
    fig = plt.figure(figsize=(PAGE_W, PAGE_H))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    return fig, ax, PAGE_H - TOP


def finish_page(pdf: PdfPages, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def ensure_space(pdf: PdfPages, fig, ax, y, needed):
    if y - needed < BOTTOM:
        finish_page(pdf, fig)
        fig, ax, y = new_page(pdf)
    return fig, ax, y


def add_text(pdf, fig, ax, y, text, size=BODY_SIZE, bold=False, indent=0.0, width=96, leading=1.18):
    wrapped = []
    for para in text.split("\n"):
        if not para.strip():
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(para, width=width))
    needed = max(1, len(wrapped)) * LINE * leading + 0.03
    fig, ax, y = ensure_space(pdf, fig, ax, y, needed)
    for line in wrapped:
        if not line:
            y -= LINE * leading
            continue
        ax.text(
            LEFT + indent,
            y,
            line,
            transform=fig.dpi_scale_trans,
            fontsize=size,
            fontweight="bold" if bold else "normal",
            family="DejaVu Serif",
            va="top",
        )
        y -= LINE * leading
    return fig, ax, y - 0.04


def add_section(pdf, fig, ax, y, title):
    fig, ax, y = ensure_space(pdf, fig, ax, y, 0.42)
    ax.text(
        LEFT,
        y,
        title,
        transform=fig.dpi_scale_trans,
        fontsize=SECTION_SIZE,
        fontweight="bold",
        family="DejaVu Serif",
        va="top",
    )
    y -= 0.33
    ax.plot([LEFT, PAGE_W - RIGHT], [y, y], transform=fig.dpi_scale_trans, color="#333333", lw=0.8)
    return fig, ax, y - 0.17


def add_math(pdf, fig, ax, y, expr, size=12.0, extra=0.10):
    lines = expr.strip().split("\n")
    needed = 0.27 * len(lines) + extra + 0.05
    fig, ax, y = ensure_space(pdf, fig, ax, y, needed)
    for line in lines:
        ax.text(
            LEFT + 0.25,
            y,
            line,
            transform=fig.dpi_scale_trans,
            fontsize=size,
            family="DejaVu Serif",
            va="top",
        )
        y -= 0.28
    return fig, ax, y - extra


def add_bullets(pdf, fig, ax, y, items, size=BODY_SIZE):
    for item in items:
        fig, ax, y = ensure_space(pdf, fig, ax, y, 0.32)
        ax.text(
            LEFT + 0.05,
            y,
            u"\u2022",
            transform=fig.dpi_scale_trans,
            fontsize=size,
            family="DejaVu Serif",
            va="top",
        )
        fig, ax, y = add_text(pdf, fig, ax, y, item, size=size, indent=0.28, width=88, leading=1.12)
    return fig, ax, y


def add_table(pdf, fig, ax, y, rows):
    fig, ax, y = ensure_space(pdf, fig, ax, y, 1.35)
    headers = ["Case", "Formula", "Pressure"]
    xs = [LEFT + 0.05, LEFT + 1.85, LEFT + 5.25]
    ax.plot([LEFT, PAGE_W - RIGHT], [y + 0.08, y + 0.08], transform=fig.dpi_scale_trans, color="#333", lw=0.8)
    for x, h in zip(xs, headers):
        ax.text(x, y, h, transform=fig.dpi_scale_trans, fontsize=9.5, fontweight="bold", family="DejaVu Serif", va="top")
    y -= 0.24
    ax.plot([LEFT, PAGE_W - RIGHT], [y + 0.08, y + 0.08], transform=fig.dpi_scale_trans, color="#555", lw=0.5)
    for row in rows:
        for x, cell in zip(xs, row):
            ax.text(x, y, cell, transform=fig.dpi_scale_trans, fontsize=9.0, family="DejaVu Serif", va="top")
        y -= 0.28
    ax.plot([LEFT, PAGE_W - RIGHT], [y + 0.10, y + 0.10], transform=fig.dpi_scale_trans, color="#333", lw=0.8)
    return fig, ax, y - 0.12


def main() -> None:
    with PdfPages(OUT) as pdf:
        fig, ax, y = new_page(pdf)

        ax.text(
            LEFT,
            y,
            "Derivation of the Ultrasound RBC Sensor SNR",
            transform=fig.dpi_scale_trans,
            fontsize=TITLE_SIZE,
            fontweight="bold",
            family="DejaVu Serif",
            va="top",
        )
        y -= 0.38
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "Grand Unified Theory of Imaging (GUTI)\nSelf-contained note from repository assumptions and result metadata.",
            size=10.5,
            width=92,
        )

        fig, ax, y = add_section(pdf, fig, ax, y, "Summary")
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "The ultrasound summary table reports an amplitude SNR of approximately 0.1445. This is not the SNR of a single isolated source-sensor pair. It is the RMS output amplitude of the full cone-scaled RBC pressure operator, restricted to one 2 MHz axial range slice and multiplied by a 1% CBV-variability factor, divided by the assumed scalar sensor noise.",
        )
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            r"$\mathrm{SNR}_{sensor}=\frac{0.00072268\ \mathrm{Pa}}{0.005\ \mathrm{Pa}}=0.1445$",
        )

        fig, ax, y = add_section(pdf, fig, ax, y, "Assumptions and References")
        fig, ax, y = add_text(pdf, fig, ax, y, "The numerical assumptions are taken from these local repository references:", width=92)
        fig, ax, y = add_bullets(
            pdf,
            fig,
            ax,
            y,
            [
                "RBC scaling and result metadata: guti/modalities/us/analytical.py.",
                "Summary constants and SNR calculation: scripts/make_modality_noise_summary.py.",
                "6000-sensor cone-slice 1% CBV JSON result: results/variants/us_free_field_analytical_50khz_128k_rbc_cone_slice_cbv1pct_20260602/json/001_50khz_128000src_6000sensors.json.",
                "Generated summary explanation: results/modality_correlated_noise_summary/README.md.",
            ],
            size=9.2,
        )
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            "\n".join(
                [
                    r"$f_{RBC}=2\,\mathrm{MHz},\quad f_{ref}=50\,\mathrm{kHz},\quad P_{external}=1\,\mathrm{MPa}$",
                    r"$\mathrm{BSC}_{10MHz}=3\times10^{-5}\,\mathrm{cm}^{-1}\mathrm{sr}^{-1},\quad \mathrm{CBV}=0.03$",
                    r"$V_{voxel}=24\,\mathrm{mm}^3,\quad r_{ref}=0.10\,\mathrm{m},\quad T_{in}=0.5,\quad T_{out}=0.1$",
                ]
            ),
            size=11.0,
        )
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "Important: the 24 mm^3 voxel volume is the backscatter/source-cell volume used in the 50 kHz reference simulation basis. It is not a claim that the 2 MHz image-resolution voxel has volume 24 mm^3. The later 2 MHz extrapolation uses the explicit spatial-mode scaling below.",
            width=92,
        )
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            r"$\left(\frac{f_{RBC}}{f_{ref}}\right)^3=\left(\frac{2\,\mathrm{MHz}}{50\,\mathrm{kHz}}\right)^3=64{,}000$",
        )

        fig, ax, y = add_section(pdf, fig, ax, y, "RBC Backscatter Model")
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "The blood backscatter coefficient is scaled from the 10 MHz reference value using the f^4 dependence used by the run:",
        )
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            "\n".join(
                [
                    r"$\mathrm{BSC}_{blood}(f)=\mathrm{BSC}_{10MHz}\left(\frac{f}{10\,\mathrm{MHz}}\right)^4$",
                    r"$\mathrm{BSC}_{blood}(2\,\mathrm{MHz})=3\times10^{-5}\left(\frac{2}{10}\right)^4=4.8\times10^{-8}\,\mathrm{cm}^{-1}\mathrm{sr}^{-1}$",
                ]
            ),
            size=11.0,
        )
        fig, ax, y = add_text(pdf, fig, ax, y, "Multiplying by cerebral blood volume gives:", width=92)
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            "\n".join(
                [
                    r"$\eta=\mathrm{CBV}\cdot \mathrm{BSC}_{blood}=0.03\cdot4.8\times10^{-8}$",
                    r"$\eta=1.44\times10^{-9}\,\mathrm{cm}^{-1}\mathrm{sr}^{-1}=1.44\times10^{-7}\,\mathrm{m}^{-1}\mathrm{sr}^{-1}$",
                ]
            ),
            size=11.0,
        )
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "The pairwise pressure model is:",
            width=92,
        )
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            r"$p_{ij}=P_{external}\,T_iT_j\,\frac{\sqrt{\eta V_{voxel}}}{r_{ij}}$",
        )
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "The square root appears because the BSC gives an intensity-like scattering fraction, while the channel model uses pressure amplitude.",
            width=92,
        )

        fig, ax, y = add_section(pdf, fig, ax, y, "Reference Pair Pressures")
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            "\n".join(
                [
                    r"$\frac{\sqrt{\eta V_{voxel}}}{r}=\frac{\sqrt{(1.44\times10^{-7})(24\times10^{-9})}}{0.10}=5.8788\times10^{-7}$",
                    r"$p_{no\ skull}=10^6\cdot5.8788\times10^{-7}=0.58788\ \mathrm{Pa}$",
                ]
            ),
            size=11.0,
        )
        fig, ax, y = add_table(
            pdf,
            fig,
            ax,
            y,
            [
                ("outside/outside", "0.58788 x 0.1 x 0.1", "0.0058788 Pa = 5.8788 mPa"),
                ("inside/outside", "0.58788 x 0.5 x 0.1", "0.029394 Pa = 29.394 mPa"),
                ("inside/inside", "0.58788 x 0.5 x 0.5", "0.14697 Pa = 146.97 mPa"),
            ],
        )
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "These are baseline 10 cm pair pressures before the 1% CBV-variability factor. The variable component used in the summary table is 0.01 times the pressure response.",
            width=92,
        )

        fig, ax, y = add_section(pdf, fig, ax, y, "From Pair Pressures to Table RMS")
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "The table uses the full cone-scaled operator, not one reference pair. The stored 6000-sensor SLQ metadata reports:",
            width=92,
        )
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            "\n".join(
                [
                    r"$\|G\|_F^2=372368.726,\qquad n_{outputs}=366000$",
                    r"$A_{all}=\sqrt{\frac{\|G\|_F^2}{n_{outputs}}}=\sqrt{\frac{372368.726}{366000}}=1.00866\,\mathrm{Pa}$",
                ]
            ),
            size=11.0,
        )
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "The row then restricts source power to one axial range slice for a two-cycle, 2 MHz pulse:",
            width=92,
        )
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            r"$f_{slice}=\frac{c(2/f_{RBC})/2}{D}=\frac{1540(2/2{,}000{,}000)/2}{0.15}=0.00513333$",
            size=11.0,
        )
        fig, ax, y = add_text(pdf, fig, ax, y, "The displayed table amplitude is:", width=92)
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            "\n".join(
                [
                    r"$A_{table}=A_{all}\sqrt{f_{slice}}\cdot0.01$",
                    r"$A_{table}=1.00866\sqrt{0.00513333}\cdot0.01=0.00072268\,\mathrm{Pa}=0.72268\,\mathrm{mPa}$",
                ]
            ),
            size=11.0,
        )

        fig, ax, y = add_section(pdf, fig, ax, y, "Sensor Noise and Final SNR")
        fig, ax, y = add_text(pdf, fig, ax, y, "The summary row assumes scalar iid pressure noise:", width=92)
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            r"$\sigma_{sensor}=5\,\mathrm{mPa}=0.005\,\mathrm{Pa}$",
        )
        fig, ax, y = add_math(
            pdf,
            fig,
            ax,
            y,
            r"$\mathrm{SNR}_{sensor}=\frac{A_{table}}{\sigma_{sensor}}=\frac{0.00072268}{0.005}=0.1445359$",
        )

        fig, ax, y = add_section(pdf, fig, ax, y, "Interpretation")
        fig, ax, y = add_text(
            pdf,
            fig,
            ax,
            y,
            "The number 0.1445 is a table-level RMS amplitude SNR under the current RBC-cone assumptions. It combines BSC-derived RBC pressure scaling at 2 MHz, cone-dependent skull transmission, the 50 kHz reference simulation basis using V = 24 mm^3, restriction to one two-cycle 2 MHz axial slice, a 1% CBV-variability amplitude factor, and fixed 5 mPa scalar sensor noise. It should not be read as the SNR of every individual voxel-to-sensor pair.",
            width=92,
        )

        fig, ax, y = add_section(pdf, fig, ax, y, "Repository References")
        fig, ax, y = add_bullets(
            pdf,
            fig,
            ax,
            y,
            [
                "guti/modalities/us/analytical.py",
                "scripts/make_modality_noise_summary.py",
                "results/variants/us_free_field_analytical_50khz_128k_rbc_cone_slice_cbv1pct_20260602/json/001_50khz_128000src_6000sensors.json",
                "results/modality_correlated_noise_summary/README.md",
            ],
            size=9.0,
        )

        finish_page(pdf, fig)


if __name__ == "__main__":
    main()
