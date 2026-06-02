"use client";

import { useEffect, useMemo, useState } from "react";

type Tier = "today" | "fundamental";

interface Variant {
  bitrate_today: number | null;
  bitrate_fundamental: number | null;
  bitrate_physical_today: number | null;
  bitrate_physical_fundamental: number | null;
  num_sensors: number | null;
  source_spacing_mm: number | null;
  grid_resolution_mm: number | null;
  psf_fwhm_mm: number | null;
  frequency_hz: number | null;
}

interface ModalityData {
  label: string;
  noise_label_today: string;
  noise_label_fundamental: string;
  variants: Variant[];
}

interface ModalityConfig {
  key: string;
  label: string;
}

const MODALITIES: ModalityConfig[] = [
  { key: "meg_opm", label: "MEG OPM" },
  { key: "meg_squid", label: "MEG SQUID" },
  { key: "eeg_openmeeg", label: "EEG" },
  { key: "cw_fnirs", label: "fNIRS CW" },
  { key: "fmri_bold", label: "fMRI BOLD" },
  { key: "us_free_field_analytical_frequency_sweep", label: "Ultrasound" },
];

function bitrateFor(variant: Variant, tier: Tier) {
  const physical =
    tier === "today" ? variant.bitrate_physical_today : variant.bitrate_physical_fundamental;
  const fallback = tier === "today" ? variant.bitrate_today : variant.bitrate_fundamental;
  return physical ?? fallback;
}

function bestVariant(variants: Variant[], tier: Tier) {
  return variants.reduce<Variant | null>((best, variant) => {
    const value = bitrateFor(variant, tier);
    if (value === null) return best;
    if (!best) return variant;
    const bestValue = bitrateFor(best, tier);
    return bestValue === null || value > bestValue ? variant : best;
  }, null);
}

function formatBits(value: number | null) {
  if (value === null) return "—";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  if (value >= 100) return value.toFixed(0);
  if (value >= 10) return value.toFixed(1);
  return value.toFixed(2);
}

function formatConfig(variant: Variant | null) {
  if (!variant) return "—";
  const parts: string[] = [];
  if (variant.num_sensors !== null) parts.push(`${variant.num_sensors} sensors`);
  if (variant.source_spacing_mm !== null) parts.push(`${variant.source_spacing_mm} mm sources`);
  if (variant.grid_resolution_mm !== null) parts.push(`${variant.grid_resolution_mm} mm voxels`);
  if (variant.psf_fwhm_mm !== null) parts.push(`${variant.psf_fwhm_mm} mm PSF`);
  if (variant.frequency_hz !== null) {
    parts.push(`${variant.frequency_hz} Hz`);
  }
  return parts.join(", ") || "best exported sweep";
}

function formatFrequencyConfig(variant: Variant | null) {
  const text = formatConfig(variant);
  return text.replace(/(\d+(?:\.\d+)?) Hz/g, (_match, value) => {
    const hz = Number(value);
    return hz >= 1000 ? `${hz / 1000} kHz` : `${hz} Hz`;
  });
}

export default function CurrentNumbersTable() {
  const [data, setData] = useState<Record<string, ModalityData>>({});

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      MODALITIES.map(async (modality) => {
        const response = await fetch(`/data/${modality.key}.json`);
        const payload = (await response.json()) as ModalityData;
        return [modality.key, payload] as const;
      }),
    ).then((entries) => {
      if (!cancelled) setData(Object.fromEntries(entries));
    });

    return () => {
      cancelled = true;
    };
  }, []);

  const rows = useMemo(() => {
    return MODALITIES.map((modality) => {
      const modalityData = data[modality.key];
      const today = modalityData ? bestVariant(modalityData.variants, "today") : null;
      const fundamental = modalityData ? bestVariant(modalityData.variants, "fundamental") : null;
      return {
        key: modality.key,
        label: modalityData?.label ?? modality.label,
        today,
        fundamental,
        todayBits: today ? bitrateFor(today, "today") : null,
        fundamentalBits: fundamental ? bitrateFor(fundamental, "fundamental") : null,
        todayNoise: modalityData?.noise_label_today ?? "—",
        fundamentalNoise: modalityData?.noise_label_fundamental ?? "—",
      };
    });
  }, [data]);

  return (
    <div className="not-prose" style={{ margin: "1.25rem 0 1.5rem", overflowX: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontFamily: "var(--sans)",
          fontSize: 13,
          lineHeight: 1.35,
        }}
      >
        <thead>
          <tr style={{ borderBottom: "1px solid #d8d6ce", color: "#4b4b46" }}>
            <th style={{ textAlign: "left", padding: "8px 10px" }}>Modality</th>
            <th style={{ textAlign: "right", padding: "8px 10px" }}>Today bit/s</th>
            <th style={{ textAlign: "left", padding: "8px 10px" }}>Today noise floor</th>
            <th style={{ textAlign: "left", padding: "8px 10px" }}>Today config</th>
            <th style={{ textAlign: "right", padding: "8px 10px" }}>Limit bit/s</th>
            <th style={{ textAlign: "left", padding: "8px 10px" }}>Limit noise floor</th>
            <th style={{ textAlign: "left", padding: "8px 10px" }}>Limit config</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.key} style={{ borderBottom: "1px solid #ece9df" }}>
              <td style={{ padding: "8px 10px", fontWeight: 600 }}>{row.label}</td>
              <td style={{ padding: "8px 10px", textAlign: "right" }}>
                {formatBits(row.todayBits)}
              </td>
              <td style={{ padding: "8px 10px", color: "#555" }}>{row.todayNoise}</td>
              <td style={{ padding: "8px 10px", color: "#555" }}>
                {formatFrequencyConfig(row.today)}
              </td>
              <td style={{ padding: "8px 10px", textAlign: "right" }}>
                {formatBits(row.fundamentalBits)}
              </td>
              <td style={{ padding: "8px 10px", color: "#555" }}>{row.fundamentalNoise}</td>
              <td style={{ padding: "8px 10px", color: "#555" }}>
                {formatFrequencyConfig(row.fundamental)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div style={{ marginTop: 6, color: "#666", fontFamily: "var(--sans)", fontSize: 12 }}>
        Physical detector-floor mode; values are the best exported sweep point for each noise tier.
      </div>
    </div>
  );
}
