"use client";

import type { CSSProperties, PointerEvent } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

interface MegNode {
  x: number;
  y: number;
  z: number;
  amp: number;
  signed: number;
  vx?: number;
  vy?: number;
  vz?: number;
  bx?: number;
  by?: number;
  bz?: number;
}

interface MegMode {
  component: number;
  singularValue: number;
  relativeGain: number;
  sources: MegNode[];
  detectors: MegNode[];
}

interface MegDataset {
  key: string;
  label: string;
  nSensors: number;
  nSources: number;
  sourceSpacingMm: number;
  sensorOffsetMm: number;
  brainRadiusMm: number;
  scalpRadiusMm: number;
  singularValues: number[];
  modes: MegMode[];
}

interface MegPayload {
  datasets: MegDataset[];
}

interface ViewState {
  azimuthDeg: number;
  elevationDeg: number;
}

const panelStyle: CSSProperties = {
  border: "1px solid #e4e4e0",
  borderRadius: 8,
  padding: 10,
  background: "#fff",
  minWidth: 0,
};

const buttonBase: CSSProperties = {
  borderWidth: 1,
  borderStyle: "solid",
  borderColor: "#d7d5cd",
  background: "#fff",
  color: "#222",
  fontFamily: "var(--sans)",
  fontSize: 12,
  lineHeight: 1,
  padding: "6px 8px",
  cursor: "pointer",
};

const activeButton: CSSProperties = {
  ...buttonBase,
  background: "#202020",
  color: "#fff",
  borderColor: "#202020",
};

const fmt = (x: number) => {
  if (Math.abs(x) >= 1e-2) return x.toFixed(3);
  return x.toExponential(2);
};

function wrapDegrees(value: number) {
  return ((((value + 180) % 360) + 360) % 360) - 180;
}

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function signedColor(value: number, alpha = 1) {
  const t = Math.min(1, Math.abs(value));
  if (value >= 0) {
    return `rgba(181, 58, 42, ${0.22 + alpha * (0.3 + 0.48 * t)})`;
  }
  return `rgba(30, 101, 136, ${0.22 + alpha * (0.3 + 0.48 * t)})`;
}

function project3d(
  point: { x: number; y: number; z: number },
  radiusMm: number,
  azimuthDeg: number,
  elevationDeg: number,
  size: number,
  margin: number,
) {
  const az = (Math.PI / 180) * azimuthDeg;
  const el = (Math.PI / 180) * elevationDeg;
  const cosAz = Math.cos(az);
  const sinAz = Math.sin(az);
  const cosEl = Math.cos(el);
  const sinEl = Math.sin(el);
  const xr = point.x * cosAz - point.y * sinAz;
  const yr = point.x * sinAz + point.y * cosAz;
  const screenY = point.z * cosEl - yr * sinEl;
  const depth = yr * cosEl + point.z * sinEl;
  const scale = (size - margin * 2) / (2.1 * radiusMm);

  return {
    x: size / 2 + xr * scale,
    y: size / 2 - screenY * scale,
    depth,
    scale,
  };
}

function shellLine(
  points: Array<{ x: number; y: number; z: number }>,
  radiusMm: number,
  azimuthDeg: number,
  elevationDeg: number,
  size: number,
  margin: number,
) {
  return points
    .map((point) => {
      const projected = project3d(point, radiusMm, azimuthDeg, elevationDeg, size, margin);
      return `${projected.x.toFixed(1)},${projected.y.toFixed(1)}`;
    })
    .join(" ");
}

function hemisphereGuides(
  radiusMm: number,
  azimuthDeg: number,
  elevationDeg: number,
  size: number,
  margin: number,
) {
  const angles = Array.from({ length: 73 }, (_, idx) => (idx / 72) * Math.PI * 2);
  const base = angles.map((theta) => ({
    x: radiusMm * Math.cos(theta),
    y: radiusMm * Math.sin(theta),
    z: 0,
  }));
  const latitude = [0.35, 0.65].map((fraction) => {
    const z = radiusMm * fraction;
    const r = Math.sqrt(radiusMm * radiusMm - z * z);
    return angles.map((theta) => ({ x: r * Math.cos(theta), y: r * Math.sin(theta), z }));
  });
  const meridians = [0, 45, 90, 135].map((deg) => {
    const phi = (Math.PI / 180) * deg;
    return Array.from({ length: 37 }, (_, idx) => {
      const theta = (idx / 36) * Math.PI;
      return {
        x: radiusMm * Math.cos(phi) * Math.cos(theta),
        y: radiusMm * Math.sin(phi) * Math.cos(theta),
        z: radiusMm * Math.sin(theta),
      };
    });
  });

  return [base, ...latitude, ...meridians].map((line) =>
    shellLine(line, radiusMm, azimuthDeg, elevationDeg, size, margin),
  );
}

function SvdPlot3D({
  title,
  nodes,
  radiusMm,
  view,
  isDragging,
  onPointerDown,
  onPointerMove,
  onPointerUp,
  vectorKeys,
  dotScale,
}: {
  title: string;
  nodes: MegNode[];
  radiusMm: number;
  view: ViewState;
  isDragging: boolean;
  onPointerDown: (event: PointerEvent<HTMLDivElement>) => void;
  onPointerMove: (event: PointerEvent<HTMLDivElement>) => void;
  onPointerUp: (event: PointerEvent<HTMLDivElement>) => void;
  vectorKeys: [keyof MegNode, keyof MegNode, keyof MegNode];
  dotScale: number;
}) {
  const margin = 14;
  const size = 320;
  const { azimuthDeg, elevationDeg } = view;
  const [xKey, yKey, zKey] = vectorKeys;

  const projectedNodes = useMemo(() => {
    const projected = nodes.map((node, idx) => ({
      node,
      idx,
      projected: project3d(node, radiusMm, azimuthDeg, elevationDeg, size, margin),
    }));
    const depths = projected.map((item) => item.projected.depth);
    const minDepth = Math.min(...depths);
    const maxDepth = Math.max(...depths);
    const span = Math.max(1e-9, maxDepth - minDepth);

    return projected
      .map((item) => ({
        ...item,
        depthT: (item.projected.depth - minDepth) / span,
      }))
      .sort((a, b) => a.projected.depth - b.projected.depth);
  }, [nodes, radiusMm, azimuthDeg, elevationDeg]);

  const maxVector = useMemo(() => {
    return Math.max(
      1e-12,
      ...nodes.map((node) => {
        const vx = Number(node[xKey] ?? 0);
        const vy = Number(node[yKey] ?? 0);
        const vz = Number(node[zKey] ?? 0);
        return Math.hypot(vx, vy, vz);
      }),
    );
  }, [nodes, xKey, yKey, zKey]);

  const guides = useMemo(
    () => hemisphereGuides(radiusMm, azimuthDeg, elevationDeg, size, margin),
    [radiusMm, azimuthDeg, elevationDeg],
  );

  return (
    <div
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      style={{
        ...panelStyle,
        cursor: isDragging ? "grabbing" : "grab",
        touchAction: "none",
        userSelect: "none",
      }}
    >
      <div style={{ fontFamily: "var(--sans)", fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
        {title}
      </div>
      <svg viewBox={`0 0 ${size} ${size}`} style={{ display: "block", width: "100%", height: "auto" }}>
        <rect x={0} y={0} width={size} height={size} fill="#fff" />
        {guides.map((points, idx) => (
          <polyline
            key={`guide-${idx}`}
            points={points}
            fill="none"
            stroke={idx === 0 ? "#d9d6cb" : "#ece9df"}
            strokeWidth={idx === 0 ? 1.1 : 0.8}
          />
        ))}
        {projectedNodes.map(({ node, idx, projected, depthT }) => {
          const r = 0.75 + node.amp * dotScale * (0.82 + depthT * 0.28);
          const opacity = 0.45 + depthT * 0.42;
          return (
            <circle
              key={`dot-${idx}`}
              cx={projected.x}
              cy={projected.y}
              r={r}
              fill={signedColor(node.signed, opacity)}
              stroke="rgba(25,25,25,0.2)"
              strokeWidth={0.25}
            />
          );
        })}
        {projectedNodes.map(({ node, idx, projected }) => {
          if (node.amp < 0.5) return null;
          const vx = Number(node[xKey] ?? 0);
          const vy = Number(node[yKey] ?? 0);
          const vz = Number(node[zKey] ?? 0);
          const end = project3d(
            {
              x: node.x + (vx / maxVector) * radiusMm * 0.12,
              y: node.y + (vy / maxVector) * radiusMm * 0.12,
              z: node.z + (vz / maxVector) * radiusMm * 0.12,
            },
            radiusMm,
            azimuthDeg,
            elevationDeg,
            size,
            margin,
          );
          const dx = end.x - projected.x;
          const dy = end.y - projected.y;
          return (
            <line
              key={`vec-${idx}`}
              x1={projected.x - dx * 0.5}
              y1={projected.y - dy * 0.5}
              x2={projected.x + dx * 0.5}
              y2={projected.y + dy * 0.5}
              stroke={signedColor(node.signed, 0.85)}
              strokeWidth={1}
              strokeLinecap="round"
            />
          );
        })}
      </svg>
    </div>
  );
}

export default function MegSVDExplorer() {
  const [payload, setPayload] = useState<MegPayload | null>(null);
  const [datasetKey, setDatasetKey] = useState("opm");
  const [component, setComponent] = useState(1);
  const [view, setView] = useState<ViewState>({ azimuthDeg: -35, elevationDeg: 24 });
  const [isDragging, setIsDragging] = useState(false);
  const dragRef = useRef<{
    pointerId: number;
    x: number;
    y: number;
    view: ViewState;
  } | null>(null);

  useEffect(() => {
    fetch("/data/meg_svd_components.json")
      .then((res) => res.json())
      .then((data: MegPayload) => setPayload(data));
  }, []);

  const dataset = payload?.datasets.find((item) => item.key === datasetKey) ?? payload?.datasets[0];
  const mode = dataset?.modes.find((item) => item.component === component) ?? dataset?.modes[0];

  useEffect(() => {
    if (dataset && !dataset.modes.some((item) => item.component === component)) {
      setComponent(dataset.modes[0]?.component ?? 1);
    }
  }, [dataset, component]);

  useEffect(() => {
    if (!dataset || dataset.modes.length < 2) return undefined;
    const interval = window.setInterval(() => {
      setComponent((current) => {
        const currentIdx = dataset.modes.findIndex((item) => item.component === current);
        const nextIdx = currentIdx >= 0 ? (currentIdx + 1) % dataset.modes.length : 0;
        return dataset.modes[nextIdx].component;
      });
    }, 3000);

    return () => window.clearInterval(interval);
  }, [dataset]);

  useEffect(() => {
    if (isDragging) return undefined;
    const interval = window.setInterval(() => {
      setView((current) => ({
        ...current,
        azimuthDeg: wrapDegrees(current.azimuthDeg + 0.28),
      }));
    }, 40);

    return () => window.clearInterval(interval);
  }, [isDragging]);

  const handlePointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      x: event.clientX,
      y: event.clientY,
      view,
    };
    setIsDragging(true);
  };

  const handlePointerMove = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = event.clientX - drag.x;
    const dy = event.clientY - drag.y;
    setView({
      azimuthDeg: wrapDegrees(drag.view.azimuthDeg - dx * 0.35),
      elevationDeg: clamp(drag.view.elevationDeg + dy * 0.28, -15, 72),
    });
  };

  const handlePointerUp = (event: PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (drag?.pointerId === event.pointerId) {
      dragRef.current = null;
      setIsDragging(false);
    }
  };

  if (!payload || !dataset || !mode) {
    return (
      <div style={{ fontFamily: "var(--sans)", fontSize: 13, color: "#666" }}>
        Loading MEG SVD components...
      </div>
    );
  }

  return (
    <div
      style={{
        fontFamily: "var(--sans)",
        fontSize: 13,
        lineHeight: 1.4,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
          marginBottom: 10,
        }}
      >
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {payload.datasets.map((item, idx) => (
            <button
              key={item.key}
              type="button"
              aria-pressed={dataset.key === item.key}
              onClick={() => setDatasetKey(item.key)}
              style={{
                ...(dataset.key === item.key ? activeButton : buttonBase),
                borderRadius: idx === 0 ? "6px 2px 2px 6px" : "2px 6px 6px 2px",
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "auto minmax(150px, 240px) auto",
            alignItems: "center",
            gap: 8,
            flex: "1 1 260px",
            maxWidth: 380,
          }}
        >
          <span style={{ color: "#555" }}>component</span>
          <input
            type="range"
            min={dataset.modes[0]?.component ?? 1}
            max={dataset.modes[dataset.modes.length - 1]?.component ?? 1}
            step={1}
            value={mode.component}
            onChange={(event) => setComponent(Number(event.target.value))}
            style={{ width: "100%" }}
          />
          <strong style={{ fontSize: 13 }}>{mode.component}</strong>
        </div>
      </div>

      <div
        style={{
          display: "flex",
          gap: 10,
          alignItems: "stretch",
          flexWrap: "wrap",
        }}
      >
        <div style={{ flex: "1 1 280px", minWidth: 250 }}>
          <SvdPlot3D
            title="Right singular vector: source pattern"
            nodes={mode.sources}
            radiusMm={dataset.brainRadiusMm}
            view={view}
            isDragging={isDragging}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            vectorKeys={["vx", "vy", "vz"]}
            dotScale={3.4}
          />
        </div>
        <div
          aria-hidden="true"
          style={{
            alignSelf: "center",
            flex: "0 0 42px",
            display: "grid",
            placeItems: "center",
          }}
        >
          <svg viewBox="0 0 48 24" style={{ width: 42, height: 24, display: "block" }}>
            <path
              d="M 4 12 H 39 M 31 5 L 40 12 L 31 19"
              fill="none"
              stroke="#9a9588"
              strokeWidth={2}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </div>
        <div style={{ flex: "1 1 280px", minWidth: 250 }}>
          <SvdPlot3D
            title="Left singular vector: detector pattern"
            nodes={mode.detectors}
            radiusMm={dataset.scalpRadiusMm + dataset.sensorOffsetMm}
            view={view}
            isDragging={isDragging}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            vectorKeys={["bx", "by", "bz"]}
            dotScale={3.6}
          />
        </div>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))",
          gap: 6,
          marginTop: 8,
          color: "#4b4b46",
        }}
      >
        <div>component {mode.component}</div>
        <div>sigma {fmt(mode.singularValue)}</div>
        <div>relative gain {(100 * mode.relativeGain).toFixed(1)}%</div>
        <div>
          {dataset.nSensors} sensors, {dataset.nSources} sources
        </div>
      </div>
    </div>
  );
}
