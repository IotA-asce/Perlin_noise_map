import React, { useEffect, useMemo, useRef, useState } from "react";
import ReactDOM from "react-dom/client";
import {
  Streamlit,
  withStreamlitConnection,
} from "streamlit-component-lib";

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function LiveSliderBase(props) {
  const args = props.args || {};
  const label = String(args.label ?? "");
  const minValue = Number(args.min_value ?? 0);
  const maxValue = Number(args.max_value ?? 1);
  const step = Number(args.step ?? 0.01);
  const throttleMs = Number(args.throttle_ms ?? 60);

  const initial = clamp(Number(args.value ?? minValue), minValue, maxValue);
  const [val, setVal] = useState(initial);
  const lastSent = useRef(0);

  const send = (v, isFinal) => {
    Streamlit.setComponentValue({ value: v, is_final: isFinal });
  };

  const maybeSend = (v, isFinal) => {
    const now = Date.now();
    if (isFinal || now - lastSent.current >= throttleMs) {
      lastSent.current = now;
      send(v, isFinal);
    }
  };

  useEffect(() => {
    Streamlit.setFrameHeight();
  }, [label]);

  // If Python pushes a new value, reflect it.
  const externalValue = useMemo(() => {
    return clamp(Number(args.value ?? minValue), minValue, maxValue);
  }, [args.value, minValue, maxValue]);

  useEffect(() => {
    setVal(externalValue);
  }, [externalValue]);

  return (
    <div
      style={{
        fontFamily: "inherit",
        width: "100%",
      }}
    >
      <div
        style={{
          fontSize: "0.85rem",
          opacity: 0.75,
          marginBottom: 6,
          display: "flex",
          justifyContent: "space-between",
        }}
      >
        <span>{label}</span>
        <span style={{ fontVariantNumeric: "tabular-nums" }}>{val.toFixed(3)}</span>
      </div>
      <input
        type="range"
        min={minValue}
        max={maxValue}
        step={step}
        value={val}
        style={{ width: "100%" }}
        onInput={(e) => {
          const v = clamp(Number(e.target.value), minValue, maxValue);
          setVal(v);
          maybeSend(v, false);
        }}
        onChange={(e) => {
          const v = clamp(Number(e.target.value), minValue, maxValue);
          setVal(v);
          maybeSend(v, true);
        }}
        onMouseUp={() => {
          maybeSend(val, true);
        }}
        onTouchEnd={() => {
          maybeSend(val, true);
        }}
      />
    </div>
  );
}

const LiveSlider = withStreamlitConnection(LiveSliderBase);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <LiveSlider />
  </React.StrictMode>
);
