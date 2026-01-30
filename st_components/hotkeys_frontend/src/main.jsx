import React, { useEffect } from "react";
import ReactDOM from "react-dom/client";
import { Streamlit, withStreamlitConnection } from "streamlit-component-lib";

function HotkeysBase(props) {
  const args = props.args || {};
  const enabled = Boolean(args.enabled ?? true);
  const allowed = Array.isArray(args.allowed) ? args.allowed.map(String) : [];

  useEffect(() => {
    Streamlit.setFrameHeight(0);

    if (!enabled) return;

    const handler = (e) => {
      if (e.repeat) return;

      const k = String(e.key || "");
      if (allowed.length && !allowed.includes(k)) return;

      Streamlit.setComponentValue({
        key: k,
        code: String(e.code || ""),
        ts: Date.now(),
      });
    };

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [enabled, allowed.join("|")]);

  return null;
}

const Hotkeys = withStreamlitConnection(HotkeysBase);

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Hotkeys />
  </React.StrictMode>
);
