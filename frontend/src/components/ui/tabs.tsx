"use client";

import { ReactNode, useId, useState } from "react";
import { cn } from "./primitives";

export function Tabs({ items, defaultTab }: { items: { id: string; label: string; panel: ReactNode }[]; defaultTab?: string }) {
  const [active, setActive] = useState(defaultTab ?? items[0]?.id);
  const [focused, setFocused] = useState(active);
  const uid = useId();
  const idx = items.findIndex((i) => i.id === focused);

  return (
    <div>
      <div role="tablist" aria-label="Run detail tabs" className="mb-4 flex gap-2 border-b border-border">
        {items.map((item) => {
          const selected = active === item.id;
          return (
            <button
              key={item.id}
              id={`${uid}-${item.id}-tab`}
              role="tab"
              tabIndex={focused === item.id ? 0 : -1}
              aria-selected={selected}
              aria-controls={`${uid}-${item.id}-panel`}
              className={cn("rounded-t-md px-3 py-2 text-sm", selected ? "bg-surface-2 text-primary" : "text-secondary")}
              onFocus={() => setFocused(item.id)}
              onClick={() => setActive(item.id)}
              onKeyDown={(e) => {
                if (e.key === "ArrowRight") setFocused(items[(idx + 1) % items.length].id);
                if (e.key === "ArrowLeft") setFocused(items[(idx - 1 + items.length) % items.length].id);
                if (e.key === "Home") setFocused(items[0].id);
                if (e.key === "End") setFocused(items[items.length - 1].id);
                if (e.key === "Enter" || e.key === " ") setActive(focused);
              }}
            >
              {item.label}
            </button>
          );
        })}
      </div>
      {items.map((item) => (
        <section key={item.id} id={`${uid}-${item.id}-panel`} role="tabpanel" aria-labelledby={`${uid}-${item.id}-tab`} hidden={active !== item.id}>
          {active === item.id ? item.panel : null}
        </section>
      ))}
    </div>
  );
}
