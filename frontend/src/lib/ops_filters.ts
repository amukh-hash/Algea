"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";

function readFromWindow(): { asof: string; session: string } {
  if (typeof window === "undefined") return { asof: "", session: "" };
  const qp = new URLSearchParams(window.location.search);
  return { asof: qp.get("asof") ?? "", session: qp.get("session") ?? "" };
}

export function useOpsFilters() {
  const router = useRouter();
  const pathname = usePathname();
  const [state, setState] = useState<{ asof: string; session: string }>({ asof: "", session: "" });

  useEffect(() => {
    setState(readFromWindow());
  }, [pathname]);

  function setFilter(next: { asof?: string; session?: string }) {
    const current = readFromWindow();
    const qp = new URLSearchParams();
    const asof = next.asof !== undefined ? next.asof : current.asof;
    const session = next.session !== undefined ? next.session : current.session;
    if (asof) qp.set("asof", asof);
    if (session) qp.set("session", session);
    const q = qp.toString();
    router.push(`${pathname}${q ? `?${q}` : ""}`);
    setState({ asof, session });
  }

  return useMemo(() => ({ asof: state.asof, session: state.session, setFilter }), [state]);
}
