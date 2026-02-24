"use client";

import { ReactNode } from "react";
import { Button } from "@/components/ui/primitives";

interface TearOffWrapperProps {
    id: string;
    title: string;
    children: ReactNode;
}

export function TearOffWrapper({ id, title, children }: TearOffWrapperProps) {
    const popOut = () => {
        // Open a new window that only renders this component id
        window.open(`/tearoff/${id}`, `_blank_${id}`, "width=800,height=600,menubar=no,toolbar=no,location=no,status=no");
    };

    return (
        <div className="group relative">
            <div className="absolute right-2 top-2 z-10 opacity-0 transition-opacity group-hover:opacity-100">
                <Button onClick={popOut} className="bg-surface-2/80 text-xs backdrop-blur">
                    ⇱ Pop Out
                </Button>
            </div>
            <div>{children}</div>
        </div>
    );
}
