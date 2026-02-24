"use client";

import { ReactNode } from "react";

interface Props {
    open: boolean;
    title: string;
    message: string | ReactNode;
    confirmLabel?: string;
    cancelLabel?: string;
    danger?: boolean;
    onConfirm: () => void;
    onCancel: () => void;
}

export function ConfirmDialog({ open, title, message, confirmLabel = "Confirm", cancelLabel = "Cancel", danger = false, onConfirm, onCancel }: Props) {
    if (!open) return null;
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm" onClick={onCancel}>
            <div className="w-full max-w-md rounded-lg border border-border bg-surface-1 p-6 shadow-2xl" onClick={(e) => e.stopPropagation()}>
                <h3 className="text-lg font-semibold mb-2">{title}</h3>
                <div className="text-sm text-secondary mb-6">{message}</div>
                <div className="flex justify-end gap-3">
                    <button onClick={onCancel} className="rounded px-4 py-2 text-sm border border-border hover:bg-surface-2 transition-colors">
                        {cancelLabel}
                    </button>
                    <button
                        onClick={onConfirm}
                        className={`rounded px-4 py-2 text-sm font-semibold transition-colors ${danger
                                ? "bg-red-600 text-white hover:bg-red-700"
                                : "bg-primary text-primary-foreground hover:bg-primary/90"
                            }`}
                    >
                        {confirmLabel}
                    </button>
                </div>
            </div>
        </div>
    );
}
