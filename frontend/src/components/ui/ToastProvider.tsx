"use client";

import React, { createContext, useContext, useState, useCallback, ReactNode } from "react";

export type ToastType = "info" | "success" | "warning" | "error";

export interface ToastProps {
    id: string;
    title: string;
    description?: string;
    type: ToastType;
    action?: {
        label: string;
        onClick: () => void;
    };
}

interface ToastContextValue {
    toasts: ToastProps[];
    addToast: (toast: Omit<ToastProps, "id">) => void;
    removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export const useToasts = () => {
    const context = useContext(ToastContext);
    if (!context) throw new Error("useToasts must be used inside ToastProvider");
    return context;
};

// Global event bus for non-React contexts (like fetchWithTimeout)
export const globalToastBus = {
    addToast: (toast: Omit<ToastProps, "id">) => { },
};

export const ToastProvider = ({ children }: { children: ReactNode }) => {
    const [toasts, setToasts] = useState<ToastProps[]>([]);

    const addToast = useCallback((toast: Omit<ToastProps, "id">) => {
        const id = Math.random().toString(36).substring(2, 9);
        setToasts((prev) => [...prev, { ...toast, id }]);
        setTimeout(() => {
            setToasts((prev) => prev.filter((t) => t.id !== id));
        }, 5000); // 5 sec auto dismiss
    }, []);

    const removeToast = useCallback((id: string) => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
    }, []);

    // Wire up the global event bus
    React.useEffect(() => {
        globalToastBus.addToast = addToast;
        return () => {
            globalToastBus.addToast = () => { };
        };
    }, [addToast]);

    return (
        <ToastContext.Provider value={{ toasts, addToast, removeToast }}>
            {children}
            <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2">
                {toasts.map((toast) => (
                    <div
                        key={toast.id}
                        className={`flex flex-col gap-1 rounded-md border p-4 shadow-lg w-80 translate-y-0 transition-all ${toast.type === "error"
                                ? "bg-red-950 border-red-800 text-red-100"
                                : toast.type === "warning"
                                    ? "bg-orange-950 border-orange-800 text-orange-100"
                                    : toast.type === "success"
                                        ? "bg-green-950 border-green-800 text-green-100"
                                        : "bg-surface-2 border-border text-primary"
                            }`}
                    >
                        <div className="flex justify-between items-start gap-2">
                            <strong className="text-sm font-semibold">{toast.title}</strong>
                            <button
                                onClick={() => removeToast(toast.id)}
                                className="text-xs opacity-50 hover:opacity-100"
                            >
                                ✕
                            </button>
                        </div>
                        {toast.description && <p className="text-xs opacity-80">{toast.description}</p>}
                        {toast.action && (
                            <button
                                className="mt-2 self-start rounded bg-white/10 px-2 py-1 text-xs hover:bg-white/20"
                                onClick={() => {
                                    toast.action!.onClick();
                                    removeToast(toast.id);
                                }}
                            >
                                {toast.action.label}
                            </button>
                        )}
                    </div>
                ))}
            </div>
        </ToastContext.Provider>
    );
};
