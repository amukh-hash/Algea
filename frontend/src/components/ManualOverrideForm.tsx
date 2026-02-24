"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { controlApi, ManualOrder } from "@/lib/control";
import { Card } from "@/components/ui/primitives";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { useToasts } from "@/components/ui/ToastProvider";

export function ManualOverrideForm() {
    const { addToast } = useToasts();
    const qc = useQueryClient();
    const [symbol, setSymbol] = useState("");
    const [qty, setQty] = useState("");
    const [side, setSide] = useState<"buy" | "sell">("buy");
    const [orderType, setOrderType] = useState<"MKT" | "LMT" | "MOC">("MKT");
    const [limitPrice, setLimitPrice] = useState("");
    const [confirmOpen, setConfirmOpen] = useState(false);

    const submitMut = useMutation({
        mutationFn: (order: ManualOrder) => controlApi.submitManualOrder(order),
        onSuccess: () => {
            addToast({ type: "success", title: "Order Submitted", description: `${side.toUpperCase()} ${qty} ${symbol.toUpperCase()} @ ${orderType}` });
            qc.invalidateQueries({ queryKey: ["control-state"] });
            setSymbol("");
            setQty("");
            setLimitPrice("");
            setConfirmOpen(false);
        },
        onError: (err: Error) => {
            addToast({ type: "error", title: "Order Failed", description: err.message });
            setConfirmOpen(false);
        },
    });

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (!symbol.trim() || !qty || parseInt(qty) <= 0) {
            addToast({ type: "error", title: "Invalid Order", description: "Symbol and positive quantity required." });
            return;
        }
        setConfirmOpen(true);
    };

    const confirmSubmit = () => {
        const order: ManualOrder = {
            symbol: symbol.trim().toUpperCase(),
            qty: parseInt(qty),
            side,
            order_type: orderType,
            limit_price: orderType === "LMT" && limitPrice ? parseFloat(limitPrice) : null,
        };
        submitMut.mutate(order);
    };

    return (
        <>
            <Card className="border-red-900/50 bg-red-950/10">
                <h2 className="mb-3 text-sm font-semibold text-red-400 flex items-center justify-between">
                    <span>⚠ Emergency Execution Override</span>
                </h2>
                <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-3 text-sm">
                    <div>
                        <label className="block text-xs text-muted mb-1">Symbol</label>
                        <input
                            type="text"
                            value={symbol}
                            onChange={(e) => setSymbol(e.target.value)}
                            placeholder="SPY"
                            className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm w-24 focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-muted mb-1">Qty</label>
                        <input
                            type="number"
                            value={qty}
                            onChange={(e) => setQty(e.target.value)}
                            placeholder="10"
                            min="1"
                            className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm w-20 focus:outline-none focus:ring-1 focus:ring-primary"
                        />
                    </div>
                    <div>
                        <label className="block text-xs text-muted mb-1">Side</label>
                        <select
                            value={side}
                            onChange={(e) => setSide(e.target.value as "buy" | "sell")}
                            className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                        >
                            <option value="buy">BUY</option>
                            <option value="sell">SELL</option>
                        </select>
                    </div>
                    <div>
                        <label className="block text-xs text-muted mb-1">Type</label>
                        <select
                            value={orderType}
                            onChange={(e) => setOrderType(e.target.value as "MKT" | "LMT" | "MOC")}
                            className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm focus:outline-none focus:ring-1 focus:ring-primary"
                        >
                            <option value="MKT">MARKET</option>
                            <option value="LMT">LIMIT</option>
                            <option value="MOC">MOC</option>
                        </select>
                    </div>
                    {orderType === "LMT" && (
                        <div>
                            <label className="block text-xs text-muted mb-1">Limit $</label>
                            <input
                                type="number"
                                step="0.01"
                                value={limitPrice}
                                onChange={(e) => setLimitPrice(e.target.value)}
                                placeholder="520.00"
                                className="rounded border border-border bg-surface-2 px-2 py-1.5 text-sm w-24 focus:outline-none focus:ring-1 focus:ring-primary"
                            />
                        </div>
                    )}
                    <button
                        type="submit"
                        disabled={submitMut.isPending}
                        className="rounded bg-red-600 px-4 py-1.5 text-sm font-semibold text-white hover:bg-red-700 transition-colors disabled:opacity-50"
                    >
                        {submitMut.isPending ? "Submitting..." : "Submit Override"}
                    </button>
                </form>
            </Card>

            <ConfirmDialog
                open={confirmOpen}
                title="Confirm Manual Order"
                message={
                    <div className="space-y-2">
                        <p>You are about to submit:</p>
                        <div className="rounded bg-surface-2 p-3 font-mono text-sm">
                            {side.toUpperCase()} {qty} {symbol.toUpperCase()} @ {orderType}
                            {orderType === "LMT" && limitPrice ? ` $${limitPrice}` : ""}
                        </div>
                        <p className="text-amber-400 text-xs">This bypasses normal orchestrator risk checks.</p>
                    </div>
                }
                confirmLabel="Submit Order"
                danger
                onConfirm={confirmSubmit}
                onCancel={() => setConfirmOpen(false)}
            />
        </>
    );
}
