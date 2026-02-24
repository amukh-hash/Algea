"use client";

import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { controlApi } from "@/lib/control";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { Card } from "@/components/ui/primitives";
import { useToasts } from "@/components/ui/ToastProvider";

export function FlattenControls() {
    const { addToast } = useToasts();
    const qc = useQueryClient();
    const [confirmOpen, setConfirmOpen] = useState(false);
    const [flattenTarget, setFlattenTarget] = useState<string | undefined>();

    const flattenMut = useMutation({
        mutationFn: (sleeve?: string) => controlApi.flatten(sleeve),
        onSuccess: () => {
            qc.invalidateQueries({ queryKey: ["orch-positions"] });
            addToast({
                type: "success",
                title: "Flatten Submitted",
                description: `${flattenTarget ? `Flatten ${flattenTarget}` : "Flatten ALL"} order queued.`,
            });
            setConfirmOpen(false);
        },
        onError: (err: Error) => {
            addToast({ type: "error", title: "Flatten Failed", description: err.message });
            setConfirmOpen(false);
        },
    });

    const triggerFlatten = (sleeve?: string) => {
        setFlattenTarget(sleeve);
        setConfirmOpen(true);
    };

    return (
        <>
            <Card className="border-red-900/50 bg-red-950/10">
                <h2 className="mb-3 text-sm font-semibold text-red-400 flex items-center gap-2">
                    <span>🚨</span>
                    <span>Emergency Controls</span>
                </h2>
                <div className="flex flex-wrap gap-2">
                    <button
                        onClick={() => triggerFlatten()}
                        className="rounded bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 transition-colors"
                    >
                        Flatten ALL
                    </button>
                    <button
                        onClick={() => triggerFlatten("core")}
                        className="rounded border border-red-600 px-3 py-2 text-xs text-red-300 hover:bg-red-900/30 transition-colors"
                    >
                        Flatten Core
                    </button>
                    <button
                        onClick={() => triggerFlatten("vrp")}
                        className="rounded border border-red-600 px-3 py-2 text-xs text-red-300 hover:bg-red-900/30 transition-colors"
                    >
                        Flatten VRP
                    </button>
                    <button
                        onClick={() => triggerFlatten("selector")}
                        className="rounded border border-red-600 px-3 py-2 text-xs text-red-300 hover:bg-red-900/30 transition-colors"
                    >
                        Flatten Selector
                    </button>
                </div>
            </Card>

            <ConfirmDialog
                open={confirmOpen}
                title={flattenTarget ? `Flatten ${flattenTarget}` : "Flatten ALL Positions"}
                message={
                    <div>
                        <p>This will close {flattenTarget ? `all positions in the <strong>${flattenTarget}</strong> sleeve` : "<strong>every open position</strong> across all sleeves"}.</p>
                        <p className="mt-2 text-amber-400">This action cannot be undone.</p>
                    </div>
                }
                confirmLabel={flattenTarget ? `Flatten ${flattenTarget}` : "Flatten ALL"}
                danger
                onConfirm={() => flattenMut.mutate(flattenTarget)}
                onCancel={() => setConfirmOpen(false)}
            />
        </>
    );
}
