"use client";

import { useQuery } from "@tanstack/react-query";
import { controlApi, JobNode } from "@/lib/control";

const STATUS_ICONS: Record<string, string> = {
    success: "✅",
    failed: "❌",
    skipped: "⏭️",
    running: "🔄",
};

const STATUS_BORDER: Record<string, string> = {
    success: "border-green-500/50",
    failed: "border-red-500/50",
    skipped: "border-yellow-500/30",
    running: "border-blue-500/50",
};

function timeAgo(iso: string | null): string {
    if (!iso) return "—";
    const diff = Date.now() - new Date(iso).getTime();
    if (diff < 60_000) return `${Math.round(diff / 1000)}s ago`;
    if (diff < 3_600_000) return `${Math.round(diff / 60_000)}m ago`;
    return `${Math.round(diff / 3_600_000)}h ago`;
}

function JobCard({ job, allJobs }: { job: JobNode; allJobs: JobNode[] }) {
    const border = STATUS_BORDER[job.last_status ?? ""] ?? "border-border";
    const icon = STATUS_ICONS[job.last_status ?? ""] ?? "•";

    return (
        <div className={`rounded-lg border-2 ${border} bg-surface-2/50 p-3 text-xs min-w-[180px]`}>
            <div className="flex items-center gap-2 mb-1">
                <span>{icon}</span>
                <span className="font-mono font-semibold">{job.name}</span>
            </div>
            <div className="space-y-0.5 text-secondary">
                <div>Sessions: {job.sessions.join(", ")}</div>
                {job.last_duration_s != null && <div>Duration: {job.last_duration_s}s</div>}
                {job.last_started && <div>Last run: {timeAgo(job.last_started)}</div>}
                {job.last_error && <div className="text-red-400 truncate" title={job.last_error}>⚠ {job.last_error}</div>}
                {job.min_interval_s > 0 && <div>Cooldown: {job.min_interval_s}s</div>}
                {job.deps.length > 0 && (
                    <div className="text-muted">Deps: {job.deps.join(" → ")}</div>
                )}
            </div>
        </div>
    );
}

export function JobDAG() {
    const { data, isLoading } = useQuery({
        queryKey: ["job-graph"],
        queryFn: controlApi.getJobGraph,
        refetchInterval: 10_000,
    });

    if (isLoading) return <div className="animate-pulse h-40 rounded-md bg-surface-2" />;
    const jobs = data?.jobs ?? [];

    // Group into layers by dependency depth
    const depthMap: Record<string, number> = {};
    function getDepth(name: string, visited: Set<string> = new Set()): number {
        if (depthMap[name] !== undefined) return depthMap[name];
        if (visited.has(name)) return 0;
        visited.add(name);
        const job = jobs.find((j) => j.name === name);
        if (!job || job.deps.length === 0) {
            depthMap[name] = 0;
            return 0;
        }
        const maxParent = Math.max(...job.deps.map((d) => getDepth(d, visited)));
        depthMap[name] = maxParent + 1;
        return depthMap[name];
    }

    jobs.forEach((j) => getDepth(j.name));
    const maxDepth = Math.max(0, ...Object.values(depthMap));
    const layers: JobNode[][] = [];
    for (let d = 0; d <= maxDepth; d++) {
        layers.push(jobs.filter((j) => depthMap[j.name] === d));
    }

    return (
        <div className="space-y-3">
            <div className="text-sm font-semibold text-secondary">Job Pipeline</div>
            <div className="flex gap-4 overflow-x-auto pb-2">
                {layers.map((layer, i) => (
                    <div key={i} className="flex flex-col gap-2 items-center">
                        <div className="text-[0.6rem] text-muted uppercase tracking-widest">Layer {i}</div>
                        {layer.map((job) => (
                            <JobCard key={job.name} job={job} allJobs={jobs} />
                        ))}
                        {i < layers.length - 1 && (
                            <div className="text-muted text-lg">→</div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
}
