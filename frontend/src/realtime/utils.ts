import { CHART_WINDOW, EVENT_WINDOW, METRIC_WINDOW } from "./useRunStream";

export function shouldAcceptEvent(lastSeenId: number, incomingId: number) {
  return incomingId > lastSeenId;
}

export function hasGap(lastSeenId: number, incomingId: number) {
  return lastSeenId > 0 && incomingId > lastSeenId + 1;
}

export function boundMetric<T>(items: T[]) { return items.slice(-METRIC_WINDOW); }
export function boundChart<T>(items: T[]) { return items.slice(-CHART_WINDOW); }
export function boundEvents<T>(items: T[]) { return items.slice(0, EVENT_WINDOW); }
