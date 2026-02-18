export const METRIC_WINDOW = 200;
export const CHART_WINDOW = 1000;
export const EVENT_WINDOW = 200;

export const shouldAcceptEvent = (lastSeenId, incomingId) => incomingId > lastSeenId;
export const hasGap = (lastSeenId, incomingId) => lastSeenId > 0 && incomingId > lastSeenId + 1;
export const boundMetric = (items) => items.slice(-METRIC_WINDOW);
export const boundChart = (items) => items.slice(-CHART_WINDOW);
export const boundEvents = (items) => items.slice(0, EVENT_WINDOW);
