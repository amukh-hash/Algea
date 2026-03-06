// shaders/fan_chart.frag — Probabilistic Fan Chart Fragment Shader
//
// Compiled to SPIR-V via: glslangValidator -V fan_chart.frag -o fan_chart.frag.spv
//
// Renders expanding IQR bands from Chronos-2 output.
// Alpha interpolated geometrically based on distance from median line.
// Execution thresholds at ±0.02 rendered as dashed red lines.
#version 450

layout(location = 0) in float v_distance_from_median;
layout(location = 1) in float v_iqr_width;

layout(location = 0) out vec4 fragColor;

void main() {
    // Interpolate alpha: full opacity at median, transparent at IQR edges
    float alpha = 1.0 - smoothstep(0.0, v_iqr_width, abs(v_distance_from_median));

    // Strict execution threshold boundary markers (+/- 0.02)
    // Rendered as dashed red lines at the 98th percentile boundary
    if (v_distance_from_median > 0.98 && mod(gl_FragCoord.x, 10.0) < 5.0) {
        fragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red dashed limit
        return;
    }

    // Base color: cool blue with alpha-driven uncertainty visualization
    fragColor = vec4(0.1, 0.4, 0.9, max(alpha, 0.05));
}
