// shaders/fan_chart.vert — Fan Chart Vertex Shader
//
// Passes quantile band data and camera-relative coordinates
// to the fragment shader.
//
// Blind Spot 1 Mitigation: All coordinates are camera-relative offsets
// calculated on the CPU in double precision. Only small float32 offsets
// arrive at the GPU, preventing IEEE 754 precision collapse.
#version 450

layout(location = 0) in vec2 a_position;      // Camera-relative (x, y)
layout(location = 1) in float a_dist_median;   // Distance from P50 median
layout(location = 2) in float a_iqr_width;     // Width of the IQR band

layout(location = 0) out float v_distance_from_median;
layout(location = 1) out float v_iqr_width;

layout(set = 0, binding = 0) uniform UBO {
    mat4 projection;
    mat4 view;
    float viewport_center_x;  // CPU-calculated double → float offset
    float viewport_center_y;
} ubo;

void main() {
    v_distance_from_median = a_dist_median;
    v_iqr_width = a_iqr_width;

    gl_Position = ubo.projection * ubo.view * vec4(a_position, 0.0, 1.0);
}
