#version 460

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// layout(std430, set = 0, binding = 0) buffer InData {
//     uint index;
// } ub;

layout(set = 0, binding = 0, rgba8) uniform writeonly image2D Data;

vec3 hsv_to_rgb (vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

int from_decimal (float n) {
    return int(mod(n, 1) * 255);
}

void write_data(vec3 res) {
    imageStore(Data, ivec2(gl_GlobalInvocationID.xy), vec4(res, 1.0));
}

void main() {
    vec2 cords = (gl_GlobalInvocationID.xy + vec2(0.5)) / vec2(imageSize(Data));
    vec2 c = (cords - vec2(0.5)) * 4.0;
    vec2 z = vec2(0.0, 0.0);
    float i;
    float maxi = 1000.0;
    float added = 1.0 / maxi;

    vec3 res;

    if (length(c) > 2.0) {
        write_data(vec3(1.0, 1.0, 1.0));
        return;
    }

    for (i = 0.0; i < 1.0; i += added) {
        z = vec2(
            z.x * z.x - z.y * z.y + c.x,
            z.y * z.x + z.x * z.y + c.y
        );

        if (length(z) > 2.0) {
            write_data(hsv_to_rgb(vec3(
                mod(i * maxi * 12.0 / 360.0, 1.0),
                1.0,
                1.0
            )));
            return;
        }
    }

    write_data(vec3(0.0, 0.0, 0.0));
}
