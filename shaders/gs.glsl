#version 330 core

layout (lines_adjacency) in;
layout (line_strip, max_vertices = 101) out;

void main() {
    vec4 a = gl_in[0].gl_Position;
    vec4 c_1 = gl_in[1].gl_Position;
    vec4 c_2 = gl_in[2].gl_Position;
    vec4 b = gl_in[3].gl_Position;
    float t = 0;

    for (int i = 0; i < 101; i++) {
        // calculate berstein form
        t = i/100.0;
        gl_Position = a * (-pow(t, 3) + 3*pow(t,2) - 3*t + 1)
            + c_1 * (3 * pow(t,3) - 6 * pow(t,2) + 3 * t)
            + c_2 * (-3 * pow(t,3) + 3 * pow(t,2))
            + b * pow(t,3);
        EmitVertex();
    }

    EndPrimitive();
}
