#version 330 core
out vec4 FragColor;

void main() {
    if (length(gl_PointCoord - vec2(0.5)) > 0.5)
        discard;
    if (length(gl_PointCoord - vec2(0.5)) < 0.3)
        FragColor = vec4(1, 1, 1, 1);
    else
        FragColor = vec4(0, 0, 0, 1);
}
