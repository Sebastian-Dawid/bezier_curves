#version 330 core
layout (location = 0) in vec2 pos;
layout (location = 1) in vec4 in_id;

out vec4 ID;

uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(pos, 0, 1);
    ID = in_id;
}
