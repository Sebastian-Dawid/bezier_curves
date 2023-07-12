extern crate glad_gl;
extern crate glfw;

use std::sync::mpsc::Receiver;
use std::{ops, usize};
use glad_gl::gl;
use glfw::{Action, Context, Key, Glfw};
use std::ffi::CString;
use std::mem::size_of;
use std::os::raw::{c_char, c_void};

#[derive(Debug)]
struct Shader {
    id: u32,
}

impl Shader {
    fn new(vertex_src: &str, fragment_src: &str) -> Shader {
        let prog_id: u32;
        let vs_str: CString = CString::new(vertex_src).unwrap();
        let vs_src: *const c_char = vs_str.as_ptr() as *const c_char;
        let fs_str: CString = CString::new(fragment_src).unwrap();
        let fs_src: *const c_char = fs_str.as_ptr() as *const c_char;

        unsafe {
            let mut success: i32 = 0;
            let mut len: i32 = 0;

            let vs: u32 = gl::CreateShader(gl::VERTEX_SHADER);
            gl::ShaderSource(vs, 1, &vs_src, std::ptr::null());
            gl::CompileShader(vs);

            gl::GetShaderiv(vs, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                gl::GetShaderiv(vs, gl::INFO_LOG_LENGTH, &mut len);
                let mut buffer: Vec<u8> = Vec::with_capacity(len as usize + 1);
                buffer.extend([b' '].iter().cycle().take(len as usize));
                let error: CString = CString::from_vec_unchecked(buffer); // this seems like a VERY
                                                                          // BAD idea
                gl::GetShaderInfoLog(vs, len, std::ptr::null_mut(), error.as_ptr() as *mut c_char);
                println!("{}", error.to_string_lossy());
            }

            let fs: u32 = gl::CreateShader(gl::FRAGMENT_SHADER);
            gl::ShaderSource(fs, 1, &fs_src, std::ptr::null());
            gl::CompileShader(fs);

            gl::GetShaderiv(fs, gl::COMPILE_STATUS, &mut success);
            if success == 0 {
                gl::GetShaderiv(fs, gl::INFO_LOG_LENGTH, &mut len);
                let mut buffer: Vec<u8> = Vec::with_capacity(len as usize + 1);
                buffer.extend([b' '].iter().cycle().take(len as usize));
                let error: CString = CString::from_vec_unchecked(buffer);
                gl::GetShaderInfoLog(fs, len, std::ptr::null_mut(), error.as_ptr() as *mut c_char);
                println!("{}", error.to_string_lossy());
            }

            prog_id = gl::CreateProgram();
            gl::AttachShader(prog_id, vs);
            gl::AttachShader(prog_id, fs);

            gl::LinkProgram(prog_id);

            gl::GetProgramiv(prog_id, gl::LINK_STATUS, &mut success);
            if success == 0 {
                gl::GetProgramiv(prog_id, gl::INFO_LOG_LENGTH, &mut len);
                let mut buffer: Vec<u8> = Vec::with_capacity(len as usize + 1);
                buffer.extend([b' '].iter().cycle().take(len as usize));
                let error: CString = CString::from_vec_unchecked(buffer);
                gl::GetProgramInfoLog(prog_id, len, std::ptr::null_mut(), error.as_ptr() as *mut c_char);
                println!("{}", error.to_string_lossy());
            }

            gl::DeleteShader(vs);
            gl::DeleteShader(fs);
        }

        return Shader { id: prog_id };
    }

    fn _use(&self) {
        unsafe {
            gl::UseProgram(self.id);
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}

impl ops::Add<Vertex> for Vertex {
    type Output = Vertex;

    fn add(self, _rhs: Vertex) -> Vertex {
        Vertex {
            position: [
                self.position[0] + _rhs.position[0],
                self.position[1] + _rhs.position[1],
            ],
        }
    }
}

impl ops::Mul<f32> for Vertex {
    type Output = Vertex;

    fn mul(self, _rhs: f32) -> Vertex {
        Vertex {
            position: [_rhs * self.position[0], _rhs * self.position[1]],
        }
    }
}

impl ops::Mul<Vertex> for f32 {
    type Output = Vertex;

    fn mul(self, _rhs: Vertex) -> Vertex {
        Vertex {
            position: [_rhs.position[0] * self, _rhs.position[1] * self],
        }
    }
}

fn lerp(a: Vertex, b: Vertex, t: f32) -> Vertex {
    (1 as f32 - t) * a + t * b
}

#[derive(Debug)]
struct CubicBezier {
    a: Vertex,
    c_1: Vertex,
    c_2: Vertex,
    b: Vertex,
}

impl CubicBezier {
    fn compute(&self, samples: u32) -> Vec<Vertex> {
        let mut res: Vec<Vertex> = vec![];
        for i in 0..=samples {
            let t: f32 = (1 as f32/samples as f32) * i as f32;

            let s_1: Vertex = lerp(self.a.clone(), self.c_1.clone(), t);
            let s_2: Vertex = lerp(self.c_1.clone(), self.c_2.clone(), t);
            let s_3: Vertex = lerp(self.c_2.clone(), self.b.clone(), t);

            let t_1: Vertex = lerp(s_1, s_2.clone(), t);
            let t_2: Vertex = lerp(s_2, s_3, t);

            res.push(lerp(t_1, t_2, t));
        }
        res
    }
}

fn handle_window_event(window: &mut glfw::Window, event: glfw::WindowEvent) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
        _ => {}
    }
}

fn main() {
    let vertex_src: String = std::fs::read_to_string("shaders/vs.glsl").expect("Unable to read file.");
    let fragment_src: String = std::fs::read_to_string("shaders/fs.glsl").expect("Unable to read file.");

    let mut glfw: Glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

    let (mut window, events): (glfw::Window, Receiver<(f64, glfw::WindowEvent)>) = glfw.create_window(300, 300, "Bezier Curves", glfw::WindowMode::Windowed)
                               .expect("Failed to create Window.");
    
    window.set_key_polling(true);
    window.make_current();

    gl::load(|e| glfw.get_proc_address_raw(e) as *const c_void);

    let shader: Shader = Shader::new(&vertex_src, &fragment_src);
    shader._use();

    let mut vao = 0;
    let mut vbo = 0;

    let a: Vertex = Vertex {
        position: [0.0, 0.0],
    };
    let c_1: Vertex = Vertex {
        position: [0.0, 1.0],
    };
    let c_2: Vertex = Vertex {
        position: [1.0, 1.0],
    };
    let b: Vertex = Vertex {
        position: [1.0, 0.0],
    };
    let bezier: CubicBezier = CubicBezier { a: a, c_1: c_1, c_2: c_2, b: b };
    let curve: Vec<f32> = bezier.compute(100)
        .iter()
        .map(|e| e.position)
        .fold(vec![], |mut acc: Vec<f32>, e: [f32; 2]| {
            let v;
            acc.extend_from_slice(&e);
            v = acc;
            return v
        });
    
    unsafe {
        gl::GenVertexArrays(1, &mut vao);
        gl::GenBuffers(1, &mut vbo);

        gl::BindVertexArray(vao);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(gl::ARRAY_BUFFER, (curve.len() * size_of::<f32>()) as isize, curve.as_ptr() as *const c_void, gl::STATIC_DRAW);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, (2 * size_of::<f32>()) as i32, 0 as *const c_void);
        gl::EnableVertexAttribArray(0);
    }

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, event);
        }

        unsafe {
            gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);

            shader._use();

            gl::BindVertexArray(vao);
            gl::DrawArrays(gl::LINE_STRIP, 0, (curve.len() / 2) as i32);
        }

        window.swap_buffers();
    }
}
