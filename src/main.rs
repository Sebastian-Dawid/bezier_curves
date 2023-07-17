extern crate glad_gl;
extern crate glfw;
extern crate nalgebra_glm as glm;

use std::sync::mpsc::Receiver;
use std::{ops, usize};
use glad_gl::gl;
use glfw::{Action, Context, Key, Glfw, MouseButton};
use std::ffi::CString;
use std::mem::size_of;
use std::os::raw::{c_char, c_void};

/// A shader object containing the shader programs ID.
#[derive(Debug)]
struct Shader {
    /// The ID of the shader program.
    id: u32,
}

impl Shader {
    /// Given a vertex and fragment shaders source this function compiles and links a new shader
    /// program with the two given shaders attached to it.
    ///
    /// # Arguments
    ///
    /// * `vertex_src` - The source code of the vertex shader to attach as a string slice.
    /// * `fragment_src` - The source code of the fragment shader to attach as a string slice.
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
                let error: CString = CString::from_vec_unchecked(buffer);
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

    /// Sets the shader program currently in use to this shader program.
    fn _use(&self) {
        unsafe {
            gl::UseProgram(self.id);
        }
    }
}

/// A 2-Dimensional Vertex with elements of type `f32`.
#[derive(Debug, Copy, Clone)]
struct Vertex {
    /// A `[f32; 2]` Array containing the x and y components of the vector.
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

#[allow(dead_code)]
// used for DeCasteljau's algorithm
fn lerp(a: Vertex, b: Vertex, t: f32) -> Vertex {
    (1 as f32 - t) * a + t * b
}

/// A cubic bezier curve with control points of type `Vertex`.
#[derive(Debug)]
struct CubicBezier {
    /// The staring point of the curve.
    a: Vertex,
    /// The first control point of the curve.
    c_1: Vertex,
    /// The second control point of the curve.
    c_2: Vertex,
    /// The endpoint point of the curve.
    b: Vertex,
}

impl CubicBezier {
    /// Computes `samples` points along the curve an returns those as a `Vec<Vertex>`.
    /// Uses the Bernstein form of the bezier to compute the points.
    ///
    /// # Arguments
    ///
    /// * `samples` - The number of samples to take along the curve.
    fn compute(&self, samples: u32) -> Vec<Vertex> {
        let mut res: Vec<Vertex> = vec![];
        for i in 0..=samples {
            let t: f32 = (1 as f32/samples as f32) * i as f32;

            // DeCasteljau's algorithm
            /* let s_1: Vertex = lerp(self.a.clone(), self.c_1.clone(), t);
               let s_2: Vertex = lerp(self.c_1.clone(), self.c_2.clone(), t);
               let s_3: Vertex = lerp(self.c_2.clone(), self.b.clone(), t);

               let t_1: Vertex = lerp(s_1, s_2.clone(), t);
               let t_2: Vertex = lerp(s_2, s_3, t);
               let p: Vertex = lerp(t_1, t_2, t);*/

            // Bernstein
            let p: Vertex = self.a * ( -t.powi(3) + 3.0*t.powi(2) - 3.0*t + 1.0 )
                + self.c_1 * ( 3.0*t.powi(3) - 6.0*t.powi(2) + 3.0*t )
                + self.c_2 * ( -3.0*t.powi(3) + 3.0*t.powi(2) )
                + self.b * ( t.powi(3) );

            res.push(p);
        }
        res
    }
}

fn handle_window_event(window: &mut glfw::Window, event: &glfw::WindowEvent, resized: &mut bool, clicked: &mut bool) {
    match event {
        glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => window.set_should_close(true),
        glfw::WindowEvent::FramebufferSize(_, _) => {
            let (width, height): (i32, i32) = window.get_framebuffer_size();
            unsafe {
                gl::Viewport(0, 0, width, height);
            }
            *resized = true;
        },
        glfw::WindowEvent::MouseButton(MouseButton::Button1, Action::Press, _) => *clicked = true,
        _ => {}
    }
}

fn main() {
    let vertex_src: String = std::fs::read_to_string("shaders/vs.glsl").expect("Unable to read file.");
    let fragment_src: String = std::fs::read_to_string("shaders/fs.glsl").expect("Unable to read file.");
    let ctrl_vs_src: String = std::fs::read_to_string("shaders/ctrl_vs.glsl").expect("Unable to read file.");
    let ctrl_fs_src: String = std::fs::read_to_string("shaders/ctrl_fs.glsl").expect("Unable to read file.");
    let picking_vs_src: String = std::fs::read_to_string("shaders/picking_vs.glsl").expect("Unbale to read file.");
    let picking_fs_src: String = std::fs::read_to_string("shaders/picking_fs.glsl").expect("Unable to read file.");

    let mut glfw: Glfw = glfw::init(glfw::FAIL_ON_ERRORS).unwrap();

    glfw.window_hint(glfw::WindowHint::ContextVersion(3, 3));
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));

    let (mut window, events): (glfw::Window, Receiver<(f64, glfw::WindowEvent)>) = glfw.create_window(300, 300, "Bezier Curves", glfw::WindowMode::Windowed)
                               .expect("Failed to create Window.");

    window.set_key_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_mouse_button_polling(true);
    window.set_scroll_polling(true);
    window.make_current();

    gl::load(|e| glfw.get_proc_address_raw(e) as *const c_void);

    let shader: Shader = Shader::new(&vertex_src, &fragment_src);
    let ctrl_pts_shader: Shader = Shader::new(&ctrl_vs_src, &ctrl_fs_src);
    let picking_shader: Shader = Shader::new(&picking_vs_src, &picking_fs_src);

    let vao: [u32; 3] = [0, 0, 0];
    let vbo: [u32; 2] = [0, 0];

    let a: Vertex = Vertex {
        position: [0.0, 0.0],
    };
    let c_1: Vertex = Vertex {
        position: [2.0, 1.0],
    };
    let c_2: Vertex = Vertex {
        position: [-1.0, 1.0],
    };
    let b: Vertex = Vertex {
        position: [1.0, 0.0],
    };
    let bezier: CubicBezier = CubicBezier { a, c_1,  c_2, b };
    let lower: f32 = vec![a, c_1, c_2, b].iter().map(|e| e.position[0].min(e.position[1])).reduce(|acc: f32, e: f32| acc.min(e)).unwrap() - 0.1;
    let upper: f32 = vec![a, c_1, c_2, b].iter().map(|e| e.position[0].max(e.position[1])).reduce(|acc: f32, e: f32| acc.max(e)).unwrap() + 0.1;
    let curve: Vec<f32> = bezier.compute(100)
        .iter()
        .map(|e| e.position)
        .fold(vec![], |mut acc: Vec<f32>, e: [f32; 2]| {
            let v;
            acc.extend_from_slice(&e);
            v = acc;
            return v
        });

    let mut idx: u32 = 0;
    let vertices: Vec<f32> = vec![a, c_1, c_2, b].iter()
        .map(|e| {
            idx += 1;
            return [e.position[0], e.position[1],
            ((idx >>  0) & 0xFF) as f32 / 0xFF as f32,
            ((idx >>  8) & 0xFF) as f32 / 0xFF as f32,
            ((idx >> 16) & 0xFF) as f32 / 0xFF as f32,
            ((idx >> 24) & 0xFF) as f32 / 0xFF as f32];
        })
        .flatten().collect();

    println!("{:#?}", vertices);

    shader._use();
    unsafe {
        gl::GenVertexArrays(3, vao.as_ptr() as *mut u32);
        gl::GenBuffers(2, vbo.as_ptr() as *mut u32);

        gl::BindVertexArray(vao[0]);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo[0]);
        gl::BufferData(gl::ARRAY_BUFFER, (curve.len() * size_of::<f32>()) as isize, curve.as_ptr() as *const c_void, gl::STATIC_DRAW);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, (2 * size_of::<f32>()) as i32, 0 as *const c_void);
        gl::EnableVertexAttribArray(0);
        gl::LineWidth(3.0);
    }

    ctrl_pts_shader._use();
    unsafe {
        gl::BindVertexArray(vao[1]);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo[1]);
        gl::BufferData(gl::ARRAY_BUFFER, (vertices.len() * size_of::<f32>()) as isize, vertices.as_ptr() as *const c_void, gl::STATIC_DRAW);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, (6 * size_of::<f32>()) as i32, 0 as *const c_void);
        gl::EnableVertexAttribArray(0);
        gl::PointSize(25.0);
    }

    picking_shader._use();
    unsafe {
        gl::BindVertexArray(vao[2]);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, (6 * size_of::<f32>()) as i32, 0 as *const c_void);
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(1, 4, gl::FLOAT, gl::FALSE, (6 * size_of::<f32>()) as i32, (2 * size_of::<f32>()) as *const c_void);
        gl::EnableVertexAttribArray(1);
    }

    let picking_framebuffer: u32 = 0;
    let picking_texture: u32 = 0;
    let depth_buffer: u32 = 0;

    unsafe {
        gl::GenFramebuffers(1, (&picking_framebuffer as *const u32) as *mut u32);
        gl::BindFramebuffer(gl::FRAMEBUFFER, picking_framebuffer);

        gl::GenTextures(1, (&picking_texture as *const u32) as *mut u32);
        gl::BindTexture(gl::TEXTURE_2D, picking_texture);
        gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA as i32, 300, 300, 0, gl::RGBA, gl::UNSIGNED_BYTE, std::ptr::null());
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TexParameteri(gl::TEXTURE_2D, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
        gl::FramebufferTexture2D(gl::FRAMEBUFFER, gl::COLOR_ATTACHMENT0, gl::TEXTURE_2D, picking_texture, 0);
        
        gl::GenRenderbuffers(1, (&depth_buffer as *const u32) as *mut u32);
        gl::BindRenderbuffer(gl::RENDERBUFFER, depth_buffer);
        gl::RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH24_STENCIL8, 300, 300);
        gl::FramebufferRenderbuffer(gl::FRAMEBUFFER, gl::DEPTH_ATTACHMENT, gl::RENDERBUFFER, depth_buffer);

        if gl::CheckFramebufferStatus(gl::FRAMEBUFFER) != gl::FRAMEBUFFER_COMPLETE {
            println!("ERROR::FRAMEBUFFER_INCOMPLETE");
        }
        gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
    }

    let mut resized: bool = true;
    let mut clicked: bool = false;

    while !window.should_close() {
        glfw.poll_events();
        for (_, event) in glfw::flush_messages(&events) {
            handle_window_event(&mut window, &event, &mut resized, &mut clicked);
            println!("{:?}", event);
        }

        if resized {
            let (width, height): (i32, i32) = window.get_framebuffer_size();
            let window_ratio: f32 = height as f32 / width as f32;
            let projection: glm::Mat4 = glm::ortho(lower, upper, window_ratio * lower, window_ratio * upper, 0.0, 1.0);
            println!("{}", projection);
            let projection_c_str: CString = CString::new("projection").unwrap();
            shader._use();
            unsafe {
                let loc: i32 = gl::GetUniformLocation(shader.id, projection_c_str.as_ptr());
                gl::UniformMatrix4fv(loc, 1, gl::FALSE, projection.as_ptr());
            }
            ctrl_pts_shader._use();
            unsafe {
                let loc = gl::GetUniformLocation(ctrl_pts_shader.id, projection_c_str.as_ptr());
                gl::UniformMatrix4fv(loc, 1, gl::FALSE, projection.as_ptr());
            }
            picking_shader._use();
            unsafe {
                let loc = gl::GetUniformLocation(picking_shader.id, projection_c_str.as_ptr());
                gl::UniformMatrix4fv(loc, 1, gl::FALSE, projection.as_ptr());
            }
            resized = false;
        }

        if clicked {
            let (width, height): (i32, i32) = window.get_framebuffer_size();
            unsafe {
                gl::BindTexture(gl::TEXTURE_2D, picking_texture);
                gl::BindRenderbuffer(gl::RENDERBUFFER, depth_buffer);
                gl::TexImage2D(gl::TEXTURE_2D, 0, gl::RGBA as i32, width, height, 0, gl::RGBA, gl::UNSIGNED_BYTE, std::ptr::null());
                gl::RenderbufferStorage(gl::RENDERBUFFER, gl::DEPTH_COMPONENT, width, height);
                gl::BindFramebuffer(gl::FRAMEBUFFER, picking_framebuffer);
                gl::Viewport(0, 0, width, height);
                gl::ClearColor(1.0, 1.0, 1.0, 1.0);
                gl::Enable(gl::DEPTH_TEST);
                gl::DepthFunc(gl::LEQUAL);
                gl::Clear(gl::COLOR_BUFFER_BIT | gl::DEPTH_BUFFER_BIT);
            }
            let (mouse_x, mouse_y): (f64, f64) = window.get_cursor_pos();
            let data: [u8; 4] = [0, 0, 0, 0];
            picking_shader._use();
            unsafe {
                gl::BindVertexArray(vao[2]);
                gl::DrawArrays(gl::POINTS, 0, 4);

                gl::ReadPixels(mouse_x as i32, height - mouse_y as i32, 1, 1, gl::RGBA, gl::UNSIGNED_BYTE, data.as_ptr() as *mut c_void);

                gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            }
            println!("{:#?}", data);
            clicked = false;
        }

        unsafe {
            gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gl::Clear(gl::COLOR_BUFFER_BIT);
        }

        shader._use();
        unsafe {
            gl::BindVertexArray(vao[0]);
            gl::DrawArrays(gl::LINE_STRIP, 0, (curve.len() / 2) as i32);
        }

        ctrl_pts_shader._use();
        unsafe {
            gl::BindVertexArray(vao[1]);
            gl::DrawArrays(gl::POINTS, 0, 4);
        }

        window.swap_buffers();
    }
}
