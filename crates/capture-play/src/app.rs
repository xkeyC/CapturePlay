//! Main thread: winit window + input. Everything heavy lives on the render
//! thread, so window drags/resizes (modal loops on Windows) never stall video.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use cp_core::settings::Settings;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Fullscreen, Window, WindowId};

use crate::cursor::CursorIdle;
use crate::render_thread::{self, RenderHandle};

/// Requests from the render thread to the window thread.
pub enum ToMain {
    SetFullscreen(bool),
    Platform(Box<egui::PlatformOutput>),
    Exit,
}

/// Messages from the window thread to the render thread.
#[derive(Debug)]
pub enum ToRender {
    Resized(u32, u32),
    /// Window moved or display configuration changed: re-query the display.
    DisplayChanged,
    Input(egui::RawInput),
    ToggleOverlay,
    ToggleFreeze,
    FullscreenChanged(bool),
    Quit,
}

struct App {
    settings: Option<Settings>,
    proxy: EventLoopProxy<ToMain>,
    window: Option<Arc<Window>>,
    egui_state: Option<egui_winit::State>,
    render: Option<RenderHandle>,
    last_click: Option<Instant>,
    /// Shared with the render thread; used to keep the cursor while it is
    /// over the overlay.
    egui_ctx: Option<egui::Context>,
    cursor: CursorIdle,
}

impl App {
    fn send(&self, msg: ToRender) {
        if let Some(r) = &self.render {
            r.send(msg);
        }
    }

    fn set_fullscreen(&self, on: bool) {
        if let Some(w) = &self.window {
            w.set_fullscreen(on.then_some(Fullscreen::Borderless(None)));
            self.send(ToRender::FullscreenChanged(on));
        }
    }

    fn is_fullscreen(&self) -> bool {
        self.window
            .as_ref()
            .is_some_and(|w| w.fullscreen().is_some())
    }

    fn set_cursor_visible(&self, visible: bool) {
        if let Some(w) = &self.window {
            w.set_cursor_visible(visible);
        }
    }

    fn shutdown(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(r) = self.render.take() {
            r.shutdown();
        }
        event_loop.exit();
    }

    /// Global shortcuts. Returns true if handled.
    fn handle_shortcut(&self, key: &Key) -> bool {
        match key {
            Key::Named(NamedKey::F11) => {
                self.set_fullscreen(!self.is_fullscreen());
                true
            }
            Key::Named(NamedKey::Escape) if self.is_fullscreen() => {
                self.set_fullscreen(false);
                true
            }
            Key::Character(c) if c.eq_ignore_ascii_case("p") => {
                self.send(ToRender::ToggleFreeze);
                true
            }
            Key::Named(NamedKey::F1) | Key::Named(NamedKey::Tab) => {
                self.send(ToRender::ToggleOverlay);
                true
            }
            _ => false,
        }
    }
}

impl ApplicationHandler<ToMain> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let Some(settings) = self.settings.take() else {
            return;
        };
        let attrs = Window::default_attributes()
            .with_title("CapturePlay")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                tracing::error!("cannot create window: {e}");
                event_loop.exit();
                return;
            }
        };
        let hwnd = match window.window_handle().map(|h| h.as_raw()) {
            Ok(RawWindowHandle::Win32(h)) => h.hwnd.get(),
            _ => {
                tracing::error!("not a Win32 window");
                event_loop.exit();
                return;
            }
        };
        let ctx = egui::Context::default();
        let ctx_for_cursor = ctx.clone();
        let state = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(8192),
        );
        let size = window.inner_size();
        let fullscreen = settings.display.start_fullscreen;
        match render_thread::spawn(
            hwnd,
            (size.width, size.height),
            ctx,
            settings,
            self.proxy.clone(),
        ) {
            Ok(r) => self.render = Some(r),
            Err(e) => {
                tracing::error!("render thread failed to start: {e:#}");
                event_loop.exit();
                return;
            }
        }
        self.window = Some(window);
        self.egui_state = Some(state);
        self.egui_ctx = Some(ctx_for_cursor);
        if fullscreen {
            self.set_fullscreen(true);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.update_cursor(event_loop);
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: ToMain) {
        match event {
            ToMain::SetFullscreen(on) => self.set_fullscreen(on),
            ToMain::Platform(out) => {
                if let (Some(w), Some(s)) = (&self.window, &mut self.egui_state) {
                    s.handle_platform_output(w, *out);
                    // egui sets the cursor icon (and thereby shows it) every
                    // frame; keep it hidden while idle.
                    if self.cursor.hidden() {
                        w.set_cursor_visible(false);
                    }
                }
            }
            ToMain::Exit => self.shutdown(event_loop),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.clone() else {
            return;
        };
        match &event {
            WindowEvent::CloseRequested => {
                self.shutdown(event_loop);
                return;
            }
            WindowEvent::Resized(size) => self.send(ToRender::Resized(size.width, size.height)),
            WindowEvent::Moved(_) | WindowEvent::ScaleFactorChanged { .. } => {
                self.send(ToRender::DisplayChanged)
            }
            _ => {}
        }
        let show_cursor = match &event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor.moved(Instant::now(), (position.x, position.y))
            }
            WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                self.cursor.activity(Instant::now())
            }
            _ => false,
        };
        if show_cursor {
            self.set_cursor_visible(true);
        }

        let response = self
            .egui_state
            .as_mut()
            .map(|s| s.on_window_event(&window, &event));
        let consumed = response.as_ref().is_some_and(|r| r.consumed);

        if !consumed {
            match &event {
                WindowEvent::KeyboardInput { event: k, .. }
                    if k.state == ElementState::Pressed && !k.repeat =>
                {
                    self.handle_shortcut(&k.logical_key);
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    let now = Instant::now();
                    if self
                        .last_click
                        .is_some_and(|t| now - t < Duration::from_millis(350))
                    {
                        self.set_fullscreen(!self.is_fullscreen());
                        self.last_click = None;
                    } else {
                        self.last_click = Some(now);
                    }
                }
                _ => {}
            }
        }

        if response.is_some_and(|r| r.repaint)
            && let Some(s) = &mut self.egui_state
        {
            let input = s.take_egui_input(&window);
            self.send(ToRender::Input(input));
        }
    }
}

impl App {
    /// Hides the cursor after it has been idle over the video.
    fn update_cursor(&mut self, event_loop: &ActiveEventLoop) {
        let over_ui = self
            .egui_ctx
            .as_ref()
            .is_some_and(egui::Context::is_pointer_over_egui);
        let (hide, deadline) = self.cursor.poll(Instant::now(), over_ui);
        if hide {
            self.set_cursor_visible(false);
        }
        event_loop.set_control_flow(deadline.map_or(ControlFlow::Wait, ControlFlow::WaitUntil));
    }
}

impl std::fmt::Debug for ToMain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SetFullscreen(on) => write!(f, "SetFullscreen({on})"),
            Self::Platform(_) => f.write_str("Platform(..)"),
            Self::Exit => f.write_str("Exit"),
        }
    }
}

pub fn run(settings: Settings) -> Result<()> {
    let event_loop = EventLoop::<ToMain>::with_user_event().build()?;
    let mut app = App {
        settings: Some(settings),
        proxy: event_loop.create_proxy(),
        window: None,
        egui_state: None,
        render: None,
        last_click: None,
        egui_ctx: None,
        cursor: CursorIdle::new(Instant::now()),
    };
    event_loop.run_app(&mut app)?;
    Ok(())
}
