//! The eframe application: menu bar, dockable panels, status bar and keyboard shortcuts.

use std::path::PathBuf;

use std::collections::HashSet;

use egui::{RichText, Ui};
use egui_dock::{DockArea, DockState, NodeIndex};
use serde::{Deserialize, Serialize};
use subcraft_app::{ACTIONS, Action, Command, Key, NewLayer, Session, Shortcut, action_for};
use subcraft_core::{ChannelKind, ChannelValue};
use subcraft_doc::{
    FillChannel, FillSource, Generator, GeneratorKind, LayerContent, MaskEffectKind,
    ProceduralKind, ProceduralParams,
};

use crate::dialogs::{self, Pending};
use crate::orbit::OrbitCamera;
use crate::state::{Tool, UiState};
use crate::widgets::{icon_button, primary_button, secondary_button, tool_button};
use crate::{icons, panels, theme};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Tab {
    Viewport,
    UvView,
    Layers,
    Properties,
    Brush,
    TextureSets,
    History,
    Shelf,
}

impl Tab {
    fn title(self) -> &'static str {
        match self {
            Tab::Viewport => "3D View",
            Tab::UvView => "2D View",
            Tab::Layers => "Layers",
            Tab::Properties => "Properties",
            Tab::Brush => "Brush",
            Tab::TextureSets => "Texture Sets",
            Tab::History => "History",
            Tab::Shelf => "Shelf",
        }
    }

    fn icon(self) -> &'static str {
        match self {
            Tab::Viewport => icons::CUBE,
            Tab::UvView => icons::GRID,
            Tab::Layers => icons::STACK,
            Tab::Properties => icons::SLIDERS,
            Tab::Brush => icons::PAINT_BRUSH,
            Tab::TextureSets => icons::CUBE_WIRE,
            Tab::History => icons::HISTORY,
            Tab::Shelf => icons::SQUARES,
        }
    }
}

/// Command-line options.
#[derive(Clone, Debug, Default)]
pub struct LaunchOptions {
    /// Project (`.subcraft`) or mesh to open instead of the sample crate.
    pub mesh: Option<PathBuf>,
    /// Build a small demo layer stack (fills, masks and painted strokes) on start.
    pub demo: bool,
    /// Save a screenshot of the window to this path after a few frames, then quit.
    pub screenshot: Option<PathBuf>,
    /// Start with the 2D (UV) view in front.
    pub view_2d: bool,
}

pub fn default_layout() -> DockState<Tab> {
    let mut dock = DockState::new(vec![Tab::Viewport, Tab::UvView]);
    let surface = dock.main_surface_mut();
    let [center, right] = surface.split_right(NodeIndex::root(), 0.74, vec![Tab::Layers]);
    surface.split_below(right, 0.4, vec![Tab::Properties, Tab::Brush]);
    let [center, left] = surface.split_left(center, 0.2, vec![Tab::TextureSets]);
    surface.split_below(left, 0.42, vec![Tab::History]);
    surface.split_below(center, 0.74, vec![Tab::Shelf]);
    dock
}

const DOCK_KEY: &str = "subcraft_studio_dock_v4";

pub fn split_layout() -> DockState<Tab> {
    let mut dock = DockState::new(vec![Tab::Viewport]);
    let tree = dock.main_surface_mut();
    let [center, right] = tree.split_right(
        NodeIndex::root(),
        0.77,
        vec![Tab::Layers, Tab::TextureSets, Tab::History],
    );
    tree.split_below(right, 0.43, vec![Tab::Properties, Tab::Brush]);
    let [center, _] = tree.split_right(center, 0.5, vec![Tab::UvView]);
    tree.split_below(center, 0.68, vec![Tab::Shelf]);
    dock
}

pub fn paint_layout() -> DockState<Tab> {
    let mut dock = DockState::new(vec![Tab::Viewport, Tab::UvView]);
    let tree = dock.main_surface_mut();
    let [center, right] = tree.split_right(
        NodeIndex::root(),
        0.76,
        vec![Tab::Layers, Tab::TextureSets, Tab::History],
    );
    tree.split_below(right, 0.43, vec![Tab::Properties, Tab::Brush]);
    tree.split_below(center, 0.68, vec![Tab::Shelf]);
    dock
}

pub struct SubcraftApp {
    state: UiState,
    dock: DockState<Tab>,
    frames: u64,
    screenshot: Option<PathBuf>,
    screenshot_requested: bool,
    /// Action approved this frame, run after the UI pass.
    pending_now: Option<Pending>,
    /// Tabs whose body was shown last frame (drawn with the accent marker).
    active_tabs: HashSet<Tab>,
    preferences: crate::workspace::Preferences,
    recovery: crate::workspace::Recovery,
    last_project: Option<PathBuf>,
}

impl SubcraftApp {
    pub fn new(cc: &eframe::CreationContext<'_>, options: LaunchOptions) -> Self {
        theme::apply(&cc.egui_ctx);
        let mut status = None;
        let is_project = |p: &std::path::Path| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case(subcraft_app::PROJECT_EXTENSION))
        };
        let session = match &options.mesh {
            Some(path) if is_project(path) => match Session::open_project(path) {
                Ok(s) => s,
                Err(e) => {
                    status = Some(format!(
                        "Could not open {}: {e}. Showing the sample.",
                        path.display()
                    ));
                    Session::sample()
                }
            },
            Some(path) => match Session::open_mesh(path, subcraft_app::DEFAULT_RESOLUTION) {
                Ok((s, warnings)) => {
                    if !warnings.is_empty() {
                        status = Some(format!(
                            "Imported with {} warning(s): {}",
                            warnings.len(),
                            warnings[0]
                        ));
                    }
                    s
                }
                Err(e) => {
                    status = Some(format!(
                        "Could not open {}: {e}. Showing the sample.",
                        path.display()
                    ));
                    Session::sample()
                }
            },
            None => Session::sample(),
        };
        let mut state = UiState::new(session, cc.wgpu_render_state.clone()).with_user_library(
            subcraft_app::assets::UserLibrary::default_root()
                .map(subcraft_app::assets::UserLibrary::new),
        );
        if let Some(s) = status {
            state.set_status(s, 0.0);
        }
        if options.demo {
            build_demo(&mut state);
        }
        let mut preferences: crate::workspace::Preferences = cc
            .storage
            .and_then(|s| s.get_string(crate::workspace::SETTINGS_KEY))
            .and_then(|j| serde_json::from_str(&j).ok())
            .unwrap_or_default();
        if !matches!(preferences.viewport_scale, 0.5 | 0.75 | 1.0) {
            preferences.viewport_scale = 1.0;
        }
        state.viewport_scale = preferences.viewport_scale;
        let dock = cc
            .storage
            .and_then(|s| s.get_string(DOCK_KEY))
            .and_then(|j| serde_json::from_str(&j).ok())
            .filter(|_| options.screenshot.is_none())
            .unwrap_or_else(paint_layout);
        let mut dock = dock;
        if options.view_2d
            && let Some(path) = dock.find_tab(&Tab::UvView)
        {
            let _ = dock.set_active_tab(path);
        }
        Self {
            state,
            dock,
            frames: 0,
            screenshot: options.screenshot,
            screenshot_requested: false,
            pending_now: None,
            active_tabs: HashSet::new(),
            preferences,
            recovery: Default::default(),
            last_project: None,
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::I) && !i.modifiers.any()) {
            self.state.tool = Tool::Eyedropper;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Num4) && !i.modifiers.any()) {
            self.state.tool = Tool::PolygonFill;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Num5) && !i.modifiers.any()) {
            self.state.view_mode = subcraft_render::ViewMode::Material;
        }
        if ctx.input(|i| i.key_pressed(egui::Key::Num6) && !i.modifiers.any()) {
            let modes = subcraft_render::ViewMode::ALL;
            let next = (modes
                .iter()
                .position(|m| *m == self.state.view_mode)
                .unwrap_or(0)
                + 1)
                % modes.len();
            self.state.view_mode = modes[next];
        }
        let pressed: Vec<Shortcut> = ctx.input(|i| {
            i.events
                .iter()
                .filter_map(|e| match e {
                    egui::Event::Key {
                        key,
                        pressed: true,
                        repeat: false,
                        modifiers,
                        ..
                    } => map_key(*key).map(|k| Shortcut {
                        ctrl: modifiers.command,
                        shift: modifiers.shift,
                        alt: modifiers.alt,
                        key: k,
                    }),
                    _ => None,
                })
                .collect()
        });
        for s in pressed {
            if let Some(a) = action_for(s) {
                self.perform(a, ctx.input(|i| i.time));
            }
        }
    }

    fn perform(&mut self, action: Action, now: f64) {
        match action {
            Action::ResetCamera => {
                self.state.camera = OrbitCamera::framing(&self.state.session.scene().mesh.bounds)
            }
            Action::BrushTool => self.state.tool = Tool::Brush,
            Action::EraserTool => self.state.tool = Tool::Eraser,
            Action::Save => {
                dialogs::save(&mut self.state, false, now);
            }
            Action::SaveAs => {
                dialogs::save(&mut self.state, true, now);
            }
            Action::OpenProject => self.request(Pending::OpenProject, now),
            Action::Bake => self.state.bake.open = true,
            Action::Export => {
                self.state.export.open = true;
                self.state.export.settings.height_to_normal = self.state.height_scale;
            }
            _ => {
                if let Err(e) = self.state.session.perform(action) {
                    self.state.set_status(e.to_string(), now);
                }
            }
        }
    }

    /// Top bar: brand, menus, project name, then undo/redo and the main actions on the right.
    fn header_bar(&mut self, ui: &mut Ui) {
        let now = ui.input(|i| i.time);
        let bar = ui.max_rect();
        // Right side first so the left side can't push it out of the bar.
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(bar)
                .layout(egui::Layout::right_to_left(egui::Align::Center)),
            |ui| {
                if primary_button(ui, format!("{}  Export", icons::EXPORT))
                    .on_hover_text("Export textures (Ctrl+Shift+E)")
                    .clicked()
                {
                    self.perform(Action::Export, now);
                }
                if secondary_button(ui, format!("{}  Bake", icons::MAGIC_WAND))
                    .on_hover_text("Bake mesh maps (Ctrl+B)")
                    .clicked()
                {
                    self.perform(Action::Bake, now);
                }
                if icon_button(ui, icons::SAVE, "Save (Ctrl+S)").clicked() {
                    self.perform(Action::Save, now);
                }
                ui.add_space(6.0);
                let h = self.state.session.history();
                let (can_undo, can_redo) = (h.can_undo(), h.can_redo());
                if ui
                    .add_enabled_ui(can_redo, |ui| icon_button(ui, icons::REDO, "Redo (Ctrl+Y)"))
                    .inner
                    .clicked()
                {
                    self.perform(Action::Redo, now);
                }
                if ui
                    .add_enabled_ui(can_undo, |ui| icon_button(ui, icons::UNDO, "Undo (Ctrl+Z)"))
                    .inner
                    .clicked()
                {
                    self.perform(Action::Undo, now);
                }
            },
        );
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(bar)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
            |ui| {
                // Brand mark.
                let (r, _) = ui.allocate_exact_size(egui::vec2(18.0, 18.0), egui::Sense::hover());
                ui.painter().rect_filled(r, 0.0, theme::ACCENT);
                ui.painter().rect_filled(
                    egui::Rect::from_min_size(r.min + egui::vec2(5.0, 5.0), egui::vec2(8.0, 8.0)),
                    0.0,
                    theme::BG_2,
                );
                ui.label(
                    RichText::new("SubCraft Studio")
                        .font(theme::font_strong(14.0))
                        .color(theme::TEXT),
                );
                ui.add_space(10.0);
                self.menu(ui);
            },
        );
        // Project name centered in the bar.
        let title = dialogs::title(&self.state);
        let name = title.trim_start_matches("SubCraft — ");
        ui.painter().text(
            bar.center(),
            egui::Align2::CENTER_CENTER,
            name,
            theme::font(12.5),
            theme::TEXT_FAINT,
        );
    }

    fn context_bar(&mut self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            let settings = &mut self.state.session.brush.settings;
            ui.label(RichText::new("Size").color(theme::TEXT_DIM));
            ui.add(
                egui::DragValue::new(&mut settings.size)
                    .speed(0.1)
                    .range(0.1..=50.0)
                    .suffix("%"),
            );
            ui.label(RichText::new("Flow").color(theme::TEXT_DIM));
            ui.add(
                egui::DragValue::new(&mut settings.flow)
                    .speed(0.01)
                    .range(0.0..=1.0)
                    .max_decimals(2),
            );
            ui.label(RichText::new("Opacity").color(theme::TEXT_DIM));
            ui.add(
                egui::DragValue::new(&mut settings.stroke_opacity)
                    .speed(0.01)
                    .range(0.0..=1.0)
                    .max_decimals(2),
            );
            ui.separator();
            egui::ComboBox::from_id_salt("active-brush-channel")
                .width(112.0)
                .selected_text(self.state.active_channel.display_name())
                .show_ui(ui, |ui| {
                    for channel in self.state.session.channels() {
                        ui.selectable_value(
                            &mut self.state.active_channel,
                            channel,
                            channel.display_name(),
                        );
                    }
                });
            let channel = self.state.active_channel;
            if let Some(entry) = self.state.session.brush.material.channels.get_mut(&channel) {
                crate::widgets::channel_value(ui, channel, &mut entry.value);
            }
            ui.separator();
            ui.menu_button(format!("{}  Workspace", icons::SQUARES), |ui| {
                if ui.button("Painting").clicked() {
                    self.dock = paint_layout();
                    ui.close();
                }
                if ui.button("3D + 2D").clicked() {
                    self.dock = split_layout();
                    ui.close();
                }
                if ui.button("Classic panels").clicked() {
                    self.dock = default_layout();
                    ui.close();
                }
            });
        });
    }

    /// Vertical tool strip left of the dock area.
    fn tool_rail(&mut self, ui: &mut Ui) {
        ui.spacing_mut().item_spacing.y = 2.0;
        ui.add_space(6.0);
        ui.vertical_centered(|ui| {
            let tool = &mut self.state.tool;
            for (t, icon, tip) in [
                (Tool::Brush, icons::PAINT_BRUSH, "Brush (B)"),
                (Tool::Eraser, icons::ERASER, "Eraser (E)"),
                (Tool::PolygonFill, icons::PAINT_BUCKET, "Polygon fill"),
                (
                    Tool::Eyedropper,
                    icons::DROP,
                    "Eyedropper (I) · sample the active channel",
                ),
            ] {
                if tool_button(ui, icon, *tool == t, tip).clicked() {
                    *tool = t;
                }
            }
            ui.add_space(6.0);
            rail_divider(ui);
            ui.add_space(6.0);
            let sym = &mut self.state.session.brush.options.symmetry[0];
            if tool_button(ui, icons::SYMMETRY, *sym, "Mirror strokes across X (X)").clicked() {
                *sym = !*sym;
            }
            let target = &mut self.state.session.brush.target;
            let on_mask = *target == subcraft_paint::PaintTarget::MaskBase;
            if tool_button(
                ui,
                icons::MASK,
                on_mask,
                "Paint the layer's mask instead of its content",
            )
            .clicked()
            {
                *target = if on_mask {
                    subcraft_paint::PaintTarget::Content
                } else {
                    subcraft_paint::PaintTarget::MaskBase
                };
            }
            ui.add_space(6.0);
            rail_divider(ui);
            ui.add_space(6.0);
            if tool_button(ui, icons::CAMERA, false, "Reset camera (Home)").clicked() {
                self.perform(Action::ResetCamera, ui.input(|i| i.time));
            }
            ui.with_layout(egui::Layout::bottom_up(egui::Align::Center), |ui| {
                ui.add_space(6.0);
                if tool_button(
                    ui,
                    icons::KEYBOARD,
                    self.state.show_shortcuts,
                    "Keyboard shortcuts",
                )
                .clicked()
                {
                    self.state.show_shortcuts = !self.state.show_shortcuts;
                }
            });
        });
    }

    fn menu(&mut self, ui: &mut Ui) {
        let now = ui.input(|i| i.time);
        egui::MenuBar::new().ui(ui, |ui| {
            let item = |ui: &mut Ui, icon: &str, label: &str, sc: &str| ui.add(egui::Button::new(format!("{icon}   {label}")).shortcut_text(sc)).clicked();
            ui.menu_button("File", |ui| {
                ui.set_min_width(240.0);
                if item(ui, icons::FILE_PLUS, "New (sample crate)", "") {
                    self.request(Pending::NewSample, now);
                    ui.close();
                }
                if item(ui, icons::CUBE, "New from mesh…", "") {
                    ui.close();
                    self.request(Pending::OpenMesh, now);
                }
                if item(ui, icons::FOLDER_OPEN, "Open project…", "Ctrl+O") {
                    ui.close();
                    self.request(Pending::OpenProject, now);
                }
                ui.menu_button("Recent projects", |ui| {
                    let recent = self.preferences.recent.clone();
                    if recent.is_empty() { ui.label("No recent projects"); }
                    for path in recent {
                        let label = path.file_name().and_then(|s| s.to_str()).unwrap_or("Project");
                        if ui.add_enabled(path.exists(), egui::Button::new(label)).on_hover_text(path.display().to_string()).clicked() {
                            self.state.pending_path = Some(path);
                            self.request(Pending::OpenPath, now);
                            ui.close();
                        }
                    }
                });
                #[cfg(not(target_arch = "wasm32"))]
                ui.menu_button("Recovery projects", |ui| {
                    let paths = crate::workspace::Recovery::files();
                    if paths.is_empty() { ui.label("No recovery copies yet"); }
                    for path in paths {
                        if ui.button(path.file_name().and_then(|s| s.to_str()).unwrap_or("Recovery")).on_hover_text(path.display().to_string()).clicked() {
                            self.state.pending_path = Some(path);
                            self.request(Pending::OpenPath, now);
                            ui.close();
                        }
                    }
                    ui.separator();
                    ui.checkbox(&mut self.preferences.autosave, "Automatic recovery saves");
                    ui.add(egui::Slider::new(&mut self.preferences.autosave_seconds, 30..=600).suffix(" s"));
                    crate::widgets::hint(ui, "Background copies preserve unsaved work. Save As to keep a recovered project under a new name.");
                });
                ui.separator();
                if item(ui, icons::SAVE, "Save", "Ctrl+S") {
                    ui.close();
                    self.perform(Action::Save, now);
                }
                if item(ui, icons::SAVE, "Save as…", "Ctrl+Shift+S") {
                    ui.close();
                    self.perform(Action::SaveAs, now);
                }
                ui.separator();
                if item(ui, icons::EXPORT, "Export textures…", "Ctrl+Shift+E") {
                    ui.close();
                    self.perform(Action::Export, now);
                }
                if item(ui, icons::MAGIC_WAND, "Bake mesh maps…", "Ctrl+B") {
                    ui.close();
                    self.perform(Action::Bake, now);
                }
                ui.separator();
                if item(ui, icons::X, "Quit", "") {
                    ui.send_viewport_cmd(egui::ViewportCommand::Close);
                }
            });
            ui.menu_button("Edit", |ui| {
                ui.set_min_width(240.0);
                let h = self.state.session.history();
                let undo = h.undo_label().map(|l| format!("Undo {l}")).unwrap_or("Undo".into());
                let redo = h.redo_label().map(|l| format!("Redo {l}")).unwrap_or("Redo".into());
                let (can_undo, can_redo) = (h.can_undo(), h.can_redo());
                if ui.add_enabled(can_undo, egui::Button::new(format!("{}   {undo}", icons::UNDO)).shortcut_text("Ctrl+Z")).clicked() {
                    self.perform(Action::Undo, now);
                }
                if ui.add_enabled(can_redo, egui::Button::new(format!("{}   {redo}", icons::REDO)).shortcut_text("Ctrl+Y")).clicked() {
                    self.perform(Action::Redo, now);
                }
            });
            ui.menu_button("Layer", |ui| {
                ui.set_min_width(240.0);
                for (a, icon) in [
                    (Action::NewPaintLayer, icons::PAINT_BRUSH),
                    (Action::NewFillLayer, icons::PAINT_BUCKET),
                    (Action::NewFolder, icons::FOLDER),
                    (Action::DuplicateLayer, icons::COPY),
                    (Action::DeleteLayer, icons::TRASH),
                ] {
                    let info = subcraft_app::actions::info(a);
                    let sc = info.shortcuts.first().map(|s| s.label()).unwrap_or_default();
                    if item(ui, icon, info.label, &sc) {
                        self.perform(a, now);
                        ui.close();
                    }
                }
                ui.separator();
                if item(ui, icons::IMAGE, "Import texture set…", "") {
                    ui.close();
                    crate::panels::layers::import_texture_set_dialog(ui.ctx());
                }
            });
            ui.menu_button("View", |ui| {
                ui.set_min_width(240.0);
                if item(ui, icons::CAMERA, "Reset camera", "Home") {
                    self.perform(Action::ResetCamera, now);
                    ui.close();
                }
                if item(ui, icons::SQUARES, "Reset layout", "") {
                    self.dock = default_layout();
                    ui.close();
                }
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label("Height bump");
                    ui.add(egui::Slider::new(&mut self.state.height_scale, 0.0..=0.1));
                });
            });
            ui.menu_button("Help", |ui| {
                ui.set_min_width(240.0);
                if item(ui, icons::KEYBOARD, "Keyboard shortcuts", "") {
                    self.state.show_shortcuts = true;
                    ui.close();
                }
            });
        });
    }

    /// Run `action` now, or after the unsaved-changes prompt.
    fn request(&mut self, action: Pending, now: f64) {
        if self.state.session.is_modified() && self.screenshot.is_none() {
            self.state.confirm = Some(action);
        } else {
            self.pending_now = Some(action);
        }
        let _ = now;
    }

    fn status_bar(&mut self, ui: &mut Ui) {
        let now = ui.input(|i| i.time);
        ui.horizontal_centered(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            let (icon, tool) = match self.state.tool {
                Tool::Brush => (icons::PAINT_BRUSH, "Brush"),
                Tool::Eraser => (icons::ERASER, "Eraser"),
                Tool::PolygonFill => (icons::PAINT_BUCKET, "Polygon fill"),
                Tool::Eyedropper => (icons::DROP, "Eyedropper"),
            };
            let small = |t: String, c| RichText::new(t).font(theme::font(11.5)).color(c);
            ui.label(small(format!("{icon}  {tool}"), theme::ACCENT));
            ui.label(small(
                format!("Size {:.1}%", self.state.session.brush.settings.size),
                theme::TEXT_DIM,
            ));
            if let Some(l) = self.state.session.selected_layer() {
                ui.label(small(
                    format!(
                        "{}  {} / {}",
                        icons::STACK,
                        self.state.session.selected_texture_set().name,
                        l.name
                    ),
                    theme::TEXT_DIM,
                ));
            }
            if let Some((msg, t)) = &self.state.status
                && now - t < 6.0
            {
                ui.label(small(format!("{}  {msg}", icons::INFO), theme::WARNING));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(small(
                    format!("{:.1} ms", self.state.frame_time_ms),
                    theme::TEXT_FAINT,
                ));
                let mesh = &self.state.session.scene().mesh;
                ui.label(small(
                    format!("{} tris", mesh.triangle_count()),
                    theme::TEXT_FAINT,
                ));
            });
        });
    }

    fn shortcuts_window(&mut self, ctx: &egui::Context) {
        let mut open = self.state.show_shortcuts;
        egui::Window::new("Keyboard shortcuts")
            .open(&mut open)
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                egui::Grid::new("shortcuts")
                    .num_columns(2)
                    .spacing([32.0, 8.0])
                    .show(ui, |ui| {
                        let key = |t: String| RichText::new(t).monospace().color(theme::ACCENT);
                        for a in ACTIONS {
                            ui.label(RichText::new(a.label).color(theme::TEXT_DIM));
                            ui.label(key(a
                                .shortcuts
                                .iter()
                                .map(|s| s.label())
                                .collect::<Vec<_>>()
                                .join(" / ")));
                            ui.end_row();
                        }
                        for (what, how) in [
                            ("Orbit", "Alt+LMB or RMB drag"),
                            ("Pan", "MMB drag"),
                            ("Zoom", "Wheel or Alt+RMB drag"),
                            ("Straight line", "Shift+click"),
                        ] {
                            ui.label(RichText::new(what).color(theme::TEXT_DIM));
                            ui.label(key(how.to_string()));
                            ui.end_row();
                        }
                        for (what, how) in [
                            ("Eyedropper (active channel)", "I"),
                            ("Polygon fill", "4"),
                            ("Material view", "5"),
                            ("Next solo channel", "6"),
                            ("Rename selected layer", "F2"),
                            ("Reorder / group layers", "Drag layer row"),
                        ] {
                            ui.label(RichText::new(what).color(theme::TEXT_DIM));
                            ui.label(key(how.to_string()));
                            ui.end_row();
                        }
                    });
            });
        self.state.show_shortcuts = open;
    }

    fn handle_screenshot(&mut self, ctx: &egui::Context) {
        let Some(path) = self.screenshot.clone() else {
            return;
        };
        ctx.request_repaint();
        if !self.screenshot_requested && self.frames >= 20 {
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
            self.screenshot_requested = true;
        }
        let image = ctx.input(|i| {
            i.raw.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(img) = image {
            let [w, h] = img.size;
            let data: Vec<f32> = img
                .pixels
                .iter()
                .flat_map(|c| c.to_array())
                .map(|b| b as f32 / 255.0)
                .collect();
            match subcraft_image::save_image(
                &path,
                w as u32,
                h as u32,
                4,
                &data,
                subcraft_image::FileFormat::Png8,
            ) {
                Ok(()) => {
                    log::info!("screenshot saved to {}", path.display());
                    log::info!(
                        "render diagnostics: UI frames={}, 3D renders={}",
                        self.frames,
                        self.state.viewport_render_count
                    );
                }
                Err(e) => log::error!("screenshot failed: {e}"),
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

fn rail_divider(ui: &mut Ui) {
    let (r, _) = ui.allocate_exact_size(egui::vec2(24.0, 1.0), egui::Sense::hover());
    ui.painter().hline(
        r.x_range(),
        r.center().y,
        egui::Stroke::new(1.0, theme::BORDER_STRONG),
    );
}

fn map_key(k: egui::Key) -> Option<Key> {
    Some(match k {
        egui::Key::B => Key::B,
        egui::Key::D => Key::D,
        egui::Key::E => Key::E,
        egui::Key::F => Key::F,
        egui::Key::G => Key::G,
        egui::Key::N => Key::N,
        egui::Key::O => Key::O,
        egui::Key::S => Key::S,
        egui::Key::X => Key::X,
        egui::Key::Y => Key::Y,
        egui::Key::Z => Key::Z,
        egui::Key::Delete => Key::Delete,
        egui::Key::OpenBracket => Key::OpenBracket,
        egui::Key::CloseBracket => Key::CloseBracket,
        egui::Key::Home => Key::Home,
        _ => return None,
    })
}

struct Viewer<'a> {
    state: &'a mut UiState,
    /// Tabs shown last frame (marked) and the set being collected this frame.
    was_active: &'a HashSet<Tab>,
    active: HashSet<Tab>,
}

impl egui_dock::TabViewer for Viewer<'_> {
    type Tab = Tab;

    fn id(&mut self, tab: &mut Tab) -> egui::Id {
        egui::Id::new(*tab)
    }

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        RichText::new(format!("{}   {}", tab.icon(), tab.title()))
            .font(theme::font_strong(12.0))
            .into()
    }

    fn on_tab_button(&mut self, tab: &mut Tab, response: &egui::Response) {
        if self.was_active.contains(tab) {
            let r = response.rect;
            let painter = response.ctx.layer_painter(response.layer_id);
            painter.rect_filled(
                egui::Rect::from_min_size(r.min, egui::vec2(r.width(), 2.0)),
                0.0,
                theme::ACCENT,
            );
        }
    }

    fn ui(&mut self, ui: &mut Ui, tab: &mut Tab) {
        self.active.insert(*tab);
        match tab {
            Tab::Viewport => crate::viewport::show(self.state, ui),
            Tab::UvView => panels::uv_view::show(self.state, ui),
            Tab::Layers => panels::layers::show(self.state, ui),
            Tab::Properties => panels::properties::show(self.state, ui),
            Tab::Brush => panels::brush::show(self.state, ui),
            Tab::TextureSets => panels::texture_sets::show(self.state, ui),
            Tab::History => panels::history::show(self.state, ui),
            Tab::Shelf => panels::shelf::show(self.state, ui),
        }
    }

    fn is_closeable(&self, _tab: &Tab) -> bool {
        false
    }

    fn clear_background(&self, tab: &Tab) -> bool {
        !matches!(tab, Tab::Viewport | Tab::UvView)
    }
}

impl eframe::App for SubcraftApp {
    fn ui(&mut self, ui: &mut Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.frames += 1;
        self.shortcuts(&ctx);
        if ctx.input(|i| i.pointer.any_released()) {
            let _ = self.state.session.execute(Command::Seal);
        }
        self.state.sync_gpu();
        self.state.poll_library();
        self.state.textures.poll(&ctx);
        if self.state.env_loader.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(200));
        }

        let bar = |fill| {
            egui::Frame::new()
                .fill(fill)
                .inner_margin(egui::Margin::symmetric(12, 0))
        };
        egui::Panel::top("header")
            .exact_size(44.0)
            .show_separator_line(false)
            .frame(bar(theme::BG_2))
            .show(ui, |ui| self.header_bar(ui));
        egui::Panel::top("context-tools")
            .exact_size(36.0)
            .show_separator_line(false)
            .frame(bar(theme::BG_1))
            .show(ui, |ui| self.context_bar(ui));
        egui::Panel::bottom("status")
            .exact_size(26.0)
            .show_separator_line(false)
            .frame(bar(theme::BG_0))
            .show(ui, |ui| self.status_bar(ui));
        egui::Panel::left("tools")
            .exact_size(48.0)
            .resizable(false)
            .show_separator_line(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_2)
                    .inner_margin(egui::Margin::symmetric(4, 0)),
            )
            .show(ui, |ui| self.tool_rail(ui));
        let style = theme::dock_style(ui.style());
        let mut viewer = Viewer {
            state: &mut self.state,
            was_active: &self.active_tabs,
            active: HashSet::new(),
        };
        egui::Frame::new()
            .fill(theme::BG_0)
            .inner_margin(egui::Margin {
                left: 3,
                right: 0,
                top: 3,
                bottom: 0,
            })
            .show(ui, |ui| {
                DockArea::new(&mut self.dock)
                    .style(style)
                    .show_close_buttons(false)
                    .show_leaf_close_all_buttons(false)
                    .show_leaf_collapse_buttons(false)
                    .show_inside(ui, &mut viewer);
            });
        let active = std::mem::take(&mut viewer.active);
        if active != self.active_tabs {
            self.active_tabs = active;
            ctx.request_repaint();
        }
        self.shortcuts_window(&ctx);
        let now = ctx.input(|i| i.time);
        if self.screenshot.is_none() {
            if let Some(path) = self.state.session.project_path().map(PathBuf::from) {
                if self.last_project.as_ref() != Some(&path) {
                    self.preferences.remember(path.clone());
                    self.last_project = Some(path);
                }
            } else {
                self.last_project = None;
            }
            self.recovery
                .poll(&mut self.state, &self.preferences, &ctx, now);
        }
        dialogs::export_window(&mut self.state, &ctx, now);
        dialogs::bake_window(&mut self.state, &ctx, now);
        dialogs::poll_bake(&mut self.state, &ctx, now);
        crate::files::poll(&mut self.state, now);
        if let Some(action) = dialogs::unsaved_prompt(&mut self.state, &ctx, now) {
            self.pending_now = Some(action);
        }
        // Closing with unsaved work asks first.
        if ctx.input(|i| i.viewport().close_requested())
            && !self.state.allow_close
            && self.state.session.is_modified()
            && self.screenshot.is_none()
        {
            ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
            self.state.confirm = Some(Pending::Close);
        }
        if let Some(action) = self.pending_now.take() {
            if action == Pending::Close {
                self.state.allow_close = true;
            }
            dialogs::run_pending(&mut self.state, action, &ctx, now);
        }
        let title = dialogs::title(&self.state);
        if title != self.state.last_title {
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title.clone()));
            self.state.last_title = title;
        }
        self.handle_screenshot(&ctx);
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        if self.screenshot.is_some() {
            return; // scripted runs never overwrite the user's layout
        }
        if let Ok(j) = serde_json::to_string(&self.dock) {
            storage.set_string(DOCK_KEY, j);
        }
        self.preferences.viewport_scale = self.state.viewport_scale;
        if let Ok(j) = serde_json::to_string(&self.preferences) {
            storage.set_string(crate::workspace::SETTINGS_KEY, j);
        }
    }
}

/// Demo stack on the sample crate: procedural base, an upward-facing "moss" fill masked by a
/// directional generator, and a few painted strokes.
fn build_demo(state: &mut UiState) {
    let s = &mut state.session;
    let run = |s: &mut Session, c: Command| {
        if let Err(e) = s.execute(c) {
            log::warn!("demo: {e}");
        }
    };
    let paint = s
        .selection()
        .layer
        .expect("new sessions select a paint layer");
    // Base material (below the paint layer).
    run(s, Command::AddLayer(NewLayer::Fill));
    let base = s.selection().layer.unwrap();
    run(s, Command::MoveDown(base));
    let wood = FillSource::Procedural {
        params: ProceduralParams {
            scale: 3.0,
            ..ProceduralParams::new(ProceduralKind::WoodRings)
        },
        low: ChannelValue::rgb(0.18, 0.09, 0.04),
        high: ChannelValue::rgb(0.45, 0.25, 0.11),
    };
    run(
        s,
        Command::SetFillChannel {
            layer: base,
            channel: ChannelKind::BaseColor,
            fill: FillChannel::new(wood),
        },
    );
    let rough = FillSource::Procedural {
        params: ProceduralParams {
            scale: 8.0,
            ..ProceduralParams::new(ProceduralKind::Grunge)
        },
        low: ChannelValue::scalar(0.45),
        high: ChannelValue::scalar(0.8),
    };
    run(
        s,
        Command::SetFillChannel {
            layer: base,
            channel: ChannelKind::Roughness,
            fill: FillChannel::new(rough),
        },
    );
    run(
        s,
        Command::Rename {
            layer: base,
            name: "Wood".into(),
        },
    );
    // Moss on top-facing surfaces.
    run(s, Command::SelectLayer(Some(paint)));
    run(s, Command::AddLayer(NewLayer::Fill));
    let moss = s.selection().layer.unwrap();
    run(
        s,
        Command::Rename {
            layer: moss,
            name: "Moss".into(),
        },
    );
    let moss_color = FillSource::Procedural {
        params: ProceduralParams {
            scale: 6.0,
            ..ProceduralParams::new(ProceduralKind::Clouds)
        },
        low: ChannelValue::rgb(0.06, 0.12, 0.03),
        high: ChannelValue::rgb(0.2, 0.35, 0.08),
    };
    run(
        s,
        Command::SetFillChannel {
            layer: moss,
            channel: ChannelKind::BaseColor,
            fill: FillChannel::new(moss_color),
        },
    );
    run(
        s,
        Command::SetFillChannel {
            layer: moss,
            channel: ChannelKind::Roughness,
            fill: FillChannel::new(FillSource::uniform(ChannelValue::scalar(0.9))),
        },
    );
    if let Some(LayerContent::Fill(f)) = s.selected_layer().map(|l| l.content.clone()) {
        for (k, ch) in f.channels {
            if !matches!(k, ChannelKind::BaseColor | ChannelKind::Roughness) && ch.enabled {
                run(
                    s,
                    Command::SetFillChannel {
                        layer: moss,
                        channel: k,
                        fill: FillChannel {
                            enabled: false,
                            ..ch
                        },
                    },
                );
            }
        }
    }
    run(
        s,
        Command::AddMask {
            layer: moss,
            white: false,
        },
    );
    let mut g = Generator::new(GeneratorKind::Directional);
    g.level = 0.85;
    g.softness = 0.2;
    g.noise_amount = 0.35;
    run(
        s,
        Command::AddMaskEffect {
            layer: moss,
            kind: MaskEffectKind::Generator(g),
        },
    );
    // Painted strokes on the paint layer (now topmost).
    run(s, Command::SelectLayer(Some(paint)));
    run(s, Command::MoveUp(paint));
    s.brush.settings.size = 4.0;
    s.brush.material = subcraft_paint::BrushMaterial::with(&[
        (ChannelKind::BaseColor, ChannelValue::rgb(0.75, 0.12, 0.03)),
        (ChannelKind::Roughness, ChannelValue::scalar(0.35)),
        (ChannelKind::Metallic, ChannelValue::scalar(0.0)),
    ]);
    let cam = state.camera.paint_camera([1280, 800]);
    let center = state.session.scene().mesh.bounds.center();
    let anchor = cam
        .project(center)
        .map(|p| glam::Vec2::new(p.x, p.y))
        .unwrap_or(glam::Vec2::new(640.0, 400.0));
    let s = &mut state.session;
    for row in 0..2 {
        if s.begin_stroke().is_err() {
            break;
        }
        for k in 0..60 {
            let t = k as f32 / 59.0;
            let p = anchor
                + glam::Vec2::new(
                    -260.0 + 420.0 * t,
                    40.0 + row as f32 * 60.0 + (t * 12.0).sin() * 18.0,
                );
            let _ = s.stroke_to(&cam, p, false);
        }
        s.end_stroke();
    }
}
