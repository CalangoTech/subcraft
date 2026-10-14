//! File dialogs and windows: save / open project, export textures, unsaved-changes prompt.

use std::path::PathBuf;

use egui::RichText;
use subcraft_app::{ExportSettings, ExportTemplate, FileFormat, Padding, Session};

use crate::state::UiState;
use crate::theme::{self, TEXT_DIM};
use crate::widgets::{caption, divider, primary_button, secondary_button};

fn dim(text: &str) -> RichText {
    RichText::new(text).color(TEXT_DIM)
}

fn window_title(text: &str) -> RichText {
    RichText::new(text).font(theme::font_strong(14.0))
}

/// Something that would discard unsaved work; asked about first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pending {
    Close,
    NewSample,
    OpenMesh,
    OpenProject,
    OpenPath,
}

pub struct ExportUi {
    pub open: bool,
    pub settings: ExportSettings,
    pub template: usize,
    pub folder: String,
}

impl Default for ExportUi {
    fn default() -> Self {
        #[cfg(target_arch = "wasm32")]
        let folder = PathBuf::from(crate::files::DOWNLOADS);
        #[cfg(not(target_arch = "wasm32"))]
        let folder = std::env::current_dir().map(|d| d.join("exports")).unwrap_or_else(|_| PathBuf::from("exports"));
        Self { open: false, settings: ExportSettings::default(), template: 0, folder: folder.display().to_string() }
    }
}

const PROJECT_FILTER: (&str, &[&str]) = ("SubCraft project", &[subcraft_app::PROJECT_EXTENSION]);

/// Save to the current path, or ask for one. Returns true when saved.
pub fn save(state: &mut UiState, save_as: bool, now: f64) -> bool {
    let path = if save_as || state.session.project_path().is_none() {
        let name = format!("{}.{}", state.session.export_name(), subcraft_app::PROJECT_EXTENSION);
        // In the browser there is no save dialog: the project is downloaded under its name.
        #[cfg(target_arch = "wasm32")]
        let picked = Some(std::path::Path::new(crate::files::DOWNLOADS).join(name));
        #[cfg(not(target_arch = "wasm32"))]
        let picked = rfd::FileDialog::new().add_filter(PROJECT_FILTER.0, PROJECT_FILTER.1).set_file_name(name).save_file();
        match picked {
            Some(mut p) => {
                if p.extension().is_none() {
                    p.set_extension(subcraft_app::PROJECT_EXTENSION);
                }
                Some(p)
            }
            None => return false,
        }
    } else {
        None
    };
    match state.session.save(path.as_deref()) {
        Ok(p) => {
            state.set_status(format!("Saved {}", p.display()), now);
            true
        }
        Err(e) => {
            state.set_status(format!("Save failed: {e}"), now);
            false
        }
    }
}

/// Run an action that replaces the session (after any unsaved-changes prompt).
pub fn run_pending(state: &mut UiState, action: Pending, ctx: &egui::Context, now: f64) {
    match action {
        Pending::Close => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        Pending::NewSample => state.set_session(Session::sample()),
        // In the browser a .gltf can come with its .bin, so meshes allow several files.
        Pending::OpenMesh => crate::files::pick(ctx, crate::files::Purpose::OpenMesh, crate::files::MESHES, cfg!(target_arch = "wasm32"), None),
        Pending::OpenProject => crate::files::pick(ctx, crate::files::Purpose::OpenProject, PROJECT_FILTER, false, None),
        Pending::OpenPath => {
            if let Some(path) = state.pending_path.take() { open_project(state, &path, now); }
        }
    }
}

pub fn open_project(state: &mut UiState, path: &std::path::Path, now: f64) {
    match Session::open_project(path) {
        Ok(s) => {
            state.set_session(s);
            state.set_status(format!("Opened {}", path.display()), now);
        }
        Err(e) => state.set_status(format!("Could not open project: {e}"), now),
    }
}

/// "Save changes?" modal. Returns the action to run once the user decided.
pub fn unsaved_prompt(state: &mut UiState, ctx: &egui::Context, now: f64) -> Option<Pending> {
    let pending = state.confirm?;
    let mut decided = None;
    egui::Modal::new(egui::Id::new("unsaved")).show(ctx, |ui| {
        ui.set_width(340.0);
        ui.label(window_title("Unsaved changes"));
        ui.add_space(4.0);
        ui.label(dim("Save your changes before continuing?"));
        ui.add_space(14.0);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if primary_button(ui, "Save").clicked() && save(state, false, now) {
                decided = Some(Some(pending));
            }
            if secondary_button(ui, "Don't save").clicked() {
                decided = Some(Some(pending));
            }
            if secondary_button(ui, "Cancel").clicked() {
                decided = Some(None);
            }
        });
    });
    match decided {
        Some(result) => {
            state.confirm = None;
            if result == Some(Pending::Close) {
                state.allow_close = true;
            }
            result
        }
        None => None,
    }
}

pub fn export_window(state: &mut UiState, ctx: &egui::Context, now: f64) {
    if !state.export.open {
        return;
    }
    let mut open = true;
    let templates = ExportTemplate::builtin();
    egui::Window::new(window_title("Export Textures")).open(&mut open).collapsible(false).default_width(480.0).show(ctx, |ui| {
        let ex = &mut state.export;
        egui::Grid::new("export_grid").num_columns(2).spacing([16.0, 10.0]).show(ui, |ui| {
            #[cfg(not(target_arch = "wasm32"))]
            {
                ui.label(dim("Folder"));
                ui.horizontal(|ui| {
                    ui.add(egui::TextEdit::singleline(&mut ex.folder).desired_width(300.0));
                    if ui.button("Browse…").clicked()
                        && let Some(p) = rfd::FileDialog::new().set_directory(&ex.folder).pick_folder()
                    {
                        ex.folder = p.display().to_string();
                    }
                });
                ui.end_row();
            }
            #[cfg(target_arch = "wasm32")]
            {
                ui.label(dim("Files"));
                ui.label(dim("Downloaded by your browser (it may ask to allow several downloads)."));
                ui.end_row();
            }
            ui.label(dim("Template"));
            egui::ComboBox::from_id_salt("template").selected_text(&templates[ex.template].name).show_ui(ui, |ui| {
                for (i, t) in templates.iter().enumerate() {
                    ui.selectable_value(&mut ex.template, i, &t.name);
                }
            });
            ui.end_row();
            ui.label(dim("File format"));
            let fmt_text = ex.settings.format_override.map_or("Template default", |f| f.label());
            egui::ComboBox::from_id_salt("format").selected_text(fmt_text).show_ui(ui, |ui| {
                ui.selectable_value(&mut ex.settings.format_override, None, "Template default");
                for f in FileFormat::ALL {
                    ui.selectable_value(&mut ex.settings.format_override, Some(f), f.label());
                }
            });
            ui.end_row();
            ui.label(dim("Padding"));
            ui.horizontal(|ui| {
                let mut kind = match ex.settings.padding {
                    Padding::Infinite => 0,
                    Padding::Texels(_) => 1,
                    Padding::None => 2,
                };
                let before = kind;
                ui.selectable_value(&mut kind, 0, "Infinite");
                ui.selectable_value(&mut kind, 1, "Texels");
                ui.selectable_value(&mut kind, 2, "None");
                if kind != before {
                    ex.settings.padding = match kind {
                        0 => Padding::Infinite,
                        1 => Padding::Texels(8),
                        _ => Padding::None,
                    };
                }
                if let Padding::Texels(t) = &mut ex.settings.padding {
                    ui.add(egui::DragValue::new(t).range(1..=64));
                }
            });
            ui.end_row();
            ui.label(dim("Height to normal"));
            ui.add(egui::Slider::new(&mut ex.settings.height_to_normal, 0.0..=0.1));
            ui.end_row();
        });
        ex.settings.template = templates[ex.template].clone();
        let settings = ex.settings.clone();
        let folder = PathBuf::from(&ex.folder);
        ui.add_space(6.0);
        divider(ui);
        ui.add_space(4.0);
        let plan = state.session.export_plan(&settings);
        ui.label(caption(&format!("{} files", plan.len())));
        egui::Frame::new().fill(theme::FIELD).inner_margin(egui::Margin::same(8)).show(ui, |ui| {
            egui::ScrollArea::vertical().max_height(170.0).show(ui, |ui| {
                ui.set_min_width(ui.available_width());
                for (_, f) in &plan {
                    ui.label(RichText::new(f).monospace().small().color(TEXT_DIM));
                }
            });
        });
        ui.add_space(6.0);
        let go = ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_enabled_ui(!plan.is_empty(), |ui| primary_button(ui, format!("{}  Export", crate::icons::EXPORT))).inner.clicked()
        });
        if go.inner {
            match state.session.export_textures(&folder, &settings) {
                #[cfg(not(target_arch = "wasm32"))]
                Ok(files) => state.set_status(format!("Exported {} files to {}", files.len(), folder.display()), now),
                #[cfg(target_arch = "wasm32")]
                Ok(files) => state.set_status(format!("Downloaded {} files", files.len()), now),
                Err(e) => state.set_status(format!("Export failed: {e}"), now),
            }
        }
    });
    state.export.open = open;
}

/// Window title with project name and unsaved marker.
pub fn title(state: &UiState) -> String {
    let name = state
        .session
        .project_path()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("{} (unsaved)", state.session.scene().mesh.name));
    let star = if state.session.is_modified() { " *" } else { "" };
    format!("SubCraft — {name}{star}")
}

// ----- baking ----------------------------------------------------------------------------------

pub struct BakeUi {
    pub open: bool,
    pub settings: subcraft_app::BakeSettings,
    pub high_poly: String,
    pub all_sets: bool,
    pub job: Option<subcraft_app::BakeJob>,
    pub acceleration: crate::panels::baking::Acceleration,
    pub selected_map: subcraft_core::MeshMapKind,
    pub view_cage: bool,
    pub maximized: bool,
    pub cage_opacity: f32,
    pub cage_show_low: bool,
    pub cage_show_front: bool,
    pub cage_show_rear: bool,
    pub auto_follow: bool,
    pub preview_textures: std::collections::HashMap<subcraft_core::MeshMapKind, egui::TextureHandle>,
    pub preview_versions: std::collections::HashMap<subcraft_core::MeshMapKind, u64>,
    pub preview_set: Option<subcraft_core::TextureSetId>,
    pub preview_generation: u64,
    pub preview_texture_revision: u64,
    pub preview_zoom: f32,
    pub preview_pan: egui::Vec2,
    pub started_at: Option<f64>,
    pub elapsed_seconds: f64,
    pub last_stage: String,
    pub last_fraction: f32,
    pub current_map: Option<subcraft_core::MeshMapKind>,
    pub map_fraction: f32,
    pub last_map_index: usize,
    pub last_maps_total: usize,
    pub backend_label: String,
    pub cancel_requested: bool,
}

impl Default for BakeUi {
    fn default() -> Self {
        Self {
            open: false, settings: subcraft_app::BakeSettings::default(), high_poly: String::new(), all_sets: true, job: None,
            acceleration: crate::panels::baking::Acceleration::Automatic,
            selected_map: subcraft_core::MeshMapKind::Normal,
            view_cage: false, maximized: true, cage_opacity: 0.45, cage_show_low: true,
            cage_show_front: true, cage_show_rear: false, auto_follow: true,
            preview_textures: Default::default(), preview_versions: Default::default(),
            preview_set: None, preview_generation: u64::MAX, preview_texture_revision: u64::MAX,
            preview_zoom: 1.0, preview_pan: egui::Vec2::ZERO,
            started_at: None, elapsed_seconds: 0.0, last_stage: "Pronto para bake".into(),
            last_fraction: 0.0, current_map: None, map_fraction: 0.0,
            last_map_index: 0, last_maps_total: 0, backend_label: String::new(), cancel_requested: false,
        }
    }
}

/// Start one worker. Device initialization and all baking stay off the UI thread.
pub fn start_bake(state: &mut UiState, now: f64) {
    if state.bake.job.is_some() { return; }
    let mut settings = state.bake.settings.clone();
    settings.maps.sort();
    settings.maps.dedup();
    if let Err(error) = settings.validate().and_then(|()| settings.validate_memory_budget()) {
        state.set_status(error.to_string(), now);
        return;
    }
    let texture_sets = if state.bake.all_sets {
        state.session.document().texture_sets.iter().map(|t| (t.id, t.material_slot)).collect()
    } else {
        let selected = state.session.selected_texture_set();
        vec![(selected.id, selected.material_slot)]
    };
    let request = subcraft_app::BakeRequest {
        settings,
        high_poly: (!state.bake.high_poly.trim().is_empty()).then(|| PathBuf::from(state.bake.high_poly.trim())),
        texture_sets,
    };
    let backend = state.bake_backends.backend(state.bake.acceleration);
    let b = &mut state.bake;
    b.preview_textures.clear();
    b.preview_versions.clear();
    b.preview_set = None;
    b.preview_generation = u64::MAX;
    b.preview_texture_revision = u64::MAX;
    b.preview_zoom = 1.0;
    b.preview_pan = egui::Vec2::ZERO;
    b.started_at = Some(now);
    b.elapsed_seconds = 0.0;
    b.last_stage = "Preparando bake…".into();
    b.last_fraction = 0.0;
    b.current_map = None;
    b.map_fraction = 0.0;
    b.last_map_index = 0;
    b.last_maps_total = b.settings.maps.len();
    b.backend_label = backend.name().into();
    b.cancel_requested = false;
    b.job = Some(state.session.start_bake_with_backend(request, backend));
    state.set_status("Bake iniciado. Acompanhe os mapas na aba Bake Studio.", now);
}

/// Capture even the final preview before consuming completion and installing the maps.
pub fn poll_bake(state: &mut UiState, ctx: &egui::Context, now: f64) {
    crate::panels::baking::capture_progress(state, ctx, now);
    let Some(job) = state.bake.job.as_mut() else { return };
    ctx.request_repaint_after(std::time::Duration::from_millis(100));
    if let Some(result) = job.poll(&mut state.session) {
        // Completion can race the capture at the top of this frame.
        crate::panels::baking::capture_progress(state, ctx, now);
        state.bake.job = None;
        state.bake.started_at = None;
        let message = match result {
            Ok(n) => {
                state.bake.last_fraction = 1.0;
                state.bake.map_fraction = 1.0;
                format!("Bake concluído: {n} texture set(s) · {} · {:.1}s", state.bake.backend_label, state.bake.elapsed_seconds)
            }
            Err(e) => format!("Bake interrompido: {e}"),
        };
        log::info!("{message}");
        state.bake.last_stage = message.clone();
        state.set_status(message, now);
    }
}
