//! Dedicated, non-blocking mesh-map baking workspace.
//!
//! The worker owns rasterization and ray casting. This panel only uploads bounded previews
//! whose revision changed, so opening the workspace does not turn the UI into a second baker.

use egui::{Color32, Rect, RichText, Sense, Stroke, Ui};
use subcraft_core::MeshMapKind;

use crate::state::UiState;
use crate::theme;
use crate::widgets::{self, caption, divider, hint, primary_button, secondary_button};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Acceleration {
    #[default]
    Automatic,
    Gpu,
    Cpu,
}

impl Acceleration {
    pub fn label(self) -> &'static str {
        match self {
            Self::Automatic => "Auto · GPU / CPU",
            Self::Gpu => "GPU",
            Self::Cpu => "CPU",
        }
    }
}

pub fn map_label(kind: MeshMapKind) -> &'static str {
    match kind {
        MeshMapKind::Normal => "Normal",
        MeshMapKind::WorldSpaceNormal => "World Space Normal",
        MeshMapKind::Id => "ID de peças",
        MeshMapKind::AmbientOcclusion => "Ambient Occlusion",
        MeshMapKind::Curvature => "Curvatura",
        MeshMapKind::Position => "Posição",
        MeshMapKind::Thickness => "Espessura",
    }
}

/// Capture before polling completion, including when the Bake tab is not visible.
pub fn capture_progress(state: &mut UiState, ctx: &egui::Context, now: f64) {
    let Some(job) = state.bake.job.as_ref() else {
        return;
    };
    let snapshot = job.snapshot();
    let previews = job.previews();
    let stage = job.stage();
    let fraction = job.fraction();
    let current_set = job.current_texture_set();
    let stage = current_set
        .and_then(|id| {
            state
                .session
                .document()
                .texture_sets
                .iter()
                .find(|set| set.id == id)
        })
        .map_or_else(|| stage.clone(), |set| format!("{} · {stage}", set.name));
    let b = &mut state.bake;
    if current_set.is_some() && b.preview_set != current_set {
        b.preview_textures.clear();
        b.preview_versions.clear();
        b.preview_set = current_set;
    }
    b.elapsed_seconds = b.started_at.map_or(0.0, |start| (now - start).max(0.0));
    b.last_stage = stage;
    b.last_fraction = fraction;
    b.current_map = snapshot.map;
    b.map_fraction = snapshot.map_fraction;
    b.last_map_index = snapshot.map_index;
    b.last_maps_total = snapshot.maps_total;
    b.backend_label = snapshot.backend;
    if b.auto_follow
        && let Some(kind) = snapshot.map
    {
        b.selected_map = kind;
    }
    for preview in previews {
        if b.preview_versions.get(&preview.kind) == Some(&preview.revision) {
            continue;
        }
        let image = egui::ColorImage::from_rgba_unmultiplied(
            [preview.width as usize, preview.height as usize],
            &preview.rgba,
        );
        if let Some(texture) = b.preview_textures.get_mut(&preview.kind) {
            texture.set(image, egui::TextureOptions::LINEAR);
        } else {
            b.preview_textures.insert(
                preview.kind,
                ctx.load_texture(
                    format!("bake-preview-{}", preview.kind.key()),
                    image,
                    egui::TextureOptions::LINEAR,
                ),
            );
        }
        b.preview_versions.insert(preview.kind, preview.revision);
    }
}

/// Bake controls and their live result share a tab; the existing painting workspace stays intact.
pub fn show(state: &mut UiState, ui: &mut Ui, now: f64) {
    let size = ui.available_size();
    if size.x < 610.0 {
        egui::ScrollArea::vertical()
            .id_salt("bake-narrow")
            .show(ui, |ui| {
                configuration(state, ui);
                action_bar(state, ui, now);
                ui.add_space(10.0);
                ui.allocate_ui(egui::vec2(ui.available_width(), 420.0), |ui| {
                    preview(state, ui)
                });
            });
        return;
    }
    let left = (size.x * 0.29).clamp(278.0, 340.0);
    ui.horizontal_top(|ui| {
        ui.allocate_ui_with_layout(
            egui::vec2(left, size.y),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                ui.set_min_width(left);
                egui::Frame::new()
                    .fill(theme::BG_1)
                    .inner_margin(12)
                    .show(ui, |ui| {
                        // Work from the frame's inner height: its vertical padding is already
                        // excluded. Reserve a fixed footer so Cancel stays reachable at 500 px.
                        let actions = 116.0;
                        let settings_height = (ui.available_height() - actions).max(32.0);
                        egui::ScrollArea::vertical()
                            .id_salt("bake-config")
                            .max_height(settings_height)
                            .auto_shrink([false, false])
                            .show(ui, |ui| configuration(state, ui));
                        divider(ui);
                        action_bar(state, ui, now);
                    });
            },
        );
        let width = ui.available_width();
        ui.allocate_ui_with_layout(
            egui::vec2(width, size.y),
            egui::Layout::top_down(egui::Align::Min),
            |ui| preview(state, ui),
        );
    });
}

fn configuration(state: &mut UiState, ui: &mut Ui) {
    let busy = state.bake.job.is_some();
    let set_name = state.session.selected_texture_set().name.clone();
    ui.label(RichText::new("Bake de mesh maps").font(theme::font_strong(18.0)));
    hint(
        ui,
        "Prepare os mapas que alimentam materiais e smart masks.",
    );
    ui.add_space(8.0);
    ui.add_enabled_ui(!busy, |ui| {
        widgets::section(ui, "bake-input", "Malhas e conjuntos", true, |ui| {
            let b = &mut state.bake;
            ui.label(caption("High poly · opcional"));
            ui.horizontal(|ui| {
                let width = (ui.available_width() - 64.0).max(80.0);
                ui.add(egui::TextEdit::singleline(&mut b.high_poly)
                    .hint_text("Sem arquivo = low sobre low")
                    .desired_width(width));
                if secondary_button(ui, "Abrir").clicked() {
                    crate::files::pick(ui.ctx(), crate::files::Purpose::BakeHighPoly,
                        crate::files::MESHES, cfg!(target_arch = "wasm32"), None);
                }
            });
            if !b.high_poly.is_empty() && ui.small_button("Remover high poly").clicked() {
                b.high_poly.clear();
            }
            hint(ui, "Somente o low poly precisa de UVs. Esculturas high poly podem estar sem UV.");
            ui.add_space(6.0);
            ui.checkbox(&mut b.all_sets, "Todos os texture sets");
            if !b.all_sets { hint(ui, &format!("Selecionado: {set_name}")); }
        });
        widgets::section(ui, "bake-common", "Qualidade e recursos", true, |ui| {
            let b = &mut state.bake;
            ui.horizontal(|ui| {
                for (label, size, ao, thickness, aa) in [
                    ("Rápido", 512, 8, 8, 1),
                    ("Equilibrado", 1024, 32, 16, 1),
                    ("Detalhado", 2048, 64, 32, 1),
                ] {
                    if ui.small_button(label).clicked() {
                        b.settings.size = size;
                        b.settings.ao_samples = ao;
                        b.settings.thickness_samples = thickness;
                        b.settings.supersampling = aa;
                    }
                }
            });
            widgets::prop_row(ui, "Resolução", |ui| {
                egui::ComboBox::from_id_salt("bake-size")
                    .selected_text(format!("{} × {}", b.settings.size, b.settings.size))
                    .show_ui(ui, |ui| {
                        for size in [128, 256, 512, 1024, 2048, 4096] {
                            ui.selectable_value(&mut b.settings.size, size, format!("{size} × {size}"));
                        }
                    });
            });
            widgets::prop_row(ui, "Anti-aliasing", |ui| {
                egui::ComboBox::from_id_salt("bake-aa")
                    .selected_text(aa_label(b.settings.supersampling))
                    .show_ui(ui, |ui| {
                        for factor in [1, 2, 4] {
                            ui.selectable_value(&mut b.settings.supersampling, factor, aa_label(factor));
                        }
                    });
            });
            widgets::prop_row(ui, "Processador", |ui| {
                egui::ComboBox::from_id_salt("bake-backend")
                    .selected_text(b.acceleration.label()).show_ui(ui, |ui| {
                        for mode in [Acceleration::Automatic, Acceleration::Gpu, Acceleration::Cpu] {
                            ui.selectable_value(&mut b.acceleration, mode, mode.label());
                        }
                    });
            });
            let processors = std::thread::available_parallelism().map_or(1, |n| n.get().min(32) as u32);
            widgets::prop_row(ui, "Threads CPU", |ui| {
                ui.add(egui::DragValue::new(&mut b.settings.cpu_threads)
                    .range(0..=processors).speed(1).prefix(" "))
                    .on_hover_text("0 = limite automático conservador. Também limita a preparação usada pelo bake GPU.");
            });
            widgets::prop_row(ui, "RAM · MiB", |ui| {
                ui.add(egui::DragValue::new(&mut b.settings.memory_limit_mb)
                    .range(256..=8192).speed(64));
            });
            hint(ui, "Auto usa GPU compatível com fallback CPU. Lotes limitados evitam trabalhos GPU longos; o limite de RAM é validado antes de alocar.");
            let sampled = b.settings.size as u64 * b.settings.supersampling as u64;
            hint(ui, &format!("{} mapas · {}² amostras · AA {}× trabalho",
                b.settings.maps.len(), sampled, b.settings.supersampling.pow(2)));
            hint(ui, &format!("Estimativa: {} MiB + malhas / BVH · limite {} MiB",
                b.settings.estimated_memory_mb(), b.settings.memory_limit_mb));
        });
    });

    widgets::section(ui, "bake-maps", "Mapas e configurações", true, |ui| {
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!busy, |ui| {
                if ui.small_button("Todos").clicked() {
                    state.bake.settings.maps = MeshMapKind::ALL.to_vec();
                }
                if ui.small_button("Nenhum").clicked() {
                    state.bake.settings.maps.clear();
                }
            });
            ui.add(
                egui::Label::new(
                    RichText::new("Clique para configurar")
                        .small()
                        .color(theme::TEXT_FAINT),
                )
                .truncate(),
            );
        });
        for kind in MeshMapKind::ALL {
            ui.push_id(kind.key(), |ui| {
                ui.horizontal(|ui| {
                    let mut enabled = state.bake.settings.maps.contains(&kind);
                    if ui
                        .add_enabled(!busy, egui::Checkbox::without_text(&mut enabled))
                        .changed()
                    {
                        if enabled {
                            state.bake.settings.maps.push(kind);
                        } else {
                            state.bake.settings.maps.retain(|&map| map != kind);
                        }
                    }
                    if ui
                        .selectable_label(state.bake.selected_map == kind, map_label(kind))
                        .clicked()
                    {
                        state.bake.selected_map = kind;
                        state.bake.auto_follow = false;
                    }
                    if busy && state.bake.current_map == Some(kind) {
                        ui.spinner();
                    } else if state.bake.preview_textures.contains_key(&kind) {
                        ui.label(RichText::new("●").color(theme::ACCENT));
                    }
                });
            });
        }
        ui.add_space(6.0);
        divider(ui);
        ui.add_space(6.0);
        ui.label(RichText::new(map_label(state.bake.selected_map)).strong());
        ui.add_enabled_ui(!busy, |ui| map_settings(&mut state.bake, ui));
    });

    widgets::section(ui, "bake-cage", "Projeção e cage", true, |ui| {
        ui.add_enabled_ui(!busy, |ui| {
            let b = &mut state.bake;
            widgets::prop_row(ui, "Frente", |ui| {
                ui.add(egui::DragValue::new(&mut b.settings.cage_front)
                    .speed(0.001).range(0.0..=1.0).max_decimals(4));
            });
            widgets::prop_row(ui, "Trás", |ui| {
                ui.add(egui::DragValue::new(&mut b.settings.cage_rear)
                    .speed(0.001).range(0.0..=1.0).max_decimals(4));
            });
            widgets::prop_row(ui, "Ray bias", |ui| {
                ui.add(egui::DragValue::new(&mut b.settings.ray_bias)
                    .speed(0.00001).range(0.000001..=0.01).max_decimals(6));
            });
            hint(ui, "Frente e trás usam a diagonal do low poly. Ray bias usa a maior diagonal entre low e high. A cage segue as normais do low poly.");
        });
        if secondary_button(ui, "Visualizar cage em 3D").clicked() {
            state.bake.view_cage = true;
        }
    });
}

fn aa_label(factor: u32) -> &'static str {
    match factor {
        2 => "4 amostras / pixel",
        4 => "16 amostras / pixel",
        _ => "1 amostra / pixel",
    }
}

fn map_settings(b: &mut crate::dialogs::BakeUi, ui: &mut Ui) {
    widgets::fill_width(ui);
    match b.selected_map {
        MeshMapKind::Normal => {
            ui.checkbox(
                &mut b.settings.normal_flip_y,
                "DirectX · inverter eixo Y (verde)",
            );
            hint(
                ui,
                "Desmarcado: OpenGL (+Y). A normal usa o espaço tangente do low poly e captura detalhes do high poly.",
            );
        }
        MeshMapKind::WorldSpaceNormal => {
            hint(
                ui,
                "Normal no espaço da malha. RGB representa XYZ. Usa a mesma cage da normal tangente.",
            );
        }
        MeshMapKind::Id => {
            widgets::prop_row(ui, "Semente", |ui| {
                ui.add(egui::DragValue::new(&mut b.settings.id_seed).speed(1));
            });
            hint(
                ui,
                "Uma cor estável por peça da malha. Altere a semente para gerar outra paleta de seleção.",
            );
        }
        MeshMapKind::AmbientOcclusion => {
            widgets::slider_row(
                ui,
                "Amostras",
                egui::Slider::new(&mut b.settings.ao_samples, 4..=256).logarithmic(true),
            );
            widgets::slider_row(
                ui,
                "Distância",
                egui::Slider::new(&mut b.settings.ao_distance, 0.001..=1.0)
                    .logarithmic(true)
                    .max_decimals(4),
            );
            hint(
                ui,
                "Mais amostras reduzem ruído. A distância limita o alcance das sombras de contato e alimenta máscaras de cavidade.",
            );
        }
        MeshMapKind::Curvature => {
            widgets::slider_row(
                ui,
                "Raio",
                egui::Slider::new(&mut b.settings.curvature_radius, 0.001..=0.2)
                    .logarithmic(true)
                    .max_decimals(4),
            );
            hint(
                ui,
                "Raio da vizinhança geométrica. Cinza = plano, claro = convexo, escuro = côncavo. Alimenta máscaras de desgaste nas bordas.",
            );
        }
        MeshMapKind::Position => {
            hint(
                ui,
                "XYZ normalizado aos limites do low poly. Utilizado por gradientes e máscaras espaciais. Resolução e cage seguem os controles comuns.",
            );
        }
        MeshMapKind::Thickness => {
            widgets::slider_row(
                ui,
                "Amostras",
                egui::Slider::new(&mut b.settings.thickness_samples, 4..=256).logarithmic(true),
            );
            widgets::slider_row(
                ui,
                "Distância",
                egui::Slider::new(&mut b.settings.thickness_distance, 0.001..=1.0)
                    .logarithmic(true)
                    .max_decimals(4),
            );
            hint(
                ui,
                "Raios para dentro da superfície medem espessura. Valores claros indicam regiões finas.",
            );
        }
    }
}

fn action_bar(state: &mut UiState, ui: &mut Ui, now: f64) {
    ui.add_space(8.0);
    if let Some(job) = state.bake.job.as_ref() {
        ui.add(
            egui::Label::new(
                RichText::new(&state.bake.last_stage)
                    .small()
                    .color(theme::TEXT_DIM),
            )
            .truncate(),
        )
        .on_hover_text(&state.bake.last_stage);
        ui.add(
            egui::ProgressBar::new(state.bake.last_fraction)
                .text(
                    RichText::new(format!("Total · {:.0}%", state.bake.last_fraction * 100.0))
                        .color(theme::TEXT),
                )
                .fill(theme::ACCENT)
                .desired_width(ui.available_width()),
        );
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(elapsed_label(state.bake.elapsed_seconds))
                    .small()
                    .color(theme::TEXT_DIM),
            );
            let text = if state.bake.cancel_requested {
                "Cancelando…"
            } else {
                "Cancelar bake"
            };
            if ui
                .add_enabled_ui(!state.bake.cancel_requested, |ui| {
                    secondary_button(ui, text)
                })
                .inner
                .clicked()
            {
                job.cancel();
                state.bake.cancel_requested = true;
            }
        });
    } else {
        let validation = state
            .bake
            .settings
            .validate()
            .and_then(|()| state.bake.settings.validate_memory_budget());
        let can_bake = validation.is_ok();
        if let Err(error) = validation {
            ui.add(
                egui::Label::new(
                    RichText::new(error.to_string())
                        .small()
                        .color(theme::WARNING),
                )
                .truncate(),
            )
            .on_hover_text(error.to_string());
        }
        if ui
            .add_enabled_ui(can_bake, |ui| {
                primary_button(ui, format!("{}  Iniciar bake", crate::icons::MAGIC_WAND))
            })
            .inner
            .clicked()
        {
            crate::dialogs::start_bake(state, now);
        }
        if state.bake.elapsed_seconds > 0.0 {
            hint(
                ui,
                &format!("Último bake: {}", elapsed_label(state.bake.elapsed_seconds)),
            );
        }
    }
}

fn preview(state: &mut UiState, ui: &mut Ui) {
    let busy = state.bake.job.is_some();
    if !busy {
        load_document_previews(state, ui.ctx());
    }
    egui::Frame::new()
        .fill(theme::BG_2)
        .inner_margin(8)
        .show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                let maximize = if state.bake.maximized {
                    "Recolher"
                } else {
                    "Expandir"
                };
                if ui
                    .small_button(maximize)
                    .on_hover_text("Alternar entre a aba no layout e o espaço de bake completo")
                    .clicked()
                {
                    state.bake.maximized = !state.bake.maximized;
                }
                widgets::segmented(
                    ui,
                    &mut state.bake.view_cage,
                    &[(false, "Mapas 2D"), (true, "Cage 3D")],
                );
                ui.separator();
                if state.bake.view_cage {
                    ui.checkbox(&mut state.bake.cage_show_low, "Low");
                    ui.checkbox(&mut state.bake.cage_show_front, "Frente");
                    ui.checkbox(&mut state.bake.cage_show_rear, "Trás");
                    ui.add(
                        egui::Slider::new(&mut state.bake.cage_opacity, 0.05..=1.0)
                            .text("Opacidade")
                            .show_value(false),
                    );
                } else {
                    ui.checkbox(&mut state.bake.auto_follow, "Seguir mapa atual");
                    if ui.small_button("Enquadrar").clicked() {
                        state.bake.preview_zoom = 1.0;
                        state.bake.preview_pan = egui::Vec2::ZERO;
                    }
                }
            });
        });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let title = if state.bake.view_cage {
            "Cage de projeção"
        } else {
            map_label(state.bake.selected_map)
        };
        ui.label(RichText::new(title).font(theme::font_strong(16.0)));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if !state.bake.backend_label.is_empty() {
                ui.add(
                    egui::Label::new(
                        RichText::new(&state.bake.backend_label)
                            .small()
                            .color(theme::ACCENT),
                    )
                    .truncate(),
                )
                .on_hover_text(&state.bake.backend_label);
            }
            if busy {
                ui.spinner();
            }
        });
    });
    let reserved = if state.bake.view_cage { 88.0 } else { 178.0 };
    let size = egui::vec2(
        ui.available_width(),
        (ui.available_height() - reserved).max(120.0),
    );
    ui.allocate_ui_with_layout(size, egui::Layout::top_down(egui::Align::Min), |ui| {
        if state.bake.view_cage {
            crate::viewport::show_bake_cage(state, ui);
        } else {
            texture_canvas(state, ui);
        }
    });
    if !state.bake.view_cage {
        ui.add_space(6.0);
        thumbnails(state, ui);
    } else {
        hint(
            ui,
            "Frente: ciano · trás: laranja. Arraste para orbitar; roda do mouse para aproximar.",
        );
    }
    ui.add_space(6.0);
    progress_footer(state, ui);
}

fn progress_footer(state: &UiState, ui: &mut Ui) {
    let b = &state.bake;
    divider(ui);
    ui.add_space(6.0);
    if b.job.is_some() {
        ui.horizontal_wrapped(|ui| {
            if let Some(kind) = b.current_map {
                ui.label(
                    RichText::new(format!(
                        "Agora: {} · {}/{}",
                        map_label(kind),
                        b.last_map_index,
                        b.last_maps_total
                    ))
                    .strong()
                    .color(theme::ACCENT),
                );
            }
            ui.label(
                RichText::new(elapsed_label(b.elapsed_seconds))
                    .small()
                    .color(theme::TEXT_DIM),
            );
        });
        let phase = b.job.as_ref().expect("bake is running").snapshot().phase;
        let phase_label = match phase {
            subcraft_bake::BakePhase::Preparing => "Preparação",
            subcraft_bake::BakePhase::Projecting => "Projeção",
            subcraft_bake::BakePhase::Baking => "Cálculo",
            subcraft_bake::BakePhase::Padding => "Padding UV",
            subcraft_bake::BakePhase::Completed => "Concluído",
        };
        ui.add(
            egui::ProgressBar::new(b.map_fraction)
                .fill(theme::ACCENT)
                .text(
                    RichText::new(format!(
                        "Mapa atual · {phase_label} · {:.0}%",
                        b.map_fraction * 100.0
                    ))
                    .color(theme::TEXT),
                ),
        )
        .on_hover_text(&b.last_stage);
    } else {
        ui.label(RichText::new(&b.last_stage).small().color(theme::TEXT_DIM));
        hint(
            ui,
            "As prévias são atualizadas durante o cálculo. Os mapas finais mantêm a resolução configurada.",
        );
    }
}

fn texture_canvas(state: &mut UiState, ui: &mut Ui) {
    let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
    if response.hovered() {
        let scroll = ui.input(|input| input.smooth_scroll_delta.y);
        state.bake.preview_zoom =
            (state.bake.preview_zoom * (scroll * 0.002).exp()).clamp(0.2, 16.0);
    }
    if response.dragged() {
        state.bake.preview_pan += response.drag_delta();
    }
    if response.double_clicked() {
        state.bake.preview_zoom = 1.0;
        state.bake.preview_pan = egui::Vec2::ZERO;
    }
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, theme::BG_0);
    let side = (rect.width().min(rect.height()) - 28.0).max(1.0) * state.bake.preview_zoom;
    let image_rect = Rect::from_center_size(
        rect.center() + state.bake.preview_pan,
        egui::Vec2::splat(side),
    );
    checkerboard(&painter, image_rect.intersect(rect));
    if let Some(texture) = state.bake.preview_textures.get(&state.bake.selected_map) {
        painter.image(
            texture.id(),
            image_rect,
            Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
            Color32::WHITE,
        );
        painter.rect_stroke(
            image_rect,
            0.0,
            Stroke::new(1.0, theme::BORDER_STRONG),
            egui::StrokeKind::Inside,
        );
    } else {
        painter.text(
            rect.center() - egui::vec2(0.0, 12.0),
            egui::Align2::CENTER_CENTER,
            "A textura aparecerá durante o bake",
            theme::font_strong(14.0),
            theme::TEXT_DIM,
        );
        painter.text(
            rect.center() + egui::vec2(0.0, 13.0),
            egui::Align2::CENTER_CENTER,
            "Selecione os mapas e clique em Iniciar bake",
            theme::font(12.0),
            theme::TEXT_FAINT,
        );
    }
    painter.text(
        rect.left_bottom() + egui::vec2(10.0, -10.0),
        egui::Align2::LEFT_BOTTOM,
        format!(
            "{}% · arraste para mover · roda para zoom",
            (state.bake.preview_zoom * 100.0).round()
        ),
        theme::font(10.0),
        theme::TEXT_DIM,
    );
}

fn checkerboard(painter: &egui::Painter, rect: Rect) {
    let cell = 24.0;
    for y in 0..(rect.height() / cell).ceil() as u32 {
        for x in 0..(rect.width() / cell).ceil() as u32 {
            let min = rect.min + egui::vec2(x as f32 * cell, y as f32 * cell);
            let tile = Rect::from_min_size(min, egui::Vec2::splat(cell)).intersect(rect);
            let color = if (x + y) % 2 == 0 {
                theme::BG_2
            } else {
                theme::BG_3
            };
            painter.rect_filled(tile, 0.0, color);
        }
    }
}

fn thumbnails(state: &mut UiState, ui: &mut Ui) {
    egui::ScrollArea::horizontal()
        .id_salt("bake-thumbnails")
        .max_height(92.0)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                for kind in MeshMapKind::ALL {
                    let selected = state.bake.selected_map == kind;
                    let (rect, response) =
                        ui.allocate_exact_size(egui::vec2(82.0, 82.0), Sense::click());
                    let image_rect = Rect::from_min_size(
                        rect.min + egui::vec2(7.0, 3.0),
                        egui::vec2(68.0, 58.0),
                    );
                    ui.painter().rect_filled(
                        rect,
                        2.0,
                        if selected {
                            theme::ACCENT_SOFT
                        } else {
                            theme::BG_1
                        },
                    );
                    if let Some(texture) = state.bake.preview_textures.get(&kind) {
                        ui.painter().image(
                            texture.id(),
                            image_rect,
                            Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    } else {
                        ui.painter().rect_filled(image_rect, 0.0, theme::BG_3);
                        ui.painter().text(
                            image_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "—",
                            theme::font(18.0),
                            theme::TEXT_FAINT,
                        );
                    }
                    let label = match kind {
                        MeshMapKind::WorldSpaceNormal => "World normal",
                        MeshMapKind::AmbientOcclusion => "AO",
                        _ => map_label(kind),
                    };
                    ui.painter().text(
                        rect.center_bottom() - egui::vec2(0.0, 7.0),
                        egui::Align2::CENTER_BOTTOM,
                        label,
                        theme::font(10.0),
                        if selected {
                            theme::ACCENT
                        } else {
                            theme::TEXT_DIM
                        },
                    );
                    if selected {
                        ui.painter().rect_stroke(
                            rect,
                            2.0,
                            Stroke::new(1.0, theme::ACCENT),
                            egui::StrokeKind::Inside,
                        );
                    }
                    if response.on_hover_text(map_label(kind)).clicked() {
                        state.bake.selected_map = kind;
                        state.bake.auto_follow = false;
                    }
                }
            });
        });
}

/// Opening an old project also shows its baked maps. Sampling is bounded to 256² once per
/// document revision, rather than flattening full-resolution textures every paint frame.
fn load_document_previews(state: &mut UiState, ctx: &egui::Context) {
    let set = state.session.selected_texture_set().id;
    let revision = state.session.revision();
    let b = &mut state.bake;
    if b.preview_set == Some(set)
        && b.preview_generation == state.session_generation
        && b.preview_texture_revision == revision
    {
        return;
    }
    let changed_set =
        b.preview_set != Some(set) || b.preview_generation != state.session_generation;
    if changed_set {
        b.preview_textures.clear();
        b.preview_versions.clear();
    }
    b.preview_set = Some(set);
    b.preview_generation = state.session_generation;
    b.preview_texture_revision = revision;
    let Some(maps) = state.session.mesh_maps(set) else {
        return;
    };
    for kind in MeshMapKind::ALL {
        let Some(image) = maps.get(kind) else {
            continue;
        };
        let width = image.width().min(256);
        let height = image.height().min(256);
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let pixel = image.get(x * image.width() / width, y * image.height() / height);
                for component in 0..3 {
                    let value = pixel[if image.comps() == 1 { 0 } else { component }];
                    rgba.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
                }
                rgba.push(255);
            }
        }
        let image =
            egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &rgba);
        if let Some(texture) = b.preview_textures.get_mut(&kind) {
            texture.set(image, egui::TextureOptions::LINEAR);
        } else {
            b.preview_textures.insert(
                kind,
                ctx.load_texture(
                    format!("bake-document-{}", kind.key()),
                    image,
                    egui::TextureOptions::LINEAR,
                ),
            );
        }
    }
}

fn elapsed_label(seconds: f64) -> String {
    let seconds = seconds.max(0.0) as u64;
    if seconds < 60 {
        format!("{seconds}s")
    } else {
        format!("{}min {:02}s", seconds / 60, seconds % 60)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visible_text(output: &egui::FullOutput, needle: &str) -> Vec<Rect> {
        fn visit(shape: &egui::Shape, clip: Rect, needle: &str, rectangles: &mut Vec<Rect>) {
            match shape {
                egui::Shape::Text(text) if text.galley.text().contains(needle) => {
                    let rect = Rect::from_min_size(text.pos, text.galley.size());
                    if clip.contains(rect.center()) {
                        rectangles.push(rect);
                    }
                }
                egui::Shape::Vec(shapes) => {
                    for shape in shapes {
                        visit(shape, clip, needle, rectangles);
                    }
                }
                _ => {}
            }
        }
        let mut rectangles = Vec::new();
        for clipped in &output.shapes {
            visit(&clipped.shape, clipped.clip_rect, needle, &mut rectangles);
        }
        rectangles
    }

    #[test]
    fn baking_workspace_keeps_start_controls_and_both_previews_visible_without_gpu() {
        let context = egui::Context::default();
        crate::theme::apply(&context);
        let mut state = UiState::new(subcraft_app::Session::sample(), None);
        for size in [egui::vec2(900.0, 700.0), egui::vec2(1175.0, 500.0)] {
            let screen = Rect::from_min_size(egui::Pos2::ZERO, size);
            for cage in [false, true] {
                state.bake.view_cage = cage;
                for frame in 0..3 {
                    let input = egui::RawInput {
                        screen_rect: Some(screen),
                        ..Default::default()
                    };
                    let mut output = context.run_ui(input, |ui| show(&mut state, ui, frame as f64));
                    // This is a layout check. Font atlas uploads do not need a native GPU.
                    output.textures_delta.clear();
                    if frame < 2 {
                        continue;
                    }
                    let starts = visible_text(
                        &output,
                        &format!("{}  Iniciar bake", crate::icons::MAGIC_WAND),
                    );
                    assert_eq!(
                        starts.len(),
                        1,
                        "The bake action must remain outside the scrolling settings."
                    );
                    assert!(
                        screen.contains_rect(starts[0]),
                        "Start action is outside the workspace."
                    );
                    let titles =
                        visible_text(&output, if cage { "Cage de projeção" } else { "Normal" });
                    assert!(
                        titles.iter().any(|title| title.left() > starts[0].right()),
                        "The preview title must appear in its own visible column, without covering the start action."
                    );
                    assert!(
                        state.bake.job.is_none(),
                        "Displaying the workspace must never start work."
                    );
                    assert!(
                        state.renderer.is_none(),
                        "The layout must also work without a GPU renderer."
                    );
                }
            }
        }
    }
}
