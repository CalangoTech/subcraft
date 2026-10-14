//! Independent, read-only cage viewport. Does not evaluate the paint document during baking.

use crate::{orbit::OrbitCamera, theme};
use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui};
use glam::Vec3;
use std::{collections::BTreeSet, sync::Arc};
use subcraft_core::ChannelKind;
use subcraft_image::TiledImage;
use subcraft_mesh::{Aabb, Mesh};
use subcraft_render::{ViewportRenderer, egui_wgpu, wgpu_api as wgpu};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CageOptions {
    pub front: f32,
    pub rear: f32,
    pub opacity: f32,
    pub show_low: bool,
    pub show_front: bool,
    pub show_rear: bool,
    pub slot: Option<u32>,
}

#[derive(Clone, Copy, PartialEq)]
struct WireKey {
    camera: OrbitCamera,
    size: [u32; 2],
    pixels_per_point: u32,
    options: CageOptions,
}

#[derive(Default)]
pub struct CageViewport {
    generation: Option<u64>,
    camera: Option<OrbitCamera>,
    renderer: Option<ViewportRenderer>,
    texture: Option<egui::TextureId>,
    rendered: Option<(OrbitCamera, [u32; 2])>,
    edges: Vec<[u32; 2]>,
    edge_slot: Option<Option<u32>>,
    wire_key: Option<WireKey>,
    lines: Vec<([Pos2; 2], Stroke)>,
}

/// Same extrusion as the ray cage: fractions of the low-poly bounding-box diagonal.
pub fn cage_position(position: Vec3, normal: Vec3, diagonal: f32, distance: f32) -> Vec3 {
    position + normal.normalize_or(Vec3::Y) * diagonal.max(1e-6) * distance
}

impl CageViewport {
    pub fn show(
        &mut self,
        ui: &mut Ui,
        mesh: &Arc<Mesh>,
        rs: Option<&egui_wgpu::RenderState>,
        generation: u64,
        options: CageOptions,
    ) {
        if self.generation != Some(generation) {
            self.generation = Some(generation);
            self.camera = Some(OrbitCamera::framing(&mesh.bounds));
            self.rendered = None;
            self.edge_slot = None;
            self.wire_key = None;
            if let Some(rs) = rs {
                let renderer = self
                    .renderer
                    .get_or_insert_with(|| ViewportRenderer::new(&rs.device, &rs.queue));
                renderer.set_mesh(&rs.device, mesh);
                // One-pixel neutral materials keep this view independent of costly composites.
                let color = TiledImage::new(1, 1, 4, [0.19, 0.22, 0.26, 1.0]);
                for slot in 0..mesh.materials.len() as u32 {
                    renderer.upload_channel(
                        &rs.device,
                        &rs.queue,
                        slot,
                        ChannelKind::BaseColor,
                        &color,
                        None,
                    );
                }
            }
        }
        let camera = self.camera.as_mut().expect("camera initialized");
        ui.horizontal_wrapped(|ui| {
            ui.label(
                egui::RichText::new("CAGE / VOLUME DE PROJEÇÃO")
                    .strong()
                    .color(theme::TEXT),
            );
            if ui.small_button("Enquadrar").clicked() {
                let reach = mesh.bounds.diagonal() * options.front.max(options.rear).max(0.0);
                *camera = OrbitCamera::framing(&Aabb {
                    min: mesh.bounds.min - Vec3::splat(reach),
                    max: mesh.bounds.max + Vec3::splat(reach),
                });
            }
            ui.checkbox(&mut camera.orthographic, "Ortho");
        });
        let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
        if rect.width() < 8.0 || rect.height() < 8.0 {
            return;
        }
        let ppp = ui.ctx().pixels_per_point();
        let size = [
            (rect.width() * ppp).round().max(8.0) as u32,
            (rect.height() * ppp).round().max(8.0) as u32,
        ];
        let (delta, wheel, alt) =
            ui.input(|i| (i.pointer.delta(), i.smooth_scroll_delta.y, i.modifiers.alt));
        if response.hovered() && wheel != 0.0 {
            camera.zoom((-wheel * 0.0015).exp());
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            camera.pan(delta.x * ppp, delta.y * ppp, size[1] as f32);
        } else if response.dragged_by(egui::PointerButton::Secondary) && alt {
            camera.zoom((delta.y * 0.01).exp());
        } else if response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Secondary)
        {
            camera.orbit(delta.x, delta.y);
        }
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, theme::BG_0);
        let physical = size.map(|s| s.min(1600));
        if options.show_low
            && let (Some(rs), Some(renderer)) = (rs, &mut self.renderer)
        {
            if self.rendered != Some((*camera, size)) {
                let mut params = camera.view_params(size);
                params.height_scale = 0.0;
                params.environment.panels = 2.0;
                let target = renderer.render(&rs.device, &rs.queue, physical, &params);
                let mut egui_renderer = rs.renderer.write();
                self.texture = Some(match self.texture {
                    Some(id) => {
                        egui_renderer.update_egui_texture_from_wgpu_texture(
                            &rs.device,
                            &target.display_view,
                            wgpu::FilterMode::Linear,
                            id,
                        );
                        id
                    }
                    None => egui_renderer.register_native_texture(
                        &rs.device,
                        &target.display_view,
                        wgpu::FilterMode::Linear,
                    ),
                });
                self.rendered = Some((*camera, size));
            }
            if let Some(texture) = self.texture {
                painter.image(
                    texture,
                    rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }
        }
        if self.edge_slot != Some(options.slot) {
            let triangles = options.slot.map_or(0..mesh.triangle_count(), |slot| {
                mesh.material_triangles(slot)
            });
            let step = triangles.len().div_ceil(2500).max(1);
            let mut edges = BTreeSet::new();
            for t in triangles.step_by(step) {
                let [a, b, c] = mesh.tri(t);
                for [a, b] in [[a, b], [b, c], [c, a]] {
                    edges.insert([a.min(b), a.max(b)]);
                }
            }
            self.edges = edges.into_iter().collect();
            self.edge_slot = Some(options.slot);
            self.wire_key = None;
        }
        let key = WireKey {
            camera: *camera,
            size,
            pixels_per_point: ppp.to_bits(),
            options,
        };
        if self.wire_key != Some(key) {
            self.lines.clear();
            let paint_camera = camera.paint_camera(size);
            let alpha = (options.opacity.clamp(0.05, 1.0) * 255.0) as u8;
            for (enabled, distance, color) in [
                (
                    options.show_low,
                    0.0,
                    Color32::from_rgba_unmultiplied(156, 168, 185, alpha / 2),
                ),
                (
                    options.show_rear,
                    -options.rear,
                    Color32::from_rgba_unmultiplied(255, 157, 77, alpha),
                ),
                (
                    options.show_front,
                    options.front,
                    Color32::from_rgba_unmultiplied(56, 210, 235, alpha),
                ),
            ] {
                if !enabled {
                    continue;
                }
                for &[a, b] in &self.edges {
                    let project = |i: u32| {
                        let i = i as usize;
                        let point = cage_position(
                            mesh.positions[i],
                            mesh.normals[i],
                            mesh.bounds.diagonal(),
                            distance,
                        );
                        paint_camera
                            .project(point)
                            .map(|p| Pos2::new(p.x / ppp, p.y / ppp))
                    };
                    if let (Some(a), Some(b)) = (project(a), project(b)) {
                        self.lines.push(([a, b], Stroke::new(0.8, color)));
                    }
                }
            }
            self.wire_key = Some(key);
        }
        for (points, stroke) in &self.lines {
            painter.line_segment(
                [
                    points[0] + rect.min.to_vec2(),
                    points[1] + rect.min.to_vec2(),
                ],
                *stroke,
            );
        }
        painter.text(
            rect.left_top() + egui::vec2(10.0, 10.0),
            egui::Align2::LEFT_TOP,
            "Ciano: frontal   ·   Laranja: traseira   ·   Cinza: low poly",
            theme::font(11.5),
            theme::TEXT,
        );
        painter.text(
            rect.left_bottom() + egui::vec2(10.0, -10.0),
            egui::Align2::LEFT_BOTTOM,
            "Arrastar: orbitar  ·  Meio: mover  ·  Roda: zoom  ·  Cage em wireframe / raio X",
            theme::font(11.0),
            theme::TEXT_DIM,
        );
        if mesh.triangle_count() > 2500 {
            painter.text(
                rect.right_top() + egui::vec2(-10.0, 10.0),
                egui::Align2::RIGHT_TOP,
                "Wireframe simplificado; bake usa a malha completa",
                theme::font(10.5),
                theme::TEXT_DIM,
            );
        }
        if response.dragged() {
            ui.ctx().request_repaint();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cage_extrusion_uses_low_scale_and_both_sides() {
        let p = Vec3::new(1.0, 2.0, 3.0);
        let n = Vec3::new(0.0, 2.0, 0.0);
        assert!((cage_position(p, n, 10.0, 0.03) - (p + Vec3::Y * 0.3)).length() < 1e-5);
        assert!((cage_position(p, n, 10.0, -0.02) - (p - Vec3::Y * 0.2)).length() < 1e-5);
    }
}
