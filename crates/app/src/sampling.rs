//! Sample material values from the composite, before lighting or display transforms.

use glam::Vec2;
use subcraft_core::{ChannelKind, ChannelValue, TextureSetId};

use crate::{Camera, Session};

impl Session {
    pub fn sample_uv(
        &mut self,
        texture_set: TextureSetId,
        uv: Vec2,
        channel: ChannelKind,
    ) -> Option<ChannelValue> {
        if !uv.is_finite() {
            return None;
        }
        self.update_composites();
        let ts = self.document().texture_set(texture_set)?;
        let image = self.composite(texture_set)?.channel(channel)?;
        // Repeat exactly like the viewport's material sampler, including negative UVs.
        let p = uv.rem_euclid(Vec2::ONE) * ts.resolution as f32;
        let value = image.get(
            (p.x as u32).min(image.width() - 1),
            (p.y as u32).min(image.height() - 1),
        );
        Some(if channel.components() == 3 {
            ChannelValue::rgb(value[0], value[1], value[2])
        } else {
            ChannelValue::scalar(value[0])
        })
    }

    /// Samples the texture set actually hit, even if another set is selected.
    pub fn sample_surface(
        &mut self,
        camera: &Camera,
        screen: Vec2,
        channel: ChannelKind,
    ) -> Option<ChannelValue> {
        let scene = self.scene();
        let (origin, direction) = camera.ray(screen);
        let hit = scene.bvh.intersect(origin, direction, 0.0, f32::INFINITY)?;
        let uv = scene.mesh.interpolate(hit.tri as usize, hit.u, hit.v).uv;
        let slot = scene.mesh.tri_material[hit.tri as usize];
        let texture_set = self.document().texture_set_for_slot(slot)?.id;
        self.sample_uv(texture_set, uv, channel)
    }
}
