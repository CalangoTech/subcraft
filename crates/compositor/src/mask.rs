//! Mask evaluation: base raster followed by the effect stack.
//!
//! Each visible effect produces a value `e` and coverage `ea` per texel and is composited onto
//! the running mask with its blend mode and opacity (`m = lerp(m, B(m, e), ea·opacity)`).

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

use glam::Vec3;
use subcraft_core::MeshMapKind;
use subcraft_doc::{Generator, GeneratorKind, Mask, MaskEffectKind, Projection, ProjectionMode};
use subcraft_image::Wrap;
use subcraft_image::{TILE_SIZE, TILE_TEXELS};
use subcraft_materials::sample::{eval_procedural, sample_projected};

use crate::TileContext;
use crate::blend::{BlendTarget, composite};
use crate::util::read_tile_rgba;

/// Shared prefix results for one immutable compositing batch. Drop the cache after that batch;
/// a fresh cache is required after changing masks, resources, mesh maps or geometry.
///
/// Entries reserve their complete tile payload before computing, so concurrent evaluation
/// cannot exceed the configured retention budget. When full, evaluation uses its local cache.
/// The bookkeeping mutex is never held while evaluating a generator, filter or dependency.
pub struct SharedMaskCache {
    entries: Mutex<HashMap<MaskStageKey, Arc<OnceLock<Arc<Vec<f32>>>>>>,
    max_entries: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct MaskStageKey {
    mask: usize,
    geometry: usize,
    resources: usize,
    maps: usize,
    tile: usize,
    end: usize,
}

impl Default for SharedMaskCache {
    fn default() -> Self {
        Self::with_byte_limit(64 * 1024 * 1024)
    }
}

impl SharedMaskCache {
    // Payload plus a conservative allowance for the Vec/Arc/OnceLock, key and hash-table slot.
    const ENTRY_BYTES: usize = TILE_TEXELS * std::mem::size_of::<f32>() + 256;

    pub fn with_byte_limit(bytes: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            max_entries: bytes / Self::ENTRY_BYTES,
        }
    }

    /// Reserved tile storage and conservative bookkeeping allowance (not evaluation scratch).
    pub fn retained_bytes(&self) -> usize {
        self.entries.lock().unwrap_or_else(|e| e.into_inner()).len() * Self::ENTRY_BYTES
    }

    fn entry(&self, key: MaskStageKey) -> Option<Arc<OnceLock<Arc<Vec<f32>>>>> {
        let mut entries = self.entries.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(entry) = entries.get(&key) {
            return Some(entry.clone());
        }
        if entries.len() >= self.max_entries {
            return None;
        }
        let entry = Arc::new(OnceLock::new());
        entries.insert(key, entry.clone());
        Some(entry)
    }
}

/// Mask values (0..1) for every texel of the tile, or `None` when the mask is absent/disabled.
pub fn eval_mask(mask: Option<&Mask>, tc: &TileContext) -> Option<Vec<f32>> {
    eval_mask_with_cache(mask, tc, None)
}

/// Evaluate a tile while sharing spatial prefix results with other tiles of the same immutable
/// batch. Standalone callers must create a new [`SharedMaskCache`] before any input is edited.
pub fn eval_mask_cached(
    mask: Option<&Mask>,
    tc: &TileContext,
    cache: &SharedMaskCache,
) -> Option<Vec<f32>> {
    eval_mask_with_cache(mask, tc, Some(cache))
}

fn eval_mask_with_cache(
    mask: Option<&Mask>,
    tc: &TileContext,
    shared: Option<&SharedMaskCache>,
) -> Option<Vec<f32>> {
    let mask = mask.filter(|m| m.enabled)?;
    // Preserve the inexpensive tile-local path for ordinary masks. Spatial filters evaluate
    // the actual preceding stack in neighbouring tiles; clamping to a tile would leave seams.
    if mask
        .effects
        .iter()
        .any(|e| e.visible && e.opacity > 0.0 && e.kind.spatial_radius() > 0)
    {
        let mut evaluator = MaskEvaluator {
            mask,
            tc,
            cache: HashMap::new(),
            shared,
        };
        // A final tile is used only once, unlike its halo dependencies: don't retain it in the
        // shared cache. This saves a whole resolution-sized image per spatially filtered mask.
        let values = evaluator.compute_stage(tc.geo.index, mask.effects.len());
        drop(evaluator);
        let mut m = Arc::try_unwrap(values).unwrap_or_else(|values| values.as_ref().clone());
        m.iter_mut().for_each(|v| *v = v.clamp(0.0, 1.0));
        return Some(m);
    }
    let mut px = vec![[0.0f32; 4]; TILE_TEXELS];
    read_tile_rgba(&mask.base, tc.geo, &mut px);
    let mut m: Vec<f32> = px.iter().map(|p| p[0]).collect();
    let mut e = vec![[0.0f32; 4]; TILE_TEXELS];
    for effect in mask.effects.iter().filter(|e| e.visible && e.opacity > 0.0) {
        eval_local_effect(&effect.kind, &m, tc, &mut e);
        blend_effect(&mut m, &e, effect);
    }
    for v in m.iter_mut() {
        *v = v.clamp(0.0, 1.0);
    }
    Some(m)
}

/// Pointwise effects share one kernel between the fast path and the halo evaluator.
fn eval_local_effect(kind: &MaskEffectKind, m: &[f32], tc: &TileContext, e: &mut [[f32; 4]]) {
    // e[i] = [value, alpha, _, _]
    match kind {
        MaskEffectKind::Paint { image } => {
            read_tile_rgba(image, tc.geo, e);
        }
        MaskEffectKind::Fill {
            value,
            procedural,
            projection,
        } => {
            for (i, o) in e.iter_mut().enumerate() {
                let v = procedural
                    .as_ref()
                    .map_or(*value, |p| eval_procedural(p, projection, tc.geo, i));
                *o = [v, 1.0, 0.0, 0.0];
            }
        }
        MaskEffectKind::Generator(g) => eval_generator(g, tc, e),
        MaskEffectKind::Levels(l) => {
            for (o, &v) in e.iter_mut().zip(m) {
                *o = [l.apply(v), 1.0, 0.0, 0.0];
            }
        }
        MaskEffectKind::Invert => {
            for (o, &v) in e.iter_mut().zip(m) {
                *o = [1.0 - v, 1.0, 0.0, 0.0];
            }
        }
        MaskEffectKind::Contrast { amount, pivot } => {
            let amount = if amount.is_finite() {
                amount.clamp(0.0, 4.0)
            } else {
                1.0
            };
            let pivot = if pivot.is_finite() {
                pivot.clamp(0.0, 1.0)
            } else {
                0.5
            };
            for (o, &v) in e.iter_mut().zip(m) {
                *o = [
                    ((v - pivot) * amount + pivot).clamp(0.0, 1.0),
                    1.0,
                    0.0,
                    0.0,
                ];
            }
        }
        MaskEffectKind::Blur { .. }
        | MaskEffectKind::Dilate { .. }
        | MaskEffectKind::Erode { .. } => {
            // Only radius zero reaches the pointwise evaluator.
            for (o, &v) in e.iter_mut().zip(m) {
                *o = [v, 1.0, 0.0, 0.0];
            }
        }
        MaskEffectKind::Image {
            resource,
            projection,
            invert,
        } => eval_image(*resource, projection, *invert, tc, e),
    }
}

fn blend_effect(m: &mut [f32], e: &[[f32; 4]], effect: &subcraft_doc::MaskEffect) {
    for (mv, ev) in m.iter_mut().zip(e) {
        let mut dst = [*mv, *mv, *mv, 1.0];
        composite(
            &mut dst,
            [ev[0]; 3],
            ev[1] * effect.opacity,
            effect.blend,
            BlendTarget::MASK,
        );
        *mv = dst[0];
    }
}

/// Only requested preceding stages are retained, per evaluation. This avoids both a dense
/// full-resolution mask and exponentially repeating the prefix when several filters stack.
struct MaskEvaluator<'a, 'b, 'c> {
    mask: &'a Mask,
    tc: &'a TileContext<'b>,
    cache: HashMap<(usize, usize), Arc<Vec<f32>>>,
    shared: Option<&'c SharedMaskCache>,
}

impl MaskEvaluator<'_, '_, '_> {
    fn stage(&mut self, tile: usize, end: usize) -> Arc<Vec<f32>> {
        if let Some(values) = self.cache.get(&(tile, end)) {
            return values.clone();
        }
        let key = MaskStageKey {
            mask: self.mask as *const Mask as usize,
            geometry: self.tc.set_geometry as *const _ as usize,
            resources: self.tc.sample.resources as *const _ as usize,
            maps: self.tc.sample.mesh_maps as *const _ as usize,
            tile,
            end,
        };
        let entry = self.shared.and_then(|cache| cache.entry(key));
        let values = if let Some(entry) = entry {
            // Every dependency has a strictly smaller `end`, so concurrent OnceLock waits
            // form a DAG and cannot create a cycle between neighbouring output tiles.
            entry.get_or_init(|| self.compute_stage(tile, end)).clone()
        } else {
            self.compute_stage(tile, end)
        };
        self.cache.insert((tile, end), values.clone());
        values
    }

    fn compute_stage(&mut self, tile: usize, end: usize) -> Arc<Vec<f32>> {
        let geo = self.tc.set_geometry.tile(tile);
        let tc = TileContext {
            geo: &geo,
            set_geometry: self.tc.set_geometry,
            sample: self.tc.sample,
        };
        // Evaluate a whole pointwise run at once, recursing only at spatial dependencies.
        let split = self.mask.effects[..end]
            .iter()
            .rposition(|e| e.visible && e.opacity > 0.0 && e.kind.spatial_radius() > 0);
        let (mut m, start) = if let Some(index) = split {
            let previous = self.stage(tile, index);
            let effect = self.mask.effects[index].clone();
            let filtered = self.spatial(tile, index, &effect.kind);
            let mut m = previous.as_ref().clone();
            for (value, filtered) in m.iter_mut().zip(filtered) {
                let mut dst = [*value, *value, *value, 1.0];
                composite(
                    &mut dst,
                    [filtered; 3],
                    effect.opacity,
                    effect.blend,
                    BlendTarget::MASK,
                );
                *value = dst[0];
            }
            (m, index + 1)
        } else {
            let mut pixels = vec![[0.0; 4]; TILE_TEXELS];
            read_tile_rgba(&self.mask.base, &geo, &mut pixels);
            (pixels.iter().map(|p| p[0]).collect(), 0)
        };
        let mut e = vec![[0.0; 4]; TILE_TEXELS];
        for effect in self.mask.effects[start..end]
            .iter()
            .filter(|e| e.visible && e.opacity > 0.0)
        {
            eval_local_effect(&effect.kind, &m, &tc, &mut e);
            blend_effect(&mut m, &e, effect);
        }
        Arc::new(m)
    }

    fn spatial(&mut self, tile: usize, prefix: usize, kind: &MaskEffectKind) -> Vec<f32> {
        let r = kind.spatial_radius() as usize;
        let side = TILE_SIZE as usize + 2 * r;
        let geo = self.tc.set_geometry.tile(tile);
        let limit = self.tc.set_geometry.resolution as i64 - 1;
        let x0 = geo.x0 as i64 - r as i64;
        let y0 = geo.y0 as i64 - r as i64;
        let tx0 = x0.clamp(0, limit) as u32 / TILE_SIZE;
        let ty0 = y0.clamp(0, limit) as u32 / TILE_SIZE;
        let tx1 = (x0 + side as i64 - 1).clamp(0, limit) as u32 / TILE_SIZE;
        let ty1 = (y0 + side as i64 - 1).clamp(0, limit) as u32 / TILE_SIZE;
        let tiles_x = self.tc.set_geometry.tiles_x;
        let span = (tx1 - tx0 + 1) as usize;
        let mut inputs = Vec::new();
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                inputs.push(self.stage((ty * tiles_x + tx) as usize, prefix));
            }
        }
        let mut source = vec![0.0; side * side];
        for y in 0..side {
            let sy = (y0 + y as i64).clamp(0, limit) as u32;
            for x in 0..side {
                let sx = (x0 + x as i64).clamp(0, limit) as u32;
                let tile =
                    ((sy / TILE_SIZE - ty0) as usize * span) + (sx / TILE_SIZE - tx0) as usize;
                let pixel = (sy % TILE_SIZE * TILE_SIZE + sx % TILE_SIZE) as usize;
                source[y * side + x] = inputs[tile][pixel];
            }
        }
        filter_square(&source, side, r, kind)
    }
}

/// Separable square filters. Rolling sums and monotonic queues make cost linear in the halo
/// size instead of quadratic in the radius. Texture borders use edge replication.
fn filter_square(input: &[f32], side: usize, radius: usize, kind: &MaskEffectKind) -> Vec<f32> {
    let width = side - 2 * radius;
    let mut horizontal = vec![0.0; side * width];
    for y in 0..side {
        filter_line(
            &input[y * side..(y + 1) * side],
            radius,
            kind,
            &mut horizontal[y * width..(y + 1) * width],
        );
    }
    let mut output = vec![0.0; width * width];
    let mut column = vec![0.0; side];
    let mut filtered = vec![0.0; width];
    for x in 0..width {
        for y in 0..side {
            column[y] = horizontal[y * width + x];
        }
        filter_line(&column, radius, kind, &mut filtered);
        for y in 0..width {
            output[y * width + x] = filtered[y];
        }
    }
    output
}

fn filter_line(input: &[f32], radius: usize, kind: &MaskEffectKind, output: &mut [f32]) {
    let window = 2 * radius + 1;
    if matches!(kind, MaskEffectKind::Blur { .. }) {
        let mut sum: f64 = input[..window].iter().map(|&v| v as f64).sum();
        output[0] = (sum / window as f64) as f32;
        for i in 1..output.len() {
            sum += input[i + window - 1] as f64 - input[i - 1] as f64;
            output[i] = (sum / window as f64) as f32;
        }
    } else {
        let maximum = matches!(kind, MaskEffectKind::Dilate { .. });
        let mut deque: VecDeque<usize> = VecDeque::with_capacity(window);
        for (i, &value) in input.iter().enumerate() {
            while deque.front().is_some_and(|&j| j + window <= i) {
                deque.pop_front();
            }
            while deque.back().is_some_and(|&j| {
                if maximum {
                    input[j] <= value
                } else {
                    input[j] >= value
                }
            }) {
                deque.pop_back();
            }
            deque.push_back(i);
            if i + 1 >= window {
                output[i + 1 - window] = input[*deque.front().unwrap()];
            }
        }
    }
}

#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Generator coverage: a mesh-derived raw value thresholded at `level` (optionally broken up by
/// noise) with a `softness`-wide transition.
fn eval_generator(g: &Generator, tc: &TileContext, out: &mut [[f32; 4]]) {
    let geo = tc.geo;
    let mut map = vec![[0.0f32; 3]; TILE_TEXELS];
    let map_kind = match g.kind {
        GeneratorKind::Edges => Some(MeshMapKind::Curvature),
        GeneratorKind::Cavity => Some(MeshMapKind::AmbientOcclusion),
        GeneratorKind::MeshMap(k) => Some(k),
        GeneratorKind::Directional | GeneratorKind::PositionGradient => None,
    };
    if let Some(k) = map_kind {
        tc.sample.mesh_maps.read_tile(k, geo, &mut map);
    }
    let dir = Vec3::from_array(g.direction).normalize_or(Vec3::Y);
    let set = tc.set_geometry;
    let ext_y = set.bounds.extent().y.max(1e-12);
    let half = g.softness.max(1e-4) * 0.5;
    // Breakup noise is projected in 3D (triplanar on object position), never in UV: UV noise
    // jumps at every seam and would make wear follow the islands instead of the shape.
    let noise_proj = Projection {
        mode: ProjectionMode::Triplanar,
        triplanar_sharpness: 4.0,
        ..Projection::default()
    };
    for (i, o) in out.iter_mut().enumerate() {
        let raw = match g.kind {
            GeneratorKind::Edges | GeneratorKind::MeshMap(_) => map[i][0],
            GeneratorKind::Cavity => 1.0 - map[i][0],
            GeneratorKind::Directional => Vec3::from_array(geo.normal(i)).dot(dir) * 0.5 + 0.5,
            GeneratorKind::PositionGradient => {
                (geo.position(i)[1] * set.max_extent / ext_y).clamp(0.0, 1.0)
            }
        };
        let level = if g.noise_amount > 0.0 {
            g.level + (eval_procedural(&g.noise, &noise_proj, geo, i) - 0.5) * g.noise_amount
        } else {
            g.level
        };
        let v = smoothstep(level - half, level + half, raw);
        *o = [if g.invert { 1.0 - v } else { v }, 1.0, 0.0, 0.0];
    }
}

/// Image mask effect. With alpha: value 1 (0 inverted) with coverage = alpha, so a logo on a
/// transparent background stamps onto the mask. Without alpha: value = Rec.709 luminance of the
/// raw (not color-transformed) pixels. A missing resource contributes nothing.
fn eval_image(
    resource: subcraft_core::ResourceId,
    projection: &Projection,
    invert: bool,
    tc: &TileContext,
    e: &mut [[f32; 4]],
) {
    let Some(bmp) = tc.sample.resources.bitmap(resource) else {
        e.fill([0.0; 4]);
        return;
    };
    let has_alpha = bmp.channels == 2 || bmp.channels == 4;
    let geo = tc.geo;
    for (i, o) in e.iter_mut().enumerate() {
        let s = sample_projected(projection, geo.uv(i), geo.position(i), geo.normal(i), |p| {
            bmp.sample(p[0], p[1], Wrap::Repeat, false)
        });
        let (v, a) = if has_alpha {
            (1.0, s[3])
        } else {
            (0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2], 1.0)
        };
        *o = [
            if invert { 1.0 - v } else { v },
            a.clamp(0.0, 1.0),
            0.0,
            0.0,
        ];
    }
}
