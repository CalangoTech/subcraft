//! Mesh-map baking (CPU, BVH ray casting, rayon).
//!
//! For every covered texel of a texture set the low-poly surface point is found by UV
//! rasterization. With a high-poly mesh, that point is projected onto the high poly along the
//! low-poly normal (cage: `front` above, `rear` below) and detail maps are taken there; without
//! one, the low poly is its own detail ("low on low": flat tangent normals).
//!
//! Maps: tangent-space normal (OpenGL / MikkTSpace frame of the low poly), world-space normal,
//! position (normalized per axis to the low-poly bounds), ID (color per mesh element), ambient
//! occlusion (cosine-weighted hemisphere rays), curvature (3D neighbourhood convexity, 0.5 flat)
//! and thickness (rays into the surface; bright = thin). Gutters are padded from the nearest
//! island texel.

pub mod curvature;
mod padding;
mod progress;
pub mod surface;

pub use padding::dilate_cancellable;
pub use progress::{BakePhase, BakePreview, Progress, ProgressSnapshot};

use std::collections::BTreeMap;

use glam::Vec3;
use rayon::prelude::*;
use subcraft_core::MeshMapKind;
use subcraft_core::color::hsv_to_rgb;
use subcraft_image::TiledImage;
use subcraft_mesh::{Bvh, Mesh, topology};

use surface::{Texel, cosine_directions, hash01, texels_cancellable};

#[derive(Clone, Debug, PartialEq)]
pub struct BakeSettings {
    /// Output size (square).
    pub size: u32,
    /// Samples per output-pixel axis: 1 = native, 2 = 4 samples, 4 = 16 samples.
    pub supersampling: u32,
    pub maps: Vec<MeshMapKind>,
    pub ao_samples: u32,
    /// Occlusion ray length as a fraction of the mesh bounding-box diagonal.
    pub ao_distance: f32,
    pub thickness_samples: u32,
    pub thickness_distance: f32,
    /// Curvature neighbourhood radius as a fraction of the diagonal.
    pub curvature_radius: f32,
    /// High-poly projection cage above / below the low poly, fractions of the diagonal.
    pub cage_front: f32,
    pub cage_rear: f32,
    /// 0 chooses half the available logical processors, capped at four workers.
    pub cpu_threads: u32,
    /// Conservative peak allocation budget, checked before rasterizing UVs.
    pub memory_limit_mb: u32,
    /// Ray-origin offset as a fraction of the geometry diagonal.
    pub ray_bias: f32,
    /// DirectX normal convention; otherwise OpenGL (+Y).
    pub normal_flip_y: bool,
    /// Changes the categorical mesh-element palette deterministically.
    pub id_seed: u32,
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            size: 1024,
            supersampling: 1,
            maps: MeshMapKind::ALL.to_vec(),
            ao_samples: 32,
            ao_distance: 0.25,
            thickness_samples: 16,
            thickness_distance: 0.5,
            curvature_radius: 0.02,
            cage_front: 0.01,
            cage_rear: 0.01,
            cpu_threads: 0,
            memory_limit_mb: 1536,
            ray_bias: 2e-4,
            normal_flip_y: false,
            id_seed: 0,
        }
    }
}

impl BakeSettings {
    /// Validate before allocating resolution-dependent buffers or launching ray work.
    pub fn validate(&self) -> Result<(), BakeError> {
        if self.size == 0 || self.size > 4096 {
            return Err(BakeError::InvalidSettings(
                "bake size must be between 1 and 4096",
            ));
        }
        if ![1, 2, 4].contains(&self.supersampling) {
            return Err(BakeError::InvalidSettings(
                "supersampling must be 1, 2 or 4 samples per axis",
            ));
        }
        if self.size * self.supersampling > 4096 {
            return Err(BakeError::InvalidSettings(
                "size × supersampling must not exceed 4096; lower the size or anti-aliasing quality",
            ));
        }
        if self.maps.is_empty() {
            return Err(BakeError::InvalidSettings("select at least one mesh map"));
        }
        if self.ao_samples == 0
            || self.ao_samples > 1024
            || self.thickness_samples == 0
            || self.thickness_samples > 1024
        {
            return Err(BakeError::InvalidSettings(
                "ray sample counts must be between 1 and 1024",
            ));
        }
        for value in [
            self.ao_distance,
            self.curvature_radius,
            self.cage_front,
            self.cage_rear,
            self.ray_bias,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(BakeError::InvalidSettings(
                    "ray distances, curvature radius and cage distances must be finite and non-negative",
                ));
            }
        }
        if !self.thickness_distance.is_finite() || self.thickness_distance <= 0.0 {
            return Err(BakeError::InvalidSettings(
                "thickness distance must be finite and greater than zero",
            ));
        }
        if self.cpu_threads > 32 {
            return Err(BakeError::InvalidSettings(
                "CPU thread limit must be between 0 (automatic) and 32",
            ));
        }
        if !(64..=8192).contains(&self.memory_limit_mb) {
            return Err(BakeError::InvalidSettings(
                "memory budget must be between 64 and 8192 MiB",
            ));
        }
        Ok(())
    }

    pub fn cpu_worker_count(&self) -> usize {
        let available = std::thread::available_parallelism().map_or(1, usize::from);
        if self.cpu_threads == 0 {
            (available / 2).clamp(1, 4)
        } else {
            (self.cpu_threads as usize).min(available).max(1)
        }
    }

    /// Includes UV samples, their projected detail, dense working buffers, gutter seeds and
    /// retained output maps. Mesh import/BVH memory is additional and does not scale with UV size.
    pub fn estimated_memory_mb(&self) -> usize {
        let pixels = self.size as u64 * self.size as u64;
        let samples = pixels * self.supersampling as u64 * self.supersampling as u64;
        let sample_bytes =
            samples * (std::mem::size_of::<Texel>() + std::mem::size_of::<Detail>() + 20) as u64;
        let output_bytes = pixels
            * self
                .maps
                .iter()
                .map(|k| k.components() as u64 * 2)
                .sum::<u64>();
        ((sample_bytes + pixels * 40 + output_bytes + 8 * 1024 * 1024).div_ceil(1024 * 1024))
            as usize
    }

    pub fn validate_memory_budget(&self) -> Result<(), BakeError> {
        let estimated_mb = self.estimated_memory_mb();
        if estimated_mb > self.memory_limit_mb as usize {
            return Err(BakeError::MemoryBudget {
                estimated_mb,
                limit_mb: self.memory_limit_mb as usize,
            });
        }
        Ok(())
    }
}

/// A mesh with its BVH.
#[derive(Clone, Copy)]
pub struct Geometry<'a> {
    pub mesh: &'a Mesh,
    pub bvh: &'a Bvh,
}

pub struct BakeInput<'a> {
    pub low: Geometry<'a>,
    pub high: Option<Geometry<'a>>,
    pub material_slot: u32,
}

/// Rendering front ends can supply a GPU baker without adding a GPU dependency to app/core.
pub trait BakeBackend: Send + Sync {
    fn name(&self) -> &str;
    fn bake(
        &self,
        input: &BakeInput<'_>,
        settings: &BakeSettings,
        progress: &Progress,
    ) -> Result<BTreeMap<MeshMapKind, TiledImage>, BakeError>;
}

#[derive(Default)]
pub struct CpuBakeBackend;

impl BakeBackend for CpuBakeBackend {
    fn name(&self) -> &str {
        "CPU"
    }
    fn bake(
        &self,
        input: &BakeInput<'_>,
        settings: &BakeSettings,
        progress: &Progress,
    ) -> Result<BTreeMap<MeshMapKind, TiledImage>, BakeError> {
        bake(input, settings, progress)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum BakeError {
    Cancelled,
    NoTexels,
    InvalidSettings(&'static str),
    Backend(String),
    MemoryBudget {
        estimated_mb: usize,
        limit_mb: usize,
    },
}

impl std::fmt::Display for BakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BakeError::Cancelled => f.write_str("bake cancelled"),
            BakeError::NoTexels => f.write_str("texture set has no UV-covered texels"),
            BakeError::InvalidSettings(reason) => write!(f, "invalid bake settings: {reason}"),
            BakeError::Backend(reason) => write!(f, "bake backend: {reason}"),
            BakeError::MemoryBudget {
                estimated_mb,
                limit_mb,
            } => write!(
                f,
                "bake needs approximately {estimated_mb} MiB; budget is {limit_mb} MiB. Lower resolution or anti-aliasing"
            ),
        }
    }
}

impl std::error::Error for BakeError {}

/// Where detail maps are sampled for one texel.
#[derive(Clone, Copy)]
struct Detail {
    position: Vec3,
    normal: Vec3,
    /// Triangle in the detail mesh.
    tri: u32,
    /// Offset direction for ray origins (geometric normal).
    offset: Vec3,
    /// Projection misses use the low-poly geometry, including its own triangle indices.
    high_poly: bool,
}

const CHUNK: usize = 2048;

/// Bake the requested maps for one texture set.
pub fn bake(
    input: &BakeInput,
    settings: &BakeSettings,
    progress: &Progress,
) -> Result<BTreeMap<MeshMapKind, TiledImage>, BakeError> {
    settings.validate()?;
    settings.validate_memory_budget()?;
    #[cfg(not(target_arch = "wasm32"))]
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(settings.cpu_worker_count())
            .thread_name(|i| format!("subcraft-bake-{i}"))
            .build()
            .map_err(|e| BakeError::Backend(e.to_string()))?;
        pool.install(|| bake_cpu(input, settings, progress))
    }
    #[cfg(target_arch = "wasm32")]
    bake_cpu(input, settings, progress)
}

fn bake_cpu(
    input: &BakeInput<'_>,
    settings: &BakeSettings,
    progress: &Progress,
) -> Result<BTreeMap<MeshMapKind, TiledImage>, BakeError> {
    if progress.is_cancelled() {
        return Err(BakeError::Cancelled);
    }
    settings.validate()?;
    if !progress.snapshot().backend.starts_with("CPU") {
        progress.set_backend("CPU");
    }
    progress.begin(0);
    let size = settings.size;
    let sample_size = size * settings.supersampling;
    let low = input.low.mesh;
    let samples = texels_cancellable(low, input.material_slot, sample_size, || {
        progress.is_cancelled()
    })
    .ok_or(BakeError::Cancelled)?;
    if samples.is_empty() {
        return Err(BakeError::NoTexels);
    }
    let detail_geo = input.high.unwrap_or(input.low);
    let diag = low
        .bounds
        .diagonal()
        .max(detail_geo.mesh.bounds.diagonal())
        .max(1e-6);
    let bias = diag * settings.ray_bias;
    let cage_diag = low.bounds.diagonal().max(1e-6);

    // Project onto the detail surface.
    let project = |t: &Texel| -> Detail {
        if let Some(high) = input.high {
            let front = settings.cage_front * cage_diag;
            let rear = settings.cage_rear * cage_diag;
            let n = t.normal.normalize_or(t.geometric_normal);
            if let Some(hit) = high
                .bvh
                .intersect(t.position + n * front, -n, 0.0, front + rear)
            {
                // Detail projection needs positions/normals only. Never read high-poly UVs
                // or tangents, even if a caller supplies geometry without those arrays.
                let [a, b, c] = high.mesh.tri(hit.tri as usize).map(|i| i as usize);
                let w = 1.0 - hit.u - hit.v;
                let position = high.mesh.positions[a] * w
                    + high.mesh.positions[b] * hit.u
                    + high.mesh.positions[c] * hit.v;
                let ng = high.mesh.geometric_normal(hit.tri as usize);
                let source_normal = (high.mesh.normals[a] * w
                    + high.mesh.normals[b] * hit.u
                    + high.mesh.normals[c] * hit.v)
                    .normalize_or(ng);
                let normal = if source_normal.dot(n) < 0.0 {
                    -source_normal
                } else {
                    source_normal
                };
                return Detail {
                    position,
                    normal,
                    tri: hit.tri,
                    offset: if ng.dot(n) < 0.0 { -ng } else { ng },
                    high_poly: true,
                };
            }
        }
        Detail {
            position: t.position,
            normal: t.normal,
            tri: t.tri,
            offset: t.geometric_normal,
            high_poly: false,
        }
    };

    let wanted = |k: MeshMapKind| settings.maps.contains(&k);
    let mut maps = settings.maps.clone();
    maps.sort();
    maps.dedup();
    let work_items = (maps.len() + 1) * samples.len() + maps.len();
    progress.set_total(work_items);
    progress.set_phase(BakePhase::Projecting);

    let mut details = Vec::with_capacity(samples.len());
    for chunk in samples.chunks(CHUNK) {
        if progress.is_cancelled() {
            return Err(BakeError::Cancelled);
        }
        details.par_extend(chunk.par_iter().map(project));
        progress.advance(chunk.len());
    }
    let low_elements = if wanted(MeshMapKind::Id) {
        Some(topology::mesh_elements(low))
    } else {
        None
    };
    let high_elements = if wanted(MeshMapKind::Id) {
        input.high.map(|high| topology::mesh_elements(high.mesh))
    } else {
        None
    };
    let bounds = low.bounds;
    let extent = bounds.extent().max(Vec3::splat(1e-6));

    let mut out = BTreeMap::new();
    for (map_index, &kind) in maps.iter().enumerate() {
        progress.set_map(kind, map_index + 1, maps.len());
        progress.set_map_work(samples.len() + 1);
        let comps = kind.components();
        // Per-texel evaluation, chunked for cancellation and progress.
        let eval = |i: usize, dirs: &mut Vec<Vec3>| -> [f32; 3] {
            let t = &samples[i];
            let d = &details[i];
            let geometry = if d.high_poly { detail_geo } else { input.low };
            match kind {
                MeshMapKind::Normal => {
                    let n = t.normal.normalize_or(t.geometric_normal);
                    let tan = (t.tangent.truncate() - n * n.dot(t.tangent.truncate()))
                        .normalize_or(n.any_orthonormal_vector());
                    let bit = n.cross(tan) * if t.tangent.w < 0.0 { -1.0 } else { 1.0 };
                    let h = d.normal;
                    let ts = Vec3::new(h.dot(tan), h.dot(bit), h.dot(n)).normalize_or(Vec3::Z);
                    [
                        ts.x * 0.5 + 0.5,
                        if settings.normal_flip_y {
                            0.5 - ts.y * 0.5
                        } else {
                            ts.y * 0.5 + 0.5
                        },
                        ts.z * 0.5 + 0.5,
                    ]
                }
                MeshMapKind::WorldSpaceNormal => {
                    let n = d.normal;
                    [n.x * 0.5 + 0.5, n.y * 0.5 + 0.5, n.z * 0.5 + 0.5]
                }
                MeshMapKind::Position => {
                    let p = ((t.position - bounds.min) / extent).clamp(Vec3::ZERO, Vec3::ONE);
                    [p.x, p.y, p.z]
                }
                MeshMapKind::Id => {
                    let elements = if d.high_poly {
                        high_elements.as_ref()
                    } else {
                        low_elements.as_ref()
                    };
                    let e = elements.map_or(0, |e| e[d.tri as usize]);
                    let hue = ((e.wrapping_add(settings.id_seed)) as f32 * 0.618_034).fract();
                    hsv_to_rgb([hue, 0.65, 0.9])
                }
                MeshMapKind::AmbientOcclusion => {
                    cosine_directions(d.normal, settings.ao_samples, hash01(t.index, 1), dirs);
                    let origin = d.position + d.offset * bias;
                    let dist = settings.ao_distance * diag;
                    let (mut occluded, mut counted) = (0u32, 0u32);
                    for dir in dirs.iter() {
                        if progress.is_cancelled() {
                            break;
                        }
                        if dir.dot(d.offset) <= 0.0 {
                            continue;
                        }
                        counted += 1;
                        if geometry.bvh.occluded(origin, *dir, 0.0, dist) {
                            occluded += 1;
                        }
                    }
                    let ao = if counted == 0 {
                        1.0
                    } else {
                        1.0 - occluded as f32 / counted as f32
                    };
                    [ao; 3]
                }
                MeshMapKind::Thickness => {
                    cosine_directions(
                        -d.normal,
                        settings.thickness_samples,
                        hash01(t.index, 2),
                        dirs,
                    );
                    let origin = d.position - d.offset * bias;
                    let max = settings.thickness_distance * diag;
                    let total: f32 = dirs
                        .iter()
                        .map(|dir| {
                            if progress.is_cancelled() {
                                return max;
                            }
                            geometry
                                .bvh
                                .intersect(origin, *dir, 0.0, max)
                                .map_or(max, |h| h.t)
                        })
                        .sum();
                    let avg = total / dirs.len() as f32 / max;
                    [1.0 - avg; 3]
                }
                MeshMapKind::Curvature => {
                    [curvature::curvature_at(
                        geometry.mesh,
                        geometry.bvh,
                        d.position,
                        d.normal,
                        settings.curvature_radius * diag,
                    ); 3]
                }
            }
        };
        let n = (size * size) as usize;
        let neutral = kind.neutral_value();
        let mut dense = vec![0.0; n * comps];
        for pixel in dense.chunks_exact_mut(comps) {
            pixel.copy_from_slice(&neutral[..comps]);
        }
        let mut covered = vec![false; n];
        let mut counts = vec![0u16; n];
        // Only one result chunk is retained. At 4096² this avoids a second ~192 MiB
        // full-resolution array, including when supersampling a smaller output.
        let mut values: Vec<[f32; 3]> = Vec::with_capacity(CHUNK);
        for start in (0..samples.len()).step_by(CHUNK) {
            if progress.is_cancelled() {
                return Err(BakeError::Cancelled);
            }
            let end = (start + CHUNK).min(samples.len());
            values.clear();
            values.par_extend(
                (start..end)
                    .into_par_iter()
                    .map_init(Vec::new, |dirs, i| eval(i, dirs)),
            );
            for (t, v) in samples[start..end].iter().zip(&values) {
                let x = t.index % sample_size / settings.supersampling;
                let y = t.index / sample_size / settings.supersampling;
                let i = (y * size + x) as usize;
                if counts[i] == 0 {
                    dense[i * comps..(i + 1) * comps].copy_from_slice(&v[..comps]);
                } else if kind != MeshMapKind::Id {
                    // ID maps stay categorical; averaging two IDs would create an invalid color.
                    for c in 0..comps {
                        dense[i * comps + c] += v[c];
                    }
                }
                counts[i] += 1;
                covered[i] = true;
            }
            progress.advance(end - start);
            if progress.should_publish_preview() {
                publish_dense_preview(progress, kind, size, &dense, Some(&counts), false);
            }
        }
        if settings.supersampling > 1 && kind != MeshMapKind::Id {
            for (pixel, &count) in dense.chunks_exact_mut(comps).zip(&counts) {
                if count == 0 {
                    continue;
                }
                for value in pixel.iter_mut() {
                    *value /= count as f32;
                }
                if matches!(kind, MeshMapKind::Normal | MeshMapKind::WorldSpaceNormal) {
                    let normal = (Vec3::new(pixel[0], pixel[1], pixel[2]) * 2.0 - Vec3::ONE)
                        .normalize_or(Vec3::Z)
                        * 0.5
                        + Vec3::splat(0.5);
                    pixel.copy_from_slice(&normal.to_array());
                }
            }
        }
        progress.set_phase(BakePhase::Padding);
        padding::dilate_cancellable(
            size as usize,
            size as usize,
            comps,
            &mut dense,
            &covered,
            progress,
        )?;
        if progress.is_cancelled() {
            return Err(BakeError::Cancelled);
        }
        let mut fill = [0.0f32; 4];
        fill[..comps].copy_from_slice(&neutral[..comps]);
        out.insert(
            kind,
            TiledImage::from_dense(size, size, comps, fill, &dense),
        );
        publish_dense_preview(progress, kind, size, &dense, None, true);
        progress.advance(1);
    }
    progress.set_phase(BakePhase::Completed);
    Ok(out)
}

/// Small genuine working-texture preview; uncovered pixels remain transparent until padded.
pub fn publish_dense_preview(
    progress: &Progress,
    kind: MeshMapKind,
    size: u32,
    dense: &[f32],
    counts: Option<&[u16]>,
    force: bool,
) {
    let side = size.min(256);
    let comps = kind.components();
    if side == 0 || dense.len() != size as usize * size as usize * comps {
        return;
    }
    let mut rgba = Vec::with_capacity(side as usize * side as usize * 4);
    for y in 0..side {
        for x in 0..side {
            let i = ((y * size / side) * size + x * size / side) as usize;
            let count = counts.map_or(1, |counts| counts[i]);
            let divisor = if kind == MeshMapKind::Id {
                1
            } else {
                count.max(1)
            } as f32;
            for c in 0..3 {
                let channel = if comps == 1 { 0 } else { c };
                rgba.push(
                    (dense[i * comps + channel] / divisor)
                        .clamp(0.0, 1.0)
                        .mul_add(255.0, 0.5) as u8,
                );
            }
            rgba.push(if count == 0 { 0 } else { 255 });
        }
    }
    progress.publish_preview(kind, side, side, rgba, force);
}

#[cfg(test)]
mod tests;
