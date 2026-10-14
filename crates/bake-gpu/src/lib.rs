//! Hardware compute mesh baker. UV rasterization and padding stay on the CPU; projection,
//! BVH ray queries and all seven map evaluations run on the GPU without CUDA/RTX requirements.
//! A separate device, tiny dispatches, bounded traversal, allocation guards and finite waits
//! keep this worker independent of the viewport and allow cancellation between submissions.

use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use bytemuck::{Pod, Zeroable};
use egui_wgpu::wgpu::{self, util::DeviceExt};
use glam::Vec3;
use subcraft_bake::surface::{Texel, hash01, texels_cancellable};
use subcraft_bake::{
    BakeBackend, BakeError, BakeInput, BakePhase, BakeSettings, Progress, publish_dense_preview,
};
use subcraft_core::MeshMapKind;
use subcraft_image::TiledImage;
use subcraft_mesh::{Bvh, Mesh, topology};

const CHUNK: usize = 8192;
const TEXELS_PER_DISPATCH: usize = 128;
const RAYS_PER_DISPATCH: u32 = 4;
const DISPATCHES_PER_SUBMISSION: usize = 32;
const GPU_BUDGET_BYTES: u64 = 512 * 1024 * 1024;
const SUBMISSION_TIMEOUT: Duration = Duration::from_secs(2);

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuSample {
    position: [f32; 4],
    normal: [f32; 4],
    tangent: [f32; 4],
    geometric: [f32; 4],
    meta: [u32; 4],
}
impl From<&Texel> for GpuSample {
    fn from(t: &Texel) -> Self {
        Self {
            position: t.position.extend(hash01(t.index, 1)).to_array(),
            normal: t.normal.extend(hash01(t.index, 2)).to_array(),
            tangent: t.tangent.to_array(),
            geometric: t.geometric_normal.extend(0.0).to_array(),
            meta: [t.index, t.tri, 0, 0],
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuTriangle {
    a: [f32; 4],
    ab: [f32; 4],
    ac: [f32; 4],
    na: [f32; 4],
    nb: [f32; 4],
    nc: [f32; 4],
    meta: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuNode {
    minimum: [f32; 4],
    maximum: [f32; 4],
    meta: [u32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    counts: [u32; 4],
    roots: [u32; 4],
    options: [u32; 4],
    distances: [f32; 4],
    cage: [f32; 4],
    bounds_min: [f32; 4],
    bounds_extent: [f32; 4],
    flags: [u32; 4],
}

struct BatchBuffers {
    samples: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    result: wgpu::Buffer,
    staging: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    stride: usize,
}

/// Construct on a background worker; native adapter/device initialization may take time.
pub struct GpuBaker {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    name: String,
    error: Arc<Mutex<Option<String>>>,
    serial: Mutex<()>,
}

pub type GpuBakeBackend = GpuBaker;

impl GpuBaker {
    pub fn new() -> Result<Self, String> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        // DX12 avoids creating a second Vulkan device on the Windows viewport driver.
        #[cfg(target_os = "windows")]
        {
            desc.backends = wgpu::Backends::DX12;
        }
        let instance = wgpu::Instance::new(desc);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            ..Default::default()
        }))
        .map_err(|e| format!("GPU indisponível: {e}"))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu {
            return Err("O adaptador disponível usa software; selecione CPU".into());
        }
        let mut limits = wgpu::Limits::default();
        limits.max_storage_buffer_binding_size = adapter
            .limits()
            .max_storage_buffer_binding_size
            .min(256 * 1024 * 1024);
        limits.max_buffer_size = adapter.limits().max_buffer_size.min(256 * 1024 * 1024);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("SubCraft bounded bake worker"),
            required_limits: limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            ..Default::default()
        }))
        .map_err(|e| format!("Não foi possível iniciar GPU: {e}"))?;
        let error = Arc::new(Mutex::new(None));
        let errors = error.clone();
        device.on_uncaptured_error(Arc::new(move |e| {
            if let Ok(mut slot) = errors.lock() {
                *slot = Some(e.to_string());
            }
        }));
        let lost_errors = error.clone();
        device.set_device_lost_callback(move |reason, message| {
            if let Ok(mut slot) = lost_errors.lock() {
                *slot = Some(format!("GPU device lost ({reason:?}): {message}"));
            }
        });
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bounded bake shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("bake.wgsl").into()),
        });
        let entries: Vec<_> = (0..8)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 0 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding <= 4,
                        }
                    },
                    has_dynamic_offset: binding == 0,
                    min_binding_size: if binding == 0 {
                        NonZeroU64::new(128)
                    } else {
                        None
                    },
                },
                count: None,
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bounded bake buffers"),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bounded bake layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("bounded mesh-map bake"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        if let Some(e) = pollster::block_on(validation.pop()) {
            return Err(format!("GPU shader incompatível: {e}"));
        }
        Ok(Self {
            device,
            queue,
            pipeline,
            layout,
            name: format!("GPU · {} ({:?})", info.name, info.backend),
            error,
            serial: Mutex::new(()),
        })
    }

    fn check(&self, progress: &Progress) -> Result<(), BakeError> {
        if progress.is_cancelled() {
            return Err(BakeError::Cancelled);
        }
        if let Ok(slot) = self.error.lock()
            && let Some(error) = slot.as_ref()
        {
            return Err(BakeError::Backend(error.clone()));
        }
        Ok(())
    }

    fn wait(&self, index: wgpu::SubmissionIndex, progress: &Progress) -> Result<(), BakeError> {
        let started = Instant::now();
        loop {
            self.check(progress)?;
            match self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(index.clone()),
                timeout: Some(Duration::from_millis(10)),
            }) {
                Ok(_) => return self.check(progress),
                Err(wgpu::PollError::Timeout) if started.elapsed() < SUBMISSION_TIMEOUT => {}
                Err(error) => {
                    self.device.destroy();
                    let message = format!(
                        "GPU parou por timeout/erro de driver ({error}). Reduza amostras/resolução ou use CPU."
                    );
                    if let Ok(mut slot) = self.error.lock() {
                        *slot = Some(message.clone());
                    }
                    return Err(BakeError::Backend(message));
                }
            }
        }
    }

    fn buffer(
        &self,
        label: &str,
        bytes: &[u8],
        usage: wgpu::BufferUsages,
    ) -> Result<wgpu::Buffer, BakeError> {
        let size = bytes.len().max(16) as u64;
        let limit = u64::from(self.device.limits().max_storage_buffer_binding_size)
            .min(self.device.limits().max_buffer_size);
        if size > limit {
            return Err(BakeError::Backend(format!(
                "{label}: {size} bytes excedem o limite GPU {limit}; use CPU ou reduza a malha"
            )));
        }
        Ok(self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: if bytes.is_empty() { &[0; 16] } else { bytes },
                usage,
            }))
    }

    fn batch_buffers(
        &self,
        geometry: &[wgpu::Buffer; 3],
        detail_count: usize,
        max_rays: u32,
    ) -> Result<BatchBuffers, BakeError> {
        let alloc = |label, size, usage| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let stride = self
            .device
            .limits()
            .min_uniform_buffer_offset_alignment
            .max(128) as usize;
        let command_count = CHUNK.div_ceil(TEXELS_PER_DISPATCH)
            * (2 + max_rays.div_ceil(RAYS_PER_DISPATCH) as usize);
        let uniforms = alloc(
            "reused bake uniforms",
            (command_count * stride) as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let samples = alloc(
            "reused bake samples",
            (CHUNK * 80) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let details = alloc(
            "bake projected detail cache",
            (detail_count * 64) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let accum = alloc(
            "reused bake accumulation",
            (CHUNK * 64) as u64,
            wgpu::BufferUsages::STORAGE,
        );
        let result = alloc(
            "reused bake result",
            (CHUNK * 16) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let staging = alloc(
            "reused bake readback",
            (CHUNK * 16) as u64,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );
        let buffers = [
            &uniforms,
            &samples,
            &geometry[0],
            &geometry[1],
            &geometry[2],
            &details,
            &accum,
            &result,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: if binding == 0 {
                    wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer,
                        offset: 0,
                        size: NonZeroU64::new(128),
                    })
                } else {
                    buffer.as_entire_binding()
                },
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reused bounded bake buffers"),
            layout: &self.layout,
            entries: &entries,
        });
        Ok(BatchBuffers {
            samples,
            uniforms,
            result,
            staging,
            bind_group,
            stride,
        })
    }

    fn run_chunk(
        &self,
        samples: &[Texel],
        base: Params,
        batch: &BatchBuffers,
        progress: &Progress,
        project: bool,
        projection_only: bool,
    ) -> Result<Vec<[f32; 4]>, BakeError> {
        self.check(progress)?;
        let gpu_samples: Vec<_> = samples.iter().map(GpuSample::from).collect();
        self.queue
            .write_buffer(&batch.samples, 0, bytemuck::cast_slice(&gpu_samples));
        // Each dynamic uniform slot describes one independent short dispatch. They may be
        // submitted together, but no individual dispatch scales with resolution or ray count.
        let stride = batch.stride;
        let mut commands = Vec::new();
        let mut append_stage = |stage: u32, ray: u32| {
            for start in (0..samples.len()).step_by(TEXELS_PER_DISPATCH) {
                let mut params = base;
                params.counts[0] = samples.len() as u32;
                params.counts[2] = ray;
                params.counts[3] = RAYS_PER_DISPATCH;
                params.flags[0] = stage;
                params.flags[2] = start as u32;
                commands.push(params);
            }
        };
        append_stage(if project { 0 } else { 3 }, 0);
        let rays = match base.counts[1] {
            4 => base.options[1],
            6 => base.options[2],
            _ => 0,
        };
        if !projection_only {
            for ray in (0..rays).step_by(RAYS_PER_DISPATCH as usize) {
                append_stage(2, ray);
            }
            append_stage(1, 0);
        }
        let mut uniform_data = vec![0u8; stride * commands.len()];
        for (index, command) in commands.iter().enumerate() {
            uniform_data[index * stride..index * stride + 128]
                .copy_from_slice(bytemuck::bytes_of(command));
        }
        self.queue.write_buffer(&batch.uniforms, 0, &uniform_data);
        for start in (0..commands.len()).step_by(DISPATCHES_PER_SUBMISSION) {
            self.check(progress)?;
            let end = (start + DISPATCHES_PER_SUBMISSION).min(commands.len());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("short bake submission"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bounded bake dispatches"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                for command in start..end {
                    pass.set_bind_group(0, &batch.bind_group, &[(command * stride) as u32]);
                    pass.dispatch_workgroups(
                        if commands[command].flags[0] == 2 {
                            8
                        } else {
                            2
                        },
                        1,
                        1,
                    );
                }
            }
            if end == commands.len() && !projection_only {
                encoder.copy_buffer_to_buffer(
                    &batch.result,
                    0,
                    &batch.staging,
                    0,
                    samples.len() as u64 * 16,
                );
            }
            let submission = self.queue.submit([encoder.finish()]);
            self.wait(submission, progress)?;
        }
        if projection_only {
            return Ok(Vec::new());
        }
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let readback_slice = batch.staging.slice(..samples.len() as u64 * 16);
        readback_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = sender.send(r);
        });
        let readback_started = Instant::now();
        loop {
            self.check(progress)?;
            self.device
                .poll(wgpu::PollType::Poll)
                .map_err(|e| BakeError::Backend(format!("GPU readback: {e}")))?;
            match receiver.try_recv() {
                Ok(mapped) => {
                    mapped.map_err(|e| BakeError::Backend(format!("GPU readback: {e}")))?;
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(BakeError::Backend(
                        "GPU readback callback desconectado".into(),
                    ));
                }
                Err(std::sync::mpsc::TryRecvError::Empty)
                    if readback_started.elapsed() < SUBMISSION_TIMEOUT =>
                {
                    std::thread::sleep(Duration::from_millis(2));
                }
                Err(_) => {
                    self.device.destroy();
                    return Err(BakeError::Backend(
                        "GPU readback excedeu o prazo seguro; bake interrompido".into(),
                    ));
                }
            }
        }
        let mapped = readback_slice
            .get_mapped_range()
            .map_err(|e| BakeError::Backend(format!("GPU readback range: {e}")))?;
        let values = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped).to_vec();
        drop(mapped);
        batch.staging.unmap();
        if values.iter().any(|v| v[3] > 0.0) {
            return Err(BakeError::Backend("Consulta BVH excedeu o limite seguro da GPU. Diminua a distância/raio da textura ou use CPU para esta malha.".into()));
        }
        if values.iter().flat_map(|v| &v[..3]).any(|v| !v.is_finite()) {
            return Err(BakeError::Backend(
                "GPU retornou valores inválidos; bake interrompido".into(),
            ));
        }
        Ok(values)
    }
}

fn append_geometry(
    mesh: &Mesh,
    bvh: &Bvh,
    with_ids: bool,
    triangles: &mut Vec<GpuTriangle>,
    nodes: &mut Vec<GpuNode>,
    tri_ids: &mut Vec<u32>,
) -> (u32, u32) {
    let tri_start = triangles.len() as u32;
    let node_start = nodes.len() as u32;
    let order_start = tri_ids.len() as u32;
    let elements = with_ids.then(|| topology::mesh_elements(mesh));
    for tri in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.tri_positions(tri);
        let [ia, ib, ic] = mesh.tri(tri).map(|i| i as usize);
        triangles.push(GpuTriangle {
            a: a.extend(0.0).to_array(),
            ab: (b - a).extend(0.0).to_array(),
            ac: (c - a).extend(0.0).to_array(),
            na: mesh.normals[ia].extend(0.0).to_array(),
            nb: mesh.normals[ib].extend(0.0).to_array(),
            nc: mesh.normals[ic].extend(0.0).to_array(),
            meta: [elements.as_ref().map_or(0, |e| e[tri]), 0, 0, 0],
        });
    }
    for n in bvh.traversal_nodes() {
        nodes.push(GpuNode {
            minimum: n.min.extend(0.0).to_array(),
            maximum: n.max.extend(0.0).to_array(),
            meta: [n.first + order_start, n.count, n.escape + node_start, 0],
        });
    }
    tri_ids.extend(bvh.triangle_order().iter().map(|t| t + tri_start));
    (node_start, nodes.len() as u32)
}

fn kind_id(kind: MeshMapKind) -> u32 {
    match kind {
        MeshMapKind::Normal => 0,
        MeshMapKind::WorldSpaceNormal => 1,
        MeshMapKind::Position => 2,
        MeshMapKind::Id => 3,
        MeshMapKind::AmbientOcclusion => 4,
        MeshMapKind::Curvature => 5,
        MeshMapKind::Thickness => 6,
    }
}

impl BakeBackend for GpuBaker {
    fn name(&self) -> &str {
        &self.name
    }

    fn bake(
        &self,
        input: &BakeInput<'_>,
        settings: &BakeSettings,
        progress: &Progress,
    ) -> Result<BTreeMap<MeshMapKind, TiledImage>, BakeError> {
        let _serial = self
            .serial
            .lock()
            .map_err(|_| BakeError::Backend("GPU worker indisponível".into()))?;
        self.check(progress)?;
        settings.validate()?;
        settings.validate_memory_budget()?;
        progress.begin(0);
        progress.set_backend(&self.name);
        let triangles_count = input.low.mesh.triangle_count() as u64
            + input.high.map_or(0, |g| g.mesh.triangle_count()) as u64;
        // Node count is <=2N; serialization is temporary and is dropped after upload.
        let gpu_estimate = triangles_count * (112 + 2 * 48 + 4) + 8 * 1024 * 1024;
        let cpu_estimate_mb =
            settings.estimated_memory_mb() + (gpu_estimate * 2).div_ceil(1024 * 1024) as usize;
        if cpu_estimate_mb > settings.memory_limit_mb as usize {
            return Err(BakeError::MemoryBudget {
                estimated_mb: cpu_estimate_mb,
                limit_mb: settings.memory_limit_mb as usize,
            });
        }
        if gpu_estimate > GPU_BUDGET_BYTES {
            return Err(BakeError::Backend(
                "Malha excede o orçamento conservador de 512 MiB da GPU; use CPU ou reduza a malha"
                    .into(),
            ));
        }
        let sample_size = settings.size * settings.supersampling;
        let samples = texels_cancellable(input.low.mesh, input.material_slot, sample_size, || {
            progress.is_cancelled()
        })
        .ok_or(BakeError::Cancelled)?;
        if samples.is_empty() {
            return Err(BakeError::NoTexels);
        }
        self.check(progress)?;
        let mut maps = settings.maps.clone();
        maps.sort();
        maps.dedup();
        let cache_bytes = samples.len() as u64 * 64;
        let cache_details = input.high.is_some()
            && maps.len() > 1
            && gpu_estimate + cache_bytes <= GPU_BUDGET_BYTES
            && cache_bytes <= u64::from(self.device.limits().max_storage_buffer_binding_size)
            && cpu_estimate_mb + cache_bytes.div_ceil(1024 * 1024) as usize
                <= settings.memory_limit_mb as usize;
        progress.set_total(
            (samples.len() + 1) * maps.len() + if cache_details { samples.len() } else { 0 },
        );
        progress.set_phase(BakePhase::Projecting);
        let mut triangles = Vec::new();
        let mut nodes = Vec::new();
        let mut tri_ids = Vec::new();
        let with_ids = maps.contains(&MeshMapKind::Id);
        let low = append_geometry(
            input.low.mesh,
            input.low.bvh,
            with_ids,
            &mut triangles,
            &mut nodes,
            &mut tri_ids,
        );
        let high = if let Some(g) = input.high {
            append_geometry(
                g.mesh,
                g.bvh,
                with_ids,
                &mut triangles,
                &mut nodes,
                &mut tri_ids,
            )
        } else {
            low
        };
        let geometry = [
            self.buffer(
                "bake triangles",
                bytemuck::cast_slice(&triangles),
                wgpu::BufferUsages::STORAGE,
            )?,
            self.buffer(
                "bake BVH",
                bytemuck::cast_slice(&nodes),
                wgpu::BufferUsages::STORAGE,
            )?,
            self.buffer(
                "bake triangle references",
                bytemuck::cast_slice(&tri_ids),
                wgpu::BufferUsages::STORAGE,
            )?,
        ];
        drop(triangles);
        drop(nodes);
        drop(tri_ids);
        let low_diagonal = input.low.mesh.bounds.diagonal().max(1e-6);
        let diagonal = low_diagonal.max(input.high.map_or(0.0, |g| g.mesh.bounds.diagonal()));
        let bounds = input.low.mesh.bounds;
        let params = Params {
            counts: [0; 4],
            roots: [low.0, low.1, high.0, high.1],
            options: [
                u32::from(input.high.is_some()),
                settings.ao_samples,
                settings.thickness_samples,
                u32::from(settings.normal_flip_y),
            ],
            distances: [
                diagonal,
                settings.ray_bias * diagonal,
                settings.ao_distance * diagonal,
                settings.thickness_distance * diagonal,
            ],
            cage: [
                settings.cage_front * low_diagonal,
                settings.cage_rear * low_diagonal,
                settings.curvature_radius * diagonal,
                0.0,
            ],
            bounds_min: bounds.min.extend(0.0).to_array(),
            bounds_extent: bounds
                .extent()
                .max(Vec3::splat(1e-6))
                .extend(0.0)
                .to_array(),
            flags: [0, settings.id_seed, 0, 0],
        };
        let max_rays = settings.ao_samples.max(settings.thickness_samples);
        let batch = self.batch_buffers(
            &geometry,
            if cache_details { samples.len() } else { CHUNK },
            max_rays,
        )?;
        if cache_details {
            // High-poly projection is identical for every map. Retain only that 64-byte detail
            // per sample when both the user budget and conservative GPU budget allow it.
            for (chunk_index, chunk) in samples.chunks(CHUNK).enumerate() {
                let mut cached_params = params;
                cached_params.flags[3] = (chunk_index * CHUNK) as u32;
                self.run_chunk(chunk, cached_params, &batch, progress, true, true)?;
                progress.advance(chunk.len());
            }
        }
        let mut output = BTreeMap::new();
        let padding_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(settings.cpu_worker_count())
            .build()
            .map_err(|e| BakeError::Backend(format!("CPU padding worker: {e}")))?;
        for (map_index, &kind) in maps.iter().enumerate() {
            self.check(progress)?;
            progress.set_map(kind, map_index + 1, maps.len());
            progress.set_map_work(samples.len() + 1);
            let components = kind.components();
            let neutral = kind.neutral_value();
            let pixels = (settings.size * settings.size) as usize;
            let mut dense = vec![0.0; pixels * components];
            for pixel in dense.chunks_exact_mut(components) {
                pixel.copy_from_slice(&neutral[..components]);
            }
            let mut counts = vec![0u16; pixels];
            let mut covered = vec![false; pixels];
            let mut map_params = params;
            map_params.counts[1] = kind_id(kind);
            for (chunk_index, chunk) in samples.chunks(CHUNK).enumerate() {
                map_params.flags[3] = if cache_details {
                    (chunk_index * CHUNK) as u32
                } else {
                    0
                };
                let values =
                    self.run_chunk(chunk, map_params, &batch, progress, !cache_details, false)?;
                for (sample, value) in chunk.iter().zip(&values) {
                    let x = sample.index % sample_size / settings.supersampling;
                    let y = sample.index / sample_size / settings.supersampling;
                    let index = (y * settings.size + x) as usize;
                    if counts[index] == 0 {
                        dense[index * components..(index + 1) * components]
                            .copy_from_slice(&value[..components]);
                    } else if kind != MeshMapKind::Id {
                        for c in 0..components {
                            dense[index * components + c] += value[c];
                        }
                    }
                    counts[index] += 1;
                    covered[index] = true;
                }
                progress.advance(chunk.len());
                if progress.should_publish_preview() {
                    publish_dense_preview(
                        progress,
                        kind,
                        settings.size,
                        &dense,
                        Some(&counts),
                        false,
                    );
                }
            }
            if settings.supersampling > 1 && kind != MeshMapKind::Id {
                for (pixel, &count) in dense.chunks_exact_mut(components).zip(&counts) {
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
            padding_pool.install(|| {
                subcraft_bake::dilate_cancellable(
                    settings.size as usize,
                    settings.size as usize,
                    components,
                    &mut dense,
                    &covered,
                    progress,
                )
            })?;
            publish_dense_preview(progress, kind, settings.size, &dense, None, true);
            let mut fill = [0.0; 4];
            fill[..components].copy_from_slice(&neutral[..components]);
            output.insert(
                kind,
                TiledImage::from_dense(settings.size, settings.size, components, fill, &dense),
            );
            progress.advance(1);
        }
        progress.set_phase(BakePhase::Completed);
        Ok(output)
    }
}

#[cfg(test)]
mod tests;
