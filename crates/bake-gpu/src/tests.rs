use super::*;
use subcraft_bake::{Geometry, bake};
use subcraft_mesh::{Corner, MeshBuilder, primitives};

fn compare(input: &BakeInput<'_>, settings: &BakeSettings, gpu: &GpuBaker) {
    let cpu = bake(input, settings, &Progress::new()).unwrap();
    let progress = Progress::new();
    let accelerated = gpu.bake(input, settings, &progress).unwrap();
    for kind in &settings.maps {
        let mut error = 0.0f32;
        let mut maximum = 0.0f32;
        for y in 0..settings.size {
            for x in 0..settings.size {
                let (a, b) = (cpu[kind].get(x, y), accelerated[kind].get(x, y));
                for c in 0..kind.components() {
                    let delta = (a[c] - b[c]).abs();
                    error += delta;
                    maximum = maximum.max(delta);
                }
            }
        }
        let mean = error / (settings.size * settings.size) as f32 / kind.components() as f32;
        let (mean_limit, max_limit) = match kind {
            // GPU transcendental functions differ near triangle/occlusion boundaries.
            MeshMapKind::AmbientOcclusion => (0.012, 0.15),
            MeshMapKind::Thickness => (0.008, 0.12),
            MeshMapKind::Curvature => (0.004, 0.08),
            _ => (0.001, 0.01),
        };
        assert!(
            mean < mean_limit && maximum < max_limit,
            "{kind:?}: CPU/GPU mean={mean:.6}, max={maximum:.6}"
        );
    }
    assert_eq!(progress.snapshot().phase, BakePhase::Completed);
    assert_eq!(progress.previews().len(), settings.maps.len());
    assert!((progress.fraction() - 1.0).abs() < 1e-6);
}

/// Small real-driver regression: all maps, nontrivial occlusion, cancellation, high poly with
/// deliberately removed UVs, projection misses, DirectX normals and AA. No stress workload.
#[test]
fn hardware_bake_matches_cpu_and_handles_uvless_high_poly() {
    let gpu = match GpuBaker::new() {
        Ok(gpu) => gpu,
        Err(e) if !e.contains("shader") && std::env::var_os("SUBCRAFT_REQUIRE_GPU").is_none() => {
            eprintln!("Skipping hardware regression: {e}");
            return;
        }
        Err(e) => panic!("Hardware regression could not initialize GPU: {e}"),
    };
    eprintln!("Testing {}", gpu.name());
    let mesh = primitives::cube(2.0);
    let bvh = Bvh::build(&mesh);
    let input = BakeInput {
        low: Geometry {
            mesh: &mesh,
            bvh: &bvh,
        },
        high: None,
        material_slot: 0,
    };
    let settings = BakeSettings {
        size: 24,
        ao_samples: 16,
        thickness_samples: 16,
        curvature_radius: 0.2,
        ..Default::default()
    };
    compare(&input, &settings, &gpu);

    // A separate occluder over a floor exercises actual AO hits and concave curvature.
    let mut builder = MeshBuilder::new("Occlusion fixture");
    let floor_material = builder.material("Floor");
    let box_material = builder.material("Occluder");
    for (source, material, translation) in [
        (primitives::plane(4.0, 2), floor_material, Vec3::ZERO),
        (
            primitives::cube(1.0),
            box_material,
            Vec3::new(0.0, 0.5, 0.0),
        ),
    ] {
        for triangle in 0..source.triangle_count() {
            let [a, b, c] = source.tri(triangle).map(|i| Corner {
                position: source.positions[i as usize] + translation,
                normal: Some(source.normals[i as usize]),
                uv: Some(source.uvs[i as usize]),
            });
            builder.push_triangle(material, [a, b, c]);
        }
    }
    let floor = builder.build().unwrap();
    let floor_bvh = Bvh::build(&floor);
    let input = BakeInput {
        low: Geometry {
            mesh: &floor,
            bvh: &floor_bvh,
        },
        high: None,
        material_slot: 0,
    };
    let settings = BakeSettings {
        size: 24,
        ao_samples: 32,
        curvature_radius: 0.06,
        maps: vec![MeshMapKind::AmbientOcclusion, MeshMapKind::Curvature],
        ..Default::default()
    };
    compare(&input, &settings, &gpu);

    let low = primitives::plane(2.0, 1);
    let low_bvh = Bvh::build(&low);
    let mut builder = MeshBuilder::new("UVless tilted high");
    let material = builder.material("High");
    // Nonuniform raised surface: projection hits on one side and intentionally misses outside.
    let point = |x: f32, z: f32| Corner {
        position: Vec3::new(x, 0.06 + x * 0.05, z),
        normal: None,
        uv: None,
    };
    builder.push_triangle(
        material,
        [point(-0.5, -1.0), point(-0.5, 1.0), point(0.8, 1.0)],
    );
    builder.push_triangle(
        material,
        [point(-0.5, -1.0), point(0.8, 1.0), point(0.8, -1.0)],
    );
    let mut high = builder.build_for_baking().unwrap();
    high.uvs.clear();
    high.tangents.clear(); // baker must never read high UV/tangents.
    let high_bvh = Bvh::build(&high);
    let input = BakeInput {
        low: Geometry {
            mesh: &low,
            bvh: &low_bvh,
        },
        high: Some(Geometry {
            mesh: &high,
            bvh: &high_bvh,
        }),
        material_slot: 0,
    };
    let settings = BakeSettings {
        size: 24,
        supersampling: 2,
        ao_samples: 8,
        thickness_samples: 8,
        cage_front: 0.08,
        cage_rear: 0.02,
        normal_flip_y: true,
        id_seed: 3,
        ..Default::default()
    };
    compare(&input, &settings, &gpu);

    let cancelled = Progress::new();
    cancelled.cancel();
    assert!(matches!(
        gpu.bake(&input, &settings, &cancelled),
        Err(BakeError::Cancelled)
    ));
}

#[test]
#[ignore = "Small manual GPU benchmark; no stress workload"]
fn small_hardware_benchmarks() {
    let gpu = GpuBaker::new().expect("hardware GPU");
    let low = primitives::uv_sphere(1.0, 16, 8);
    let low_bvh = Bvh::build(&low);
    for (size, segments, rings) in [(128, 64, 32), (256, 128, 64)] {
        let mut high = primitives::uv_sphere(1.02, segments, rings);
        high.uvs.clear();
        high.tangents.clear();
        let high_bvh = Bvh::build(&high);
        let input = BakeInput {
            low: Geometry {
                mesh: &low,
                bvh: &low_bvh,
            },
            high: Some(Geometry {
                mesh: &high,
                bvh: &high_bvh,
            }),
            material_slot: 0,
        };
        let settings = BakeSettings {
            size,
            ao_samples: 32,
            thickness_samples: 16,
            cage_front: 0.02,
            cage_rear: 0.02,
            ..Default::default()
        };
        let cpu_start = Instant::now();
        let cpu = bake(&input, &settings, &Progress::new()).unwrap();
        let cpu_time = cpu_start.elapsed();
        let gpu_start = Instant::now();
        let maps = gpu.bake(&input, &settings, &Progress::new()).unwrap();
        let gpu_time = gpu_start.elapsed();
        eprintln!(
            "BENCH {}: {size}²/7maps/high={}tri/AO32/thickness16 CPU={cpu_time:?} GPU={gpu_time:?}",
            gpu.name(),
            high.triangle_count()
        );
        assert_eq!(cpu.len(), maps.len());
        // Cached detail offsets cross several 8192-texel batches; spot-check every map.
        for kind in &settings.maps {
            let mut mean = 0.0;
            let mut count = 0;
            for y in (0..size).step_by(11) {
                for x in (0..size).step_by(13) {
                    let a = cpu[kind].get(x, y);
                    let b = maps[kind].get(x, y);
                    mean += a[..kind.components()]
                        .iter()
                        .zip(&b)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f32>();
                    count += kind.components();
                }
            }
            assert!(mean / (count as f32) < 0.02, "cached map parity {kind:?}");
        }
    }
}

#[test]
fn gpu_memory_guard_precedes_uv_allocation() {
    // Guard arithmetic is deterministic and does not initialize the GPU.
    assert_eq!(std::mem::size_of::<GpuSample>(), 80);
    assert_eq!(std::mem::size_of::<GpuTriangle>(), 112);
    assert_eq!(std::mem::size_of::<GpuNode>(), 48);
    assert_eq!(std::mem::size_of::<Params>(), 128);
    let settings = BakeSettings {
        size: 4096,
        memory_limit_mb: 64,
        ..Default::default()
    };
    assert!(matches!(
        settings.validate_memory_budget(),
        Err(BakeError::MemoryBudget { .. })
    ));
    assert_eq!(TEXELS_PER_DISPATCH, 128);
    assert_eq!(RAYS_PER_DISPATCH, 4);
}
