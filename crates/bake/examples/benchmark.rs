//! Reproducible CPU bake measurement using an actual OBJ high poly with no UVs.
//! cargo run --release -p subcraft-bake --example benchmark -- 256 1 3
//! Arguments: output size, samples per axis (1/2/4), measured iterations.

use std::error::Error;
use std::path::Path;
use std::time::Instant;

use subcraft_bake::{BakeInput, BakeSettings, Geometry, Progress, bake};
use subcraft_core::MeshMapKind;
use subcraft_mesh::{Bvh, import_high_poly, primitives};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let number = |i: usize, default: u32| -> Result<u32, Box<dyn Error>> {
        Ok(match args.get(i) {
            Some(value) => value.parse()?,
            None => default,
        })
    };
    let settings = BakeSettings {
        size: number(0, 256)?,
        supersampling: number(1, 1)?,
        maps: vec![
            MeshMapKind::Normal,
            MeshMapKind::AmbientOcclusion,
            MeshMapKind::Curvature,
            MeshMapKind::Thickness,
        ],
        ao_samples: 32,
        thickness_samples: 16,
        cage_front: 0.15,
        cage_rear: 0.15,
        ..Default::default()
    };
    settings.validate()?;
    let iterations = number(2, 3)?.clamp(1, 100);
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../mesh/tests/fixtures/high-poly-no-uv.obj");
    let import_start = Instant::now();
    let report = import_high_poly(&source)?;
    let import_ms = import_start.elapsed().as_secs_f64() * 1000.0;
    let low = primitives::plane(2.0, 8);
    let bvh_start = Instant::now();
    let (lb, hb) = (Bvh::build(&low), Bvh::build(&report.mesh));
    let bvh_ms = bvh_start.elapsed().as_secs_f64() * 1000.0;
    let input = BakeInput {
        low: Geometry {
            mesh: &low,
            bvh: &lb,
        },
        high: Some(Geometry {
            mesh: &report.mesh,
            bvh: &hb,
        }),
        material_slot: 0,
    };
    println!(
        "SubCraft UVless CPU bake | size={} | samples/pixel={} | rayon_threads={} | high_triangles={} | missing_uv_triangles={} | generated_normals={} | import_ms={import_ms:.3} | bvh_ms={bvh_ms:.3}",
        settings.size,
        settings.supersampling.pow(2),
        rayon::current_num_threads(),
        report.mesh.triangle_count(),
        report.stats.missing_uv_triangles,
        report.stats.generated_normals
    );
    // Warm-up excludes Rayon initialization and one-time allocator effects from measurements.
    bake(&input, &settings, &Progress::new())?;
    let mut timings = Vec::new();
    for i in 0..iterations {
        let start = Instant::now();
        let maps = bake(&input, &settings, &Progress::new())?;
        let millis = start.elapsed().as_secs_f64() * 1000.0;
        let checksum: f64 = maps
            .values()
            .flat_map(|map| map.to_dense())
            .map(f64::from)
            .sum();
        println!(
            "iteration={} | bake_ms={millis:.3} | maps={} | checksum={checksum:.6}",
            i + 1,
            maps.len()
        );
        timings.push(millis);
    }
    timings.sort_by(f64::total_cmp);
    println!(
        "median_ms={:.3} | min_ms={:.3} | max_ms={:.3}",
        timings[timings.len() / 2],
        timings[0],
        timings[timings.len() - 1]
    );
    Ok(())
}
