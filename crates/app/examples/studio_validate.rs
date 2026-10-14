//! Reproducible desktop-independent Studio workflow and a ready-to-open demo project.
//!
//! Run: cargo run -p subcraft-app --example studio_validate -- <output_dir>
//! Generates a genuinely UV-less subdivided OBJ, bakes every mesh map through BakeJob,
//! builds/paints/filters/organizes layers, checks undo/redo and project round-trip, and
//! exports/decodes PBR, Unity HDRP and Unreal texture sets. No UI or mouse automation.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, ensure};
use glam::{Vec2, Vec3};
use subcraft_app::{BakeRequest, BakeSettings, Camera, Command, CommandError, NewLayer, Session};
use subcraft_assets::Library;
use subcraft_core::{ChannelKind, ChannelValue, MeshMapKind};
use subcraft_doc::{Generator, GeneratorKind, LayerContent, MaskEffectKind};
use subcraft_export::{ExportSettings, ExportTemplate};
use subcraft_image::load_bitmap;
use subcraft_mesh::{Mesh, import_high_poly, import_mesh, primitives};
use subcraft_paint::{BrushMaterial, PaintError};

/// Subdivide every triangle into four without writing texture coordinates or normals.
/// Surface geometry stays identical to the low poly, so the bake has a deterministic cage hit.
fn write_uvless_high_poly(path: &Path, low: &Mesh) -> Result<()> {
    let mut out = BufWriter::new(File::create(path)?);
    writeln!(
        out,
        "# SubCraft Studio smoke: high-poly bake source without UVs"
    )?;
    writeln!(out, "o HighPoly_SemUV")?;
    let mut vertex = 1usize;
    for triangle in 0..low.triangle_count() {
        let [a, b, c] = low.tri_positions(triangle);
        let ab = (a + b) * 0.5;
        let bc = (b + c) * 0.5;
        let ca = (c + a) * 0.5;
        for face in [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]] {
            for p in face {
                writeln!(out, "v {} {} {}", p.x, p.y, p.z)?;
            }
            writeln!(out, "f {} {} {}", vertex, vertex + 1, vertex + 2)?;
            vertex += 3;
        }
    }
    out.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let started = Instant::now();
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("studio-validation"));
    fs::create_dir_all(&output)
        .with_context(|| format!("create output folder {}", output.display()))?;
    let mut session = Session::new(primitives::sample_crate(), 256);
    let sets: Vec<_> = session
        .document()
        .texture_sets
        .iter()
        .map(|ts| (ts.id, ts.material_slot, ts.name.clone()))
        .collect();
    ensure!(
        sets.len() == 2,
        "sample crate must have Body and Trim texture sets"
    );

    let high_path = output.join("HighPoly_SemUV.obj");
    write_uvless_high_poly(&high_path, &session.scene().mesh)?;
    let obj = fs::read_to_string(&high_path)?;
    ensure!(
        !obj.lines()
            .any(|line| line.starts_with("vt ") || line.starts_with("vn ")),
        "test source unexpectedly contains UVs/normals"
    );
    let high = import_high_poly(&high_path)?;
    ensure!(
        high.mesh.triangle_count() == session.scene().mesh.triangle_count() * 4,
        "high-poly subdivision count"
    );
    ensure!(
        high.mesh.uvs.iter().all(|uv| *uv == Vec2::ZERO),
        "high-poly UV placeholders must be zero"
    );
    ensure!(
        import_mesh(&high_path).is_err(),
        "the low-poly paint target must still require UVs"
    );
    println!(
        "HIGH_POLY_NO_UV_OK triangles={} missing_uv={}",
        high.mesh.triangle_count(),
        high.stats.missing_uv_triangles
    );

    let bake_started = Instant::now();
    let mut job = session.start_bake(BakeRequest {
        settings: BakeSettings {
            size: 128,
            ao_samples: 8,
            thickness_samples: 8,
            maps: MeshMapKind::ALL.to_vec(),
            ..BakeSettings::default()
        },
        high_poly: Some(high_path.clone()),
        texture_sets: sets.iter().map(|(id, slot, _)| (*id, *slot)).collect(),
    });
    let baked_sets = loop {
        if let Some(result) = job.poll(&mut session) {
            break result.map_err(anyhow::Error::msg)?;
        }
        if bake_started.elapsed() > Duration::from_secs(120) {
            job.cancel();
            anyhow::bail!("bake timed out after 120s: {}", job.stage());
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    ensure!(
        baked_sets == sets.len(),
        "all texture sets must complete baking"
    );
    for (id, _, name) in &sets {
        let maps = session
            .mesh_maps(*id)
            .context("bake did not install maps")?;
        ensure!(
            maps.len() == MeshMapKind::ALL.len(),
            "missing mesh maps for {name}"
        );
        for (kind, image) in maps.iter() {
            ensure!(
                image.width() == 128 && image.height() == 128,
                "unexpected bake dimensions for {kind:?}"
            );
            ensure!(
                image.to_dense().iter().all(|value| value.is_finite()),
                "non-finite bake values for {kind:?}"
            );
        }
    }
    let bake_seconds = bake_started.elapsed().as_secs_f64();
    println!(
        "BAKE_ALL_MAPS_OK sets={} maps={} seconds={:.3}",
        baked_sets,
        MeshMapKind::ALL.len(),
        bake_seconds
    );

    let library = Library::builtin();
    ensure!(
        library.smart_materials.len() >= 8 && library.smart_masks.len() >= 12,
        "Studio asset library is incomplete"
    );
    let body = sets
        .iter()
        .find(|(_, _, name)| name == "Body")
        .context("Body texture set")?
        .0;
    let trim = sets
        .iter()
        .find(|(_, _, name)| name == "Trim")
        .context("Trim texture set")?
        .0;
    session.execute(Command::SelectTextureSet(body))?;
    let enamel = library
        .smart_materials
        .iter()
        .find(|material| material.name == "Chipped Enamel")
        .context("Chipped Enamel preset")?;
    session.add_smart_material(enamel)?;
    let enamel_layer = session.selection().layer.context("enamel layer")?;
    session.execute(Command::Rename {
        layer: enamel_layer,
        name: "Esmalte azul com desgaste".into(),
    })?;
    session.execute(Command::SelectTextureSet(trim))?;
    let steel = library
        .materials
        .iter()
        .find(|material| material.name == "Steel")
        .context("Steel preset")?;
    session.add_material(steel)?;
    let steel_layer = session.selection().layer.context("steel layer")?;
    session.execute(Command::Rename {
        layer: steel_layer,
        name: "Estrutura de aco".into(),
    })?;

    session.execute(Command::SelectTextureSet(body))?;
    session.execute(Command::AddLayer(NewLayer::Paint))?;
    let paint = session.selection().layer.context("paint layer")?;
    session.execute(Command::Rename {
        layer: paint,
        name: "Marca de tinta laranja".into(),
    })?;
    session.brush.settings.size = 7.0;
    session.brush.settings.stroke_opacity = 0.9;
    session.brush.material = BrushMaterial::with(&[
        (ChannelKind::BaseColor, ChannelValue::rgb(0.95, 0.25, 0.035)),
        (ChannelKind::Roughness, ChannelValue::scalar(0.5)),
    ]);
    let camera = Camera::perspective(
        Vec3::new(1.5, 1.2, 3.0),
        Vec3::ZERO,
        Vec3::Y,
        40.0,
        0.05,
        100.0,
        [800, 600],
    );
    session.begin_stroke()?;
    let mut dirty_tiles = 0;
    for i in 0..24 {
        let t = i as f32 / 23.0;
        let point = Vec3::new(-0.25 + t * 0.5, -0.15 + t * 0.3, 0.5);
        let screen = camera
            .project(point)
            .context("project paint path")?
            .truncate();
        dirty_tiles += session.stroke_to(&camera, screen, false)?.len();
    }
    session.end_stroke();
    ensure!(dirty_tiles > 0, "paint stroke missed the sample mesh");
    ensure!(
        matches!(&session.selected_layer().context("paint layer")?.content, LayerContent::Paint(content) if content.channels.contains_key(&ChannelKind::BaseColor)),
        "stroke did not write the Base Color raster"
    );

    session.execute(Command::AddMask {
        layer: paint,
        white: false,
    })?;
    let mut front = Generator::new(GeneratorKind::Directional);
    front.direction = [0.0, 0.0, 1.0];
    front.level = 0.4;
    front.softness = 0.3;
    for kind in [
        MaskEffectKind::Generator(front),
        MaskEffectKind::Blur { radius: 2 },
        MaskEffectKind::Contrast {
            amount: 1.3,
            pivot: 0.5,
        },
    ] {
        session.execute(Command::AddMaskEffect { layer: paint, kind })?;
    }
    session.execute(Command::MoveDown(paint))?;
    let moved = session.document().clone();
    session.execute(Command::Undo)?;
    ensure!(
        session.document() != &moved,
        "move undo did not restore stack order"
    );
    session.execute(Command::Redo)?;
    ensure!(
        session.document() == &moved,
        "move redo did not restore stack order"
    );
    session.execute(Command::MoveUp(paint))?;
    session.execute(Command::SetLocked {
        layer: paint,
        locked: true,
    })?;
    ensure!(
        session.begin_stroke() == Err(PaintError::LayerLocked),
        "locked layer accepted a stroke"
    );
    ensure!(
        session.execute(Command::DeleteLayer(paint)) == Err(CommandError::LayerLocked),
        "locked layer accepted delete"
    );
    session.execute(Command::Undo)?;
    ensure!(
        !session.selected_layer().context("paint selection")?.locked,
        "lock undo"
    );
    session.execute(Command::Redo)?;
    ensure!(
        session.selected_layer().context("paint selection")?.locked,
        "lock redo"
    );
    println!(
        "LAYERS_PAINT_FILTER_UNDO_LOCK_OK smart_materials={} smart_masks={} painted_tile_updates={}",
        library.smart_materials.len(),
        library.smart_masks.len(),
        dirty_tiles
    );

    session.update_composites();
    let project_path = output.join("Projeto_Demo_Studio.subcraft");
    session
        .save(Some(&project_path))
        .map_err(anyhow::Error::msg)?;
    ensure!(!session.is_modified(), "saved session must be clean");
    let mut reopened = Session::open_project(&project_path).map_err(anyhow::Error::msg)?;
    ensure!(
        reopened.document() == session.document(),
        "project document round-trip differs"
    );
    ensure!(!reopened.is_modified(), "reopened session must be clean");
    for (id, _, name) in &sets {
        ensure!(
            reopened.mesh_maps(*id) == session.mesh_maps(*id),
            "baked maps changed on reopening {name}"
        );
    }
    reopened.update_composites();
    for (id, _, name) in &sets {
        ensure!(
            reopened.composite(*id) == session.composite(*id),
            "material composite changed on reopening {name}"
        );
    }

    let export_started = Instant::now();
    let mut png_count = 0;
    for (folder, template) in [
        ("PBR", ExportTemplate::pbr_metal_rough()),
        ("Unity_HDRP", ExportTemplate::unity_hdrp()),
        ("Unreal", ExportTemplate::unreal()),
    ] {
        let settings = ExportSettings {
            template,
            ..ExportSettings::default()
        };
        let files = reopened
            .export_textures(&output.join("Texturas").join(folder), &settings)
            .map_err(anyhow::Error::msg)?;
        ensure!(!files.is_empty(), "{folder} exported no files");
        for path in &files {
            let bitmap = load_bitmap(path)?;
            ensure!(
                bitmap.width == 256 && bitmap.height == 256,
                "wrong exported dimensions: {}",
                path.display()
            );
            ensure!(
                fs::metadata(path)?.len() > 0,
                "empty exported PNG: {}",
                path.display()
            );
            png_count += 1;
        }
        println!(
            "EXPORT_PNG_OK template={folder} files={} dimensions=256x256",
            files.len()
        );
    }
    let report = format!(
        "STUDIO_WORKFLOW_OK\nProjeto: {}\nHigh-poly sem UV: {} ({} triangulos)\nTexture sets: {}\nMesh maps por set: {}\nSmart materials: {}\nSmart masks: {}\nPNGs exportados e decodificados: {}\nBake: {:.3}s\nExport: {:.3}s\nTotal: {:.3}s\n",
        project_path.display(),
        high_path.display(),
        high.mesh.triangle_count(),
        sets.len(),
        MeshMapKind::ALL.len(),
        library.smart_materials.len(),
        library.smart_masks.len(),
        png_count,
        bake_seconds,
        export_started.elapsed().as_secs_f64(),
        started.elapsed().as_secs_f64()
    );
    fs::write(output.join("workflow-report.txt"), &report)?;
    println!(
        "STUDIO_WORKFLOW_OK texture_sets={} mesh_maps={} png_files={} total_seconds={:.3} project={}",
        sets.len(),
        sets.len() * MeshMapKind::ALL.len(),
        png_count,
        started.elapsed().as_secs_f64(),
        project_path.display()
    );
    Ok(())
}
