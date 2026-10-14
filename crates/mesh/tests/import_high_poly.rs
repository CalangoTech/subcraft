//! Real-file regression for geometry-only high-poly import across every supported format.
use std::path::{Path, PathBuf};

use subcraft_mesh::{BuildError, ImportError, import_high_poly, import_mesh};

fn fixture(extension: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/fixtures/high-poly-no-uv.{extension}"))
}

fn assert_uvless_source(path: &Path) {
    let report = import_high_poly(path).unwrap();
    assert_eq!(report.mesh.triangle_count(), 2, "{}", path.display());
    assert_eq!(report.stats.missing_uv_triangles, 2);
    assert!(report.mesh.validate().is_empty());
    assert!(
        report
            .mesh
            .normals
            .iter()
            .all(|n| n.is_finite() && (n.length() - 1.0).abs() < 1e-4)
    );
    assert!(report.mesh.tangents.iter().all(|t| t.is_finite()));
    assert!(
        matches!(
            import_mesh(path),
            Err(ImportError::Build(BuildError::NoUvs))
        ),
        "the regular low-poly import must still require UVs"
    );
}

#[test]
fn obj_high_poly_does_not_require_uvs() {
    assert_uvless_source(&fixture("obj"));
}

#[test]
fn gltf_high_poly_needs_neither_uvs_nor_material_texture_files() {
    assert_uvless_source(&fixture("gltf"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn fbx_high_poly_does_not_require_uvs() {
    assert_uvless_source(&fixture("fbx"));
}

#[test]
fn glb_high_poly_does_not_require_uvs() {
    // Actual GLB file: JSON and binary chunks, with positions + indices and no TEXCOORD.
    let mut binary = Vec::new();
    for f in [
        -1.0f32, -0.1, -1.0, -1.0, -0.1, 1.0, 1.0, 0.1, 1.0, 1.0, 0.1, -1.0,
    ] {
        binary.extend_from_slice(&f.to_le_bytes());
    }
    for i in [0u16, 1, 2, 0, 2, 3] {
        binary.extend_from_slice(&i.to_le_bytes());
    }
    let mut json = br#"{"asset":{"version":"2.0"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],"meshes":[{"primitives":[{"attributes":{"POSITION":0},"indices":1}]}],"buffers":[{"byteLength":60}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":48},{"buffer":0,"byteOffset":48,"byteLength":12}],"accessors":[{"bufferView":0,"componentType":5126,"count":4,"type":"VEC3","min":[-1,-0.1,-1],"max":[1,0.1,1]},{"bufferView":1,"componentType":5123,"count":6,"type":"SCALAR"}]}"#.to_vec();
    while !json.len().is_multiple_of(4) {
        json.push(b' ');
    }
    let total = 12 + 8 + json.len() + 8 + binary.len();
    let mut glb = Vec::new();
    for word in [
        0x4654_6c67u32,
        2,
        total as u32,
        json.len() as u32,
        0x4e4f_534a,
    ] {
        glb.extend_from_slice(&word.to_le_bytes());
    }
    glb.extend_from_slice(&json);
    glb.extend_from_slice(&(binary.len() as u32).to_le_bytes());
    glb.extend_from_slice(&0x004e_4942u32.to_le_bytes());
    glb.extend_from_slice(&binary);
    let path = std::env::temp_dir().join(format!(
        "subcraft-high-poly-no-uv-{}.glb",
        std::process::id()
    ));
    std::fs::write(&path, glb).unwrap();
    assert_uvless_source(&path);
    std::fs::remove_file(path).unwrap();
}
