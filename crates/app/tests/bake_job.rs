use std::path::Path;
use std::time::{Duration, Instant};

use subcraft_app::{BakeRequest, BakeSettings, MeshMapKind, Session};
use subcraft_mesh::primitives;

#[test]
fn background_job_imports_a_real_high_poly_obj_without_uvs() {
    let mut session = Session::new(primitives::plane(2.0, 1), 64);
    let ts = session.document().texture_sets[0].id;
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../mesh/tests/fixtures/high-poly-no-uv.obj");
    let mut job = session.start_bake(BakeRequest {
        settings: BakeSettings {
            size: 64,
            maps: vec![MeshMapKind::WorldSpaceNormal],
            cage_front: 0.15,
            cage_rear: 0.15,
            ..Default::default()
        },
        high_poly: Some(path),
        texture_sets: vec![(ts, 0)],
    });
    let deadline = Instant::now() + Duration::from_secs(30);
    let result = loop {
        if let Some(result) = job.poll(&mut session) {
            break result;
        }
        assert!(
            Instant::now() < deadline,
            "background bake must finish or report failure"
        );
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(result, Ok(1));
    let normal = session.mesh_maps(ts).unwrap().get(MeshMapKind::WorldSpaceNormal).unwrap().get(32, 32);
    assert!(
        normal[0] < 0.46 && normal[1] > 0.99,
        "the source slope must reach the session: {normal:?}"
    );
    assert_eq!(job.fraction(), 1.0);
}
