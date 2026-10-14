use std::path::Path;
use std::time::{Duration, Instant};

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use subcraft_app::{BakeRequest, BakeSettings, MeshMapKind, Session};
use subcraft_bake::{BakeBackend, BakeError, BakeInput, BakePhase, Progress};
use subcraft_image::TiledImage;
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
    let normal = session
        .mesh_maps(ts)
        .unwrap()
        .get(MeshMapKind::WorldSpaceNormal)
        .unwrap()
        .get(32, 32);
    assert!(
        normal[0] < 0.46 && normal[1] > 0.99,
        "the source slope must reach the session: {normal:?}"
    );
    assert_eq!(job.fraction(), 1.0);
}

#[test]
fn supplied_backend_runs_in_background_and_exposes_preview() {
    struct TestBackend(Arc<AtomicUsize>);
    impl BakeBackend for TestBackend {
        fn name(&self) -> &str {
            "Test hardware backend"
        }
        fn bake(
            &self,
            _: &BakeInput<'_>,
            settings: &BakeSettings,
            progress: &Progress,
        ) -> Result<BTreeMap<MeshMapKind, TiledImage>, BakeError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            progress.begin(1);
            progress.set_map(MeshMapKind::Position, 1, 1);
            progress.set_map_work(1);
            progress.publish_preview(MeshMapKind::Position, 1, 1, vec![64, 128, 191, 255], true);
            progress.advance(1);
            progress.set_phase(BakePhase::Completed);
            Ok(BTreeMap::from([(
                MeshMapKind::Position,
                TiledImage::new(settings.size, settings.size, 3, [0.25, 0.5, 0.75, 0.0]),
            )]))
        }
    }
    let mut session = Session::new(primitives::plane(2.0, 1), 32);
    let ts = session.document().texture_sets[0].id;
    let calls = Arc::new(AtomicUsize::new(0));
    let mut job = session.start_bake_with_backend(
        BakeRequest {
            settings: BakeSettings {
                size: 32,
                maps: vec![MeshMapKind::Position],
                ..Default::default()
            },
            high_poly: None,
            texture_sets: vec![(ts, 0)],
        },
        Arc::new(TestBackend(calls.clone())),
    );
    let deadline = Instant::now() + Duration::from_secs(10);
    let result = loop {
        if let Some(result) = job.poll(&mut session) {
            break result;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(result, Ok(1));
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(job.current_texture_set(), Some(ts));
    assert_eq!(job.texture_sets_progress(), (1, 1));
    assert_eq!(job.snapshot().backend, "Test hardware backend");
    assert_eq!(job.previews()[0].rgba, vec![64, 128, 191, 255]);
    assert_eq!(
        session
            .mesh_maps(ts)
            .unwrap()
            .get(MeshMapKind::Position)
            .unwrap()
            .get(16, 16)[..3],
        [0.25, 0.5, 0.75]
    );
}
