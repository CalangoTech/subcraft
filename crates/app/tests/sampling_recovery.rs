use glam::{Vec2, Vec3};
use subcraft_app::{Camera, Command, NewLayer, Session};
use subcraft_core::{ChannelKind, ChannelValue};
use subcraft_doc::{FillChannel, FillSource};
use subcraft_mesh::primitives;

fn filled() -> Session {
    let mut s = Session::new(primitives::cube(2.0), 128);
    s.execute(Command::AddLayer(NewLayer::Fill)).unwrap();
    let layer = s.selection().layer.unwrap();
    s.execute(Command::SetFillChannel {
        layer,
        channel: ChannelKind::BaseColor,
        fill: FillChannel::new(FillSource::uniform(ChannelValue::rgb(0.2, 0.4, 0.6))),
    })
    .unwrap();
    s.execute(Command::SetFillChannel {
        layer,
        channel: ChannelKind::Roughness,
        fill: FillChannel::new(FillSource::uniform(ChannelValue::scalar(0.65))),
    })
    .unwrap();
    s
}

#[test]
fn eyedropper_samples_linear_material_values_before_lighting() {
    let mut s = filled();
    let cam = Camera::perspective(
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::ZERO,
        Vec3::Y,
        40.0,
        0.1,
        50.0,
        [400, 400],
    );
    let color = s
        .sample_surface(&cam, Vec2::splat(200.0), ChannelKind::BaseColor)
        .unwrap();
    assert!((color.0[0] - 0.2).abs() < 0.001);
    assert!((color.0[1] - 0.4).abs() < 0.001);
    let rough = s
        .sample_surface(&cam, Vec2::splat(200.0), ChannelKind::Roughness)
        .unwrap();
    assert!((rough.0[0] - 0.65).abs() < 0.001);
    assert!(
        s.sample_surface(&cam, Vec2::ZERO, ChannelKind::BaseColor)
            .is_none()
    );
    let ts = s.selection().texture_set;
    assert_eq!(
        s.sample_uv(ts, Vec2::new(-0.8, 1.2), ChannelKind::BaseColor),
        s.sample_uv(ts, Vec2::splat(0.2), ChannelKind::BaseColor)
    );
    assert!(
        s.sample_uv(ts, Vec2::splat(f32::NAN), ChannelKind::BaseColor)
            .is_none()
    );
}

#[test]
fn recovery_snapshot_is_independent_and_does_not_mark_project_saved() {
    let mut s = filled();
    let path = std::env::temp_dir().join(format!(
        "subcraft-recovery-test-{}.subcraft",
        subcraft_core::LayerId::new().0
    ));
    let snapshot = s.recovery_snapshot().unwrap();
    let before = s.document().clone();
    s.execute(Command::AddLayer(NewLayer::Paint)).unwrap();
    snapshot.save(&path).unwrap();
    let recovered = Session::open_project(&path).unwrap();
    assert_eq!(recovered.document(), &before);
    assert!(s.is_modified());
    assert!(s.project_path().is_none());
    s.begin_stroke().unwrap();
    assert!(s.recovery_snapshot().is_none());
    s.cancel_stroke();
    std::fs::remove_file(path).unwrap();
}
