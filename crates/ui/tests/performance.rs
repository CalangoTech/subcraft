use subcraft_app::{Command, NewLayer, Session};
use subcraft_mesh::primitives;
use subcraft_ui::state::UiState;

#[test]
fn synchronization_refreshes_after_edit_undo_and_session_replacement() {
    let session = Session::new(primitives::cube(2.0), 128);
    let mut ui = UiState::new(session, None);
    ui.sync_gpu();
    assert_eq!(ui.synced_revision, Some(ui.session.revision()));
    assert!(!ui.gpu_dirty);
    let ts = ui.session.selection().texture_set;
    let before = ui.session.composite(ts).unwrap().channels.clone();
    for _ in 0..100 {
        ui.sync_gpu();
    }
    ui.session
        .execute(Command::AddLayer(NewLayer::Fill))
        .unwrap();
    ui.sync_gpu();
    assert_eq!(ui.synced_revision, Some(ui.session.revision()));
    ui.session.execute(Command::Undo).unwrap();
    ui.sync_gpu();
    assert_eq!(ui.session.composite(ts).unwrap().channels, before);
    ui.set_session(Session::new(primitives::cube(1.0), 64));
    assert!(ui.synced_revision.is_none());
    ui.sync_gpu();
    let ts = ui.session.selection().texture_set;
    assert_eq!(ui.session.composite(ts).unwrap().resolution, 64);
}

fn gpu_ui() -> Option<UiState> {
    use std::sync::Arc;
    use subcraft_render::{egui_wgpu, headless, wgpu_api as wgpu};
    let (adapter, device, queue) = headless::device()?;
    let format = wgpu::TextureFormat::Rgba8Unorm;
    let egui_renderer = egui_wgpu::Renderer::new(&device, format, Default::default());
    let rs = egui_wgpu::RenderState {
        adapter,
        available_adapters: Vec::new(),
        instance: wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle()),
        device,
        queue,
        target_format: format,
        renderer: Arc::new(egui::epaint::mutex::RwLock::new(egui_renderer)),
        surface_config: egui_wgpu::SurfaceConfig::LOW_LATENCY,
    };
    Some(UiState::new(
        Session::new(primitives::cube(2.0), 128),
        Some(rs),
    ))
}

#[test]
fn export_preview_cannot_consume_pending_gpu_uploads() {
    use subcraft_core::{ChannelKind, ChannelValue};
    use subcraft_doc::{FillChannel, FillSource};
    let Some(mut ui) = gpu_ui() else { return };
    ui.sync_gpu();
    let generation = ui.gpu_generation;
    ui.session
        .execute(Command::AddLayer(NewLayer::Fill))
        .unwrap();
    let layer = ui.session.selection().layer.unwrap();
    ui.session
        .execute(Command::SetFillChannel {
            layer,
            channel: ChannelKind::BaseColor,
            fill: FillChannel::new(FillSource::uniform(ChannelValue::rgb(0.8, 0.02, 0.02))),
        })
        .unwrap();
    ui.session.export_plan(&Default::default()); // Evaluates before the UI upload pass.
    ui.sync_gpu();
    assert!(ui.gpu_generation > generation);
    let ts = ui.session.selection().texture_set;
    let uploaded = &ui.uploaded_channels[&(ts, ChannelKind::BaseColor)];
    assert!(
        uploaded.same_storage(
            ui.session
                .composite(ts)
                .unwrap()
                .channel(ChannelKind::BaseColor)
                .unwrap()
        )
    );
    let params = ui.camera.view_params([128, 128]);
    let rs = ui.render_state.as_ref().unwrap();
    let bytes = ui
        .renderer
        .as_mut()
        .unwrap()
        .render(&rs.device, &rs.queue, [128, 128], &params)
        .read_rgba8(&rs.device, &rs.queue);
    let center = &bytes[(64 * 128 + 64) * 4..][..4];
    assert!(
        center[0] > 2 * center[1],
        "GPU should have the new red material: {center:?}"
    );
    let generation = ui.gpu_generation;
    for _ in 0..100 {
        ui.sync_gpu();
    }
    assert_eq!(generation, ui.gpu_generation);
    ui.session
        .execute(Command::Rename {
            layer,
            name: "Metadata only".into(),
        })
        .unwrap();
    ui.sync_gpu();
    assert_eq!(
        generation, ui.gpu_generation,
        "renaming must not reupload textures"
    );
}

#[test]
fn viewport_reuses_idle_frames_and_refreshes_camera_quality_and_environment() {
    let Some(mut state) = gpu_ui() else { return };
    let ctx = egui::Context::default();
    subcraft_ui::theme::apply(&ctx);
    let draw = |state: &mut UiState| {
        let input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(800.0, 600.0),
            )),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, |ui| subcraft_ui::viewport::show(state, ui));
        output.textures_delta.clear(); // This test renders the scene only, not egui font textures.
    };
    for _ in 0..3 {
        draw(&mut state);
    }
    let frames = state.viewport_render_count;
    for _ in 0..10 {
        draw(&mut state);
    }
    assert_eq!(
        frames, state.viewport_render_count,
        "moving the UI through idle passes must not rerender the 3D scene"
    );
    state.camera.orbit(10.0, 5.0);
    draw(&mut state);
    assert_eq!(frames + 1, state.viewport_render_count);
    state.viewport_scale = 0.5;
    draw(&mut state);
    assert_eq!(frames + 2, state.viewport_render_count);
    let mut environment = state.library.environments[state.env.index].clone();
    environment.id = subcraft_core::AssetId::from_name("cache-test-identical-lighting");
    state.library.environments.push(environment);
    state.env.index = state.library.environments.len() - 1;
    draw(&mut state);
    assert_eq!(
        frames + 3,
        state.viewport_render_count,
        "asset identity must invalidate even with identical lighting parameters"
    );
}

#[test]
#[ignore = "manual timing benchmark; no wall-clock assertion in CI"]
fn benchmark_idle_sync() {
    use std::{hint::black_box, time::Instant};
    let mut session = Session::new(primitives::cube(2.0), 128);
    for _ in 0..100 {
        session.execute(Command::AddLayer(NewLayer::Paint)).unwrap();
    }
    let mut ui = UiState::new(session, None);
    ui.sync_gpu();
    let iterations = 100_000;
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(&mut ui.session).update_composites();
    }
    let baseline = start.elapsed();
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(&mut ui).sync_gpu();
    }
    let cached = start.elapsed();
    println!(
        "IDLE_SYNC iterations={iterations} layers=101 baseline_ms={:.3} cached_ms={:.3} speedup={:.1}x",
        baseline.as_secs_f64() * 1000.0,
        cached.as_secs_f64() * 1000.0,
        baseline.as_secs_f64() / cached.as_secs_f64().max(1e-9)
    );
}
