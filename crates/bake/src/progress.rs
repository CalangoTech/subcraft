//! Bounded progress shared by CPU/GPU bakers and the background job.
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use web_time::Instant;

use subcraft_core::MeshMapKind;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BakePhase {
    #[default]
    Preparing,
    Projecting,
    Baking,
    Padding,
    Completed,
}

impl BakePhase {
    pub fn label(self) -> &'static str {
        match self {
            Self::Preparing => "Preparando geometria e UVs",
            Self::Projecting => "Projetando high poly",
            Self::Baking => "Construindo textura",
            Self::Padding => "Preenchendo bordas UV",
            Self::Completed => "Concluído",
        }
    }
}

#[derive(Clone, Debug)]
pub struct BakePreview {
    pub kind: MeshMapKind,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub revision: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ProgressSnapshot {
    pub map: Option<MeshMapKind>,
    /// One-based index of the current map.
    pub map_index: usize,
    pub maps_total: usize,
    pub phase: BakePhase,
    pub fraction: f32,
    pub map_fraction: f32,
    pub backend: String,
    pub preview: Option<Arc<BakePreview>>,
}

#[derive(Default)]
struct State {
    snapshot: ProgressSnapshot,
    previews: BTreeMap<MeshMapKind, Arc<BakePreview>>,
    published: Option<Instant>,
    revision: u64,
}

/// A single latest preview per map; no frame queue can grow while the UI is busy.
#[derive(Default)]
pub struct Progress {
    done: AtomicUsize,
    total: AtomicUsize,
    map_done: AtomicUsize,
    map_total: AtomicUsize,
    cancel: AtomicBool,
    state: Mutex<State>,
}

impl Progress {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn fraction(&self) -> f32 {
        let total = self.total.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            (self.done.load(Ordering::Relaxed) as f32 / total as f32).min(1.0)
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Start a texture set without clearing the shared cancellation token.
    pub fn begin(&self, total: usize) {
        self.done.store(0, Ordering::Relaxed);
        self.total.store(total, Ordering::Relaxed);
        self.map_done.store(0, Ordering::Relaxed);
        self.map_total.store(0, Ordering::Relaxed);
        if let Ok(mut state) = self.state.lock() {
            state.snapshot.map = None;
            state.snapshot.map_index = 0;
            state.snapshot.maps_total = 0;
            state.snapshot.preview = None;
            state.snapshot.phase = BakePhase::Preparing;
            state.previews.clear();
            state.published = None;
        }
    }

    /// Change the work estimate after UV coverage is known, preserving the phase/preview.
    pub fn set_total(&self, total: usize) {
        self.total.store(total, Ordering::Relaxed);
    }

    pub fn advance(&self, amount: usize) {
        self.done.fetch_add(amount, Ordering::Relaxed);
        if self.map_total.load(Ordering::Relaxed) > 0 {
            self.map_done.fetch_add(amount, Ordering::Relaxed);
        }
    }

    pub fn reset_fraction(&self) {
        self.done.store(0, Ordering::Relaxed);
        self.total.store(0, Ordering::Relaxed);
    }

    pub fn set_backend(&self, name: &str) {
        if let Ok(mut state) = self.state.lock() {
            state.snapshot.backend = name.to_owned();
        }
    }

    pub fn set_phase(&self, phase: BakePhase) {
        if let Ok(mut state) = self.state.lock() {
            state.snapshot.phase = phase;
        }
    }

    pub fn set_map(&self, kind: MeshMapKind, index: usize, total: usize) {
        self.map_done.store(0, Ordering::Relaxed);
        self.map_total.store(0, Ordering::Relaxed);
        if let Ok(mut state) = self.state.lock() {
            state.snapshot.map = Some(kind);
            state.snapshot.map_index = index;
            state.snapshot.maps_total = total;
            state.snapshot.phase = BakePhase::Baking;
            state.snapshot.preview = state.previews.get(&kind).cloned();
            state.published = None;
        }
    }

    pub fn set_map_work(&self, total: usize) {
        self.map_total.store(total, Ordering::Relaxed);
    }

    /// Check before constructing a thumbnail. Publication rechecks under the same mutex.
    pub fn should_publish_preview(&self) -> bool {
        self.state.lock().is_ok_and(|s| {
            s.published
                .is_none_or(|t| t.elapsed() >= Duration::from_millis(100))
        })
    }

    /// Publish RGBA8 data (row order matches TiledImage). Each of the seven map previews is
    /// bounded to 512²; intermediate updates are throttled to 10 Hz. Completion can force one.
    pub fn publish_preview(
        &self,
        kind: MeshMapKind,
        width: u32,
        height: u32,
        rgba: Vec<u8>,
        force: bool,
    ) -> bool {
        if width == 0
            || height == 0
            || width > 512
            || height > 512
            || rgba.len() != width as usize * height as usize * 4
        {
            return false;
        }
        let Ok(mut state) = self.state.lock() else {
            return false;
        };
        if !force
            && state
                .published
                .is_some_and(|t| t.elapsed() < Duration::from_millis(100))
        {
            return false;
        }
        state.revision = state.revision.wrapping_add(1);
        let preview = Arc::new(BakePreview {
            kind,
            width,
            height,
            rgba,
            revision: state.revision,
        });
        state.previews.insert(kind, preview.clone());
        if state.snapshot.map == Some(kind) {
            state.snapshot.preview = Some(preview);
        }
        state.published = Some(Instant::now());
        true
    }

    pub fn snapshot(&self) -> ProgressSnapshot {
        let mut snapshot = self
            .state
            .lock()
            .map(|s| s.snapshot.clone())
            .unwrap_or_default();
        snapshot.fraction = self.fraction();
        let map_total = self.map_total.load(Ordering::Relaxed);
        snapshot.map_fraction = if map_total == 0 {
            0.0
        } else {
            (self.map_done.load(Ordering::Relaxed) as f32 / map_total as f32).min(1.0)
        };
        snapshot
    }

    pub fn previews(&self) -> Vec<Arc<BakePreview>> {
        self.state
            .lock()
            .map(|s| s.previews.values().cloned().collect())
            .unwrap_or_default()
    }
}
