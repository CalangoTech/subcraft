//! Workspace preferences and background recovery saves (desktop only).

use crate::state::UiState;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub const SETTINGS_KEY: &str = "subcraft_studio_preferences_v1";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Preferences {
    pub recent: Vec<PathBuf>,
    pub autosave: bool,
    pub autosave_seconds: u32,
    pub viewport_scale: f32,
}

impl Default for Preferences {
    fn default() -> Self {
        Self {
            recent: Vec::new(),
            autosave: true,
            autosave_seconds: 120,
            viewport_scale: 1.0,
        }
    }
}

impl Preferences {
    pub fn remember(&mut self, path: PathBuf) {
        self.recent.retain(|p| p != &path);
        self.recent.insert(0, path);
        self.recent.truncate(12);
    }
}

#[derive(Default)]
pub struct Recovery {
    #[cfg(not(target_arch = "wasm32"))]
    job: Option<std::sync::mpsc::Receiver<Result<PathBuf, String>>>,
    #[cfg(not(target_arch = "wasm32"))]
    last_check: f64,
    #[cfg(not(target_arch = "wasm32"))]
    saved: Option<(u64, u64)>,
    #[cfg(not(target_arch = "wasm32"))]
    session_key: Option<u64>,
    #[cfg(not(target_arch = "wasm32"))]
    file: Option<PathBuf>,
}

impl Recovery {
    pub fn folder() -> Option<PathBuf> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::env::var_os("SUBCRAFT_RECOVERY")
                .filter(|p| !p.is_empty())
                .map(PathBuf::from)
                .or_else(|| {
                    std::env::current_exe().ok().and_then(|p| {
                        p.parent()
                            .filter(|d| d.join("PORTABLE").is_file())
                            .map(|d| d.join("Recovery"))
                    })
                })
                .or_else(|| {
                    dirs::data_local_dir().map(|d| d.join("SubCraft Studio").join("Recovery"))
                })
        }
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
    }

    /// List only recovery files produced by this app; opening still uses the unsaved prompt.
    pub fn files() -> Vec<PathBuf> {
        let mut paths: Vec<PathBuf> = Self::folder()
            .and_then(|root| std::fs::read_dir(root).ok())
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.is_file() && is_recovery_file(p))
            .collect();
        paths.sort_by_key(|p| std::cmp::Reverse(p.metadata().and_then(|m| m.modified()).ok()));
        paths.truncate(12);
        paths
    }

    pub fn poll(
        &mut self,
        state: &mut UiState,
        prefs: &Preferences,
        ctx: &egui::Context,
        now: f64,
    ) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(job) = &self.job {
                match job.try_recv() {
                    Ok(Ok(path)) => {
                        state.set_status(format!("Recovery saved: {}", path.display()), now);
                        self.job = None;
                    }
                    Ok(Err(error)) => {
                        state.set_status(format!("Recovery save failed: {error}"), now);
                        self.job = None;
                        self.saved = None;
                    }
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.job = None;
                        self.saved = None;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        ctx.request_repaint_after(std::time::Duration::from_millis(200));
                        return;
                    }
                }
            }
            if !prefs.autosave {
                return;
            }
            let interval = prefs.autosave_seconds.clamp(30, 1800) as f64;
            ctx.request_repaint_after(std::time::Duration::from_secs_f64(
                (interval - (now - self.last_check)).max(1.0),
            ));
            if now - self.last_check < interval || state.session.is_stroking() {
                return;
            }
            self.last_check = now;
            let key = state.session_generation;
            let revision = state.session.revision();
            if !state.session.is_modified() || self.saved == Some((key, revision)) {
                return;
            }
            let Some(snapshot) = state.session.recovery_snapshot() else {
                return;
            };
            let Some(folder) = Self::folder() else { return };
            if self.session_key != Some(key) {
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                self.file =
                    Some(folder.join(format!("recovery-{stamp}-{}.subcraft", std::process::id())));
                self.session_key = Some(key);
            }
            let Some(path) = self.file.clone() else {
                return;
            };
            let (tx, rx) = std::sync::mpsc::channel();
            self.job = Some(rx);
            self.saved = Some((key, revision));
            let ctx = ctx.clone();
            let protected = state.session.project_path().map(PathBuf::from);
            subcraft_core::task::spawn(move || {
                let result = std::fs::create_dir_all(&folder)
                    .map_err(|e| e.to_string())
                    .and_then(|_| snapshot.save(&path))
                    .map(|_| {
                        prune_recovery(&folder, &path, protected.as_deref());
                        path
                    });
                let _ = tx.send(result);
                ctx.request_repaint();
            });
        }
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (state, prefs, ctx, now);
        }
    }
}

fn is_recovery_file(path: &std::path::Path) -> bool {
    let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
        return false;
    };
    let Some(stem) = name
        .strip_prefix("recovery-")
        .and_then(|s| s.strip_suffix(".subcraft"))
    else {
        return false;
    };
    let Some((stamp, process)) = stem.split_once('-') else {
        return false;
    };
    !stamp.is_empty()
        && !process.is_empty()
        && stamp.bytes().all(|c| c.is_ascii_digit())
        && process.bytes().all(|c| c.is_ascii_digit())
}

/// Retain the twenty newest app-generated session copies. Ordinary files and the
/// currently opened recovery project are protected; no directories are removed.
#[cfg(not(target_arch = "wasm32"))]
fn prune_recovery(
    folder: &std::path::Path,
    current: &std::path::Path,
    protected: Option<&std::path::Path>,
) {
    let Ok(entries) = std::fs::read_dir(folder) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.is_file() && is_recovery_file(p))
        .collect();
    paths.sort_by_key(|p| {
        (
            std::cmp::Reverse(p.metadata().and_then(|m| m.modified()).ok()),
            std::cmp::Reverse(p.clone()),
        )
    });
    for path in paths.into_iter().skip(20) {
        if path != current && protected != Some(path.as_path()) && path.parent() == Some(folder)
            && let Err(error) = std::fs::remove_file(&path)
        {
            log::warn!("old recovery retained: {}: {error}", path.display());
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn recovery_retention_preserves_originals_and_open_recovery() {
        let root = std::env::temp_dir().join(format!(
            "subcraft-retention-{}",
            subcraft_core::LayerId::new().0
        ));
        std::fs::create_dir_all(&root).unwrap();
        for i in 0..25 {
            std::fs::write(
                root.join(format!("recovery-{i:03}-123.subcraft")),
                b"backup",
            )
            .unwrap();
        }
        let current = root.join("recovery-000-123.subcraft");
        let protected = root.join("recovery-001-123.subcraft");
        for name in [
            "my-project.subcraft",
            "recovery-manual.subcraft",
            "notes.txt",
        ] {
            std::fs::write(root.join(name), b"original").unwrap();
        }
        prune_recovery(&root, &current, Some(&protected));
        assert!(current.exists() && protected.exists());
        assert!(
            std::fs::read_dir(&root)
                .unwrap()
                .filter_map(Result::ok)
                .filter(|e| is_recovery_file(&e.path()))
                .count()
                <= 22
        );
        for name in [
            "my-project.subcraft",
            "recovery-manual.subcraft",
            "notes.txt",
        ] {
            assert_eq!(std::fs::read(root.join(name)).unwrap(), b"original");
        }
        for file in std::fs::read_dir(&root).unwrap().filter_map(Result::ok) {
            std::fs::remove_file(file.path()).unwrap();
        }
        std::fs::remove_dir(root).unwrap();
    }
}
