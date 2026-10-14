//! Select a bake device on the worker, keeping expensive initialization off the UI thread.
use crate::panels::baking::Acceleration;
use std::{collections::BTreeMap, sync::Arc};
use subcraft_bake::{BakeBackend, BakeError, BakeInput, BakeSettings, CpuBakeBackend, Progress};
use subcraft_core::MeshMapKind;
use subcraft_image::TiledImage;

#[derive(Default)]
pub struct BakeBackends {
    #[cfg(not(target_arch = "wasm32"))]
    gpu: Arc<std::sync::OnceLock<Result<subcraft_bake_gpu::GpuBaker, String>>>,
}

impl BakeBackends {
    /// Retain the device/pipeline across jobs; initialization still happens only on the worker.
    pub fn backend(&self, mode: Acceleration) -> Arc<dyn BakeBackend> {
        if mode == Acceleration::Cpu {
            return Arc::new(CpuBakeBackend);
        }
        Arc::new(LazyGpu {
            mode,
            #[cfg(not(target_arch = "wasm32"))]
            gpu: self.gpu.clone(),
        })
    }
}

struct LazyGpu {
    mode: Acceleration,
    #[cfg(not(target_arch = "wasm32"))]
    gpu: Arc<std::sync::OnceLock<Result<subcraft_bake_gpu::GpuBaker, String>>>,
}

impl BakeBackend for LazyGpu {
    fn name(&self) -> &str {
        "Inicializando GPU…"
    }
    fn bake(
        &self,
        input: &BakeInput<'_>,
        settings: &BakeSettings,
        progress: &Progress,
    ) -> Result<BTreeMap<MeshMapKind, TiledImage>, BakeError> {
        if progress.is_cancelled() {
            return Err(BakeError::Cancelled);
        }
        #[cfg(not(target_arch = "wasm32"))]
        let result = match self.gpu.get_or_init(subcraft_bake_gpu::GpuBaker::new) {
            Ok(gpu) => gpu.bake(input, settings, progress),
            Err(reason) => Err(BakeError::Backend(reason.clone())),
        };
        #[cfg(target_arch = "wasm32")]
        let result = Err(BakeError::Backend(
            "Bake GPU disponível no aplicativo desktop".into(),
        ));
        match result {
            Err(BakeError::Backend(reason))
                if self.mode == Acceleration::Automatic && !progress.is_cancelled() =>
            {
                log::warn!("GPU bake fallback: {reason}");
                progress.set_backend(&format!("CPU · fallback automático ({reason})"));
                CpuBakeBackend.bake(input, settings, progress)
            }
            other => other,
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use subcraft_bake::Geometry;
    use subcraft_mesh::{Bvh, primitives};

    #[test]
    fn forced_gpu_reports_failure_while_auto_explicitly_falls_back() {
        let low = primitives::plane(1.0, 1);
        let bvh = Bvh::build(&low);
        let input = BakeInput {
            low: Geometry {
                mesh: &low,
                bvh: &bvh,
            },
            high: None,
            material_slot: 0,
        };
        let settings = BakeSettings {
            size: 16,
            maps: vec![MeshMapKind::Position],
            ..Default::default()
        };
        for mode in [Acceleration::Gpu, Acceleration::Automatic] {
            let lazy = LazyGpu {
                mode,
                gpu: Arc::new(std::sync::OnceLock::from(Err(
                    "dispositivo indisponível para teste".into(),
                ))),
            };
            let progress = Progress::new();
            let result = lazy.bake(&input, &settings, &progress);
            if mode == Acceleration::Gpu {
                assert!(matches!(result, Err(BakeError::Backend(_))));
                assert!(progress.previews().is_empty());
            } else {
                assert!(result.is_ok());
                assert!(
                    progress
                        .snapshot()
                        .backend
                        .starts_with("CPU · fallback automático")
                );
                assert_eq!(progress.previews().len(), 1);
            }
        }
    }
}
