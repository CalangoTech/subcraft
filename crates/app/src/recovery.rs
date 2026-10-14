//! Copy-on-write project snapshots for background recovery saves.

use std::{path::Path, sync::Arc};
use subcraft_doc::{Document, ResourceStore};
use subcraft_mesh::Mesh;
use subcraft_project::MeshMapSet;

use crate::Session;

pub struct ProjectSnapshot {
    mesh: Arc<Mesh>,
    document: Document,
    resources: ResourceStore,
    maps: MeshMapSet,
}

impl ProjectSnapshot {
    /// Atomic write, independent of the live session's saved path and modified flag.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        subcraft_project::save_project(
            path,
            &self.mesh,
            &self.document,
            &self.resources,
            &self.maps,
        )
        .map_err(|e| e.to_string())
    }
}

impl Session {
    /// Called only between strokes. Image storage, resources and geometry are shared until edited.
    pub fn recovery_snapshot(&self) -> Option<ProjectSnapshot> {
        if self.is_stroking() {
            return None;
        }
        Some(ProjectSnapshot {
            mesh: self.scene().mesh.clone(),
            document: self.document().clone(),
            resources: self.resources().clone(),
            maps: self
                .document()
                .texture_sets
                .iter()
                .filter_map(|ts| {
                    self.mesh_maps(ts.id)
                        .map(|maps| (ts.id, maps.as_map().clone()))
                })
                .collect(),
        })
    }
}
