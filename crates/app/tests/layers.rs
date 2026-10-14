//! Layer organization and protection stay undoable and cannot leave dangling selections.

use subcraft_app::{Command, CommandError, NewLayer, Session};
use subcraft_core::{ChannelKind, ChannelValue, LayerId};
use subcraft_doc::{FillChannel, FillSource, LayerContent};
use subcraft_mesh::primitives;
use subcraft_paint::PaintError;

fn session() -> Session {
    Session::new(primitives::cube(2.0), 128)
}

fn add(s: &mut Session, kind: NewLayer) -> LayerId {
    s.execute(Command::AddLayer(kind)).unwrap();
    s.selection().layer.unwrap()
}

#[test]
fn moving_into_closed_folder_is_one_undo_step_and_rejects_cycles_atomically() {
    let mut s = session();
    let base = s.selection().layer.unwrap();
    let folder = add(&mut s, NewLayer::Folder);
    s.execute(Command::SetFolderExpanded {
        layer: folder,
        expanded: false,
    })
    .unwrap();
    let before = s.document().clone();
    let count = s.history().entries().len();
    s.execute(Command::MoveLayer {
        layer: base,
        parent: Some(folder),
        index: 0,
    })
    .unwrap();
    assert_eq!(s.history().entries().len(), count + 1);
    assert_eq!(
        s.selected_texture_set().location(base).unwrap().parent,
        Some(folder)
    );
    assert!(
        matches!(&s.selected_texture_set().find(folder).unwrap().content, LayerContent::Group(group) if group.expanded)
    );
    s.execute(Command::Undo).unwrap();
    assert_eq!(s.document(), &before);
    s.execute(Command::Redo).unwrap();
    let nested = add(&mut s, NewLayer::Folder);
    s.execute(Command::MoveLayer {
        layer: nested,
        parent: Some(folder),
        index: 1,
    })
    .unwrap();
    let before = s.document().clone();
    let count = s.history().entries().len();
    assert!(
        s.execute(Command::MoveLayer {
            layer: folder,
            parent: Some(nested),
            index: 0
        })
        .is_err()
    );
    assert_eq!(
        s.document(),
        &before,
        "rejected cycles leave the whole tree intact"
    );
    assert_eq!(s.history().entries().len(), count);
}

#[test]
fn locked_folder_protects_nested_content_masks_and_structure_until_undo() {
    let mut s = session();
    let paint = s.selection().layer.unwrap();
    let fill = add(&mut s, NewLayer::Fill);
    let folder = add(&mut s, NewLayer::Folder);
    s.execute(Command::MoveLayer {
        layer: paint,
        parent: Some(folder),
        index: 0,
    })
    .unwrap();
    s.execute(Command::MoveLayer {
        layer: fill,
        parent: Some(folder),
        index: 1,
    })
    .unwrap();
    s.execute(Command::SetLocked {
        layer: folder,
        locked: true,
    })
    .unwrap();
    assert_eq!(s.selected_texture_set().locked_by(fill), Some(folder));
    let count = s.history().entries().len();
    for command in [
        Command::Rename {
            layer: fill,
            name: "Accidental edit".into(),
        },
        Command::SetVisible {
            layer: paint,
            visible: false,
        },
        Command::SetFillChannel {
            layer: fill,
            channel: ChannelKind::Roughness,
            fill: FillChannel::new(FillSource::uniform(ChannelValue::scalar(0.1))),
        },
        Command::AddMask {
            layer: paint,
            white: true,
        },
        Command::SetLocked {
            layer: fill,
            locked: false,
        },
        Command::DeleteLayer(folder),
        Command::DuplicateLayer(fill),
        Command::MoveLayer {
            layer: paint,
            parent: None,
            index: 0,
        },
    ] {
        assert_eq!(s.execute(command), Err(CommandError::LayerLocked));
    }
    s.execute(Command::SelectLayer(Some(paint))).unwrap();
    assert_eq!(s.begin_stroke(), Err(PaintError::LayerLocked));
    assert_eq!(
        s.execute(Command::AddLayer(NewLayer::Paint)),
        Err(CommandError::LayerLocked)
    );
    assert_eq!(s.history().entries().len(), count);
    s.execute(Command::Undo).unwrap();
    assert_eq!(s.selected_texture_set().locked_by(paint), None);
    s.begin_stroke().unwrap();
    s.cancel_stroke();
    s.execute(Command::Redo).unwrap();
    assert_eq!(s.selected_texture_set().locked_by(paint), Some(folder));
    s.execute(Command::SetLocked {
        layer: folder,
        locked: false,
    })
    .unwrap();
    s.execute(Command::Rename {
        layer: fill,
        name: "Editable".into(),
    })
    .unwrap();
}

#[test]
fn locked_destination_and_locked_descendants_cannot_be_overwritten_or_removed() {
    let mut s = session();
    let paint = s.selection().layer.unwrap();
    let folder = add(&mut s, NewLayer::Folder);
    s.execute(Command::SetLocked {
        layer: folder,
        locked: true,
    })
    .unwrap();
    assert_eq!(
        s.execute(Command::MoveLayer {
            layer: paint,
            parent: Some(folder),
            index: 0
        }),
        Err(CommandError::LayerLocked)
    );
    s.execute(Command::SetLocked {
        layer: folder,
        locked: false,
    })
    .unwrap();
    s.execute(Command::MoveLayer {
        layer: paint,
        parent: Some(folder),
        index: 0,
    })
    .unwrap();
    s.execute(Command::SetLocked {
        layer: paint,
        locked: true,
    })
    .unwrap();
    assert_eq!(
        s.execute(Command::DeleteLayer(folder)),
        Err(CommandError::LayerLocked)
    );
    assert_eq!(
        s.execute(Command::DuplicateLayer(folder)),
        Err(CommandError::LayerLocked)
    );
}

#[test]
fn group_selects_folder_and_deleting_it_recovers_a_selected_child() {
    let mut s = session();
    let paint = s.selection().layer.unwrap();
    s.execute(Command::GroupLayer(paint)).unwrap();
    let folder = s.selection().layer.unwrap();
    assert!(s.selected_layer().unwrap().is_folder());
    s.execute(Command::SelectLayer(Some(paint))).unwrap();
    s.execute(Command::DeleteLayer(folder)).unwrap();
    assert_eq!(s.selection().layer, None);
    assert!(s.selected_layer().is_none());
    s.execute(Command::Undo).unwrap();
    assert!(s.selected_texture_set().find(paint).is_some());
}

#[test]
fn no_op_edits_keep_history_and_redo_and_names_are_normalized() {
    let mut s = session();
    let paint = s.selection().layer.unwrap();
    s.execute(Command::Rename {
        layer: paint,
        name: "  Details  ".into(),
    })
    .unwrap();
    assert_eq!(s.selected_layer().unwrap().name, "Details");
    s.execute(Command::Undo).unwrap();
    let count = s.history().entries().len();
    s.execute(Command::SetVisible {
        layer: paint,
        visible: true,
    })
    .unwrap();
    s.execute(Command::MoveLayer {
        layer: paint,
        parent: None,
        index: 0,
    })
    .unwrap();
    assert_eq!(s.history().entries().len(), count);
    assert!(s.history().can_redo());
    assert!(
        s.execute(Command::Rename {
            layer: paint,
            name: " \n ".into()
        })
        .is_err()
    );
    assert_eq!(s.selected_layer().unwrap().name, "Layer 1");
}

#[test]
fn lock_rename_and_folder_expansion_do_not_invalidate_material_tiles() {
    let mut s = session();
    let paint = s.selection().layer.unwrap();
    let folder = add(&mut s, NewLayer::Folder);
    s.execute(Command::MoveLayer {
        layer: paint,
        parent: Some(folder),
        index: 0,
    })
    .unwrap();
    s.update_composites();
    s.execute(Command::Rename {
        layer: folder,
        name: "Details".into(),
    })
    .unwrap();
    s.execute(Command::SetFolderExpanded {
        layer: folder,
        expanded: false,
    })
    .unwrap();
    s.execute(Command::SetLocked {
        layer: folder,
        locked: true,
    })
    .unwrap();
    assert!(
        s.update_composites().is_empty(),
        "organizational metadata must not trigger expensive recomposites"
    );
}

#[test]
fn modified_cache_tracks_undo_saved_state_and_saving_without_a_new_revision() {
    let mut s = session();
    assert!(!s.is_modified());
    let paint = s.selection().layer.unwrap();
    s.execute(Command::Rename {
        layer: paint,
        name: "Saved".into(),
    })
    .unwrap();
    assert!(s.is_modified());
    s.execute(Command::Undo).unwrap();
    assert!(!s.is_modified(), "undo to the saved document is clean");
    s.execute(Command::Redo).unwrap();
    assert!(s.is_modified());
    let path = std::env::temp_dir().join(format!(
        "subcraft-modified-cache-{}.subcraft",
        LayerId::new()
    ));
    let revision = s.revision();
    s.save(Some(&path)).unwrap();
    assert_eq!(s.revision(), revision);
    assert!(
        !s.is_modified(),
        "saving resets the cache even when revision did not change"
    );
    s.execute(Command::Undo).unwrap();
    assert!(
        s.is_modified(),
        "an older document differs from the newly saved state"
    );
    s.execute(Command::Redo).unwrap();
    assert!(!s.is_modified());
    s.begin_stroke().unwrap();
    let camera = subcraft_app::Camera::perspective(
        glam::Vec3::new(0.0, 0.0, 4.0),
        glam::Vec3::ZERO,
        glam::Vec3::Y,
        40.0,
        0.05,
        100.0,
        [400, 400],
    );
    assert!(
        !s.stroke_to(&camera, glam::Vec2::new(200.0, 200.0), false)
            .unwrap()
            .is_empty()
    );
    assert!(s.is_modified());
    s.cancel_stroke();
    assert!(
        !s.is_modified(),
        "cancelling a live stroke restores the saved snapshot"
    );
    std::fs::remove_file(path).unwrap();
}
