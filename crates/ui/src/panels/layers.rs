//! Layer stack: tree with visibility, selection, creation/reorder buttons and per-channel blending.

use std::sync::Arc;

use egui::{RichText, Ui};
use subcraft_app::{Command, NewLayer};
use subcraft_core::{ChannelKind, LayerId, TextureSetId};
use subcraft_doc::{BlendMode, Layer, LayerKind, LayerLocation, TextureSet};

use crate::icons;
use crate::state::UiState;
use crate::theme::{self, ACCENT, PAD, TEXT_DIM, TEXT_FAINT};
use crate::widgets::{
    badge, divider, empty_state, fill_width, icon_button, list_row, prop_row, row_text, slider_row,
};

struct LayerRow<'a> {
    depth: usize,
    layer: &'a Layer,
    protected: bool,
    subtree_locked: bool,
}

/// Collect visible rows and lock metadata in one linear tree walk. Repeated ancestor/subtree
/// queries for every row otherwise turn large, nested layer stacks into quadratic UI work.
fn rows<'a>(
    list: &'a [Arc<Layer>],
    depth: usize,
    parent_locked: bool,
    visible: bool,
    out: &mut Vec<LayerRow<'a>>,
) -> bool {
    let mut any_locked = false;
    for l in list.iter().rev() {
        let protected = parent_locked || l.locked;
        let row_index = visible.then_some(out.len());
        if visible {
            out.push(LayerRow {
                depth,
                layer: l,
                protected,
                subtree_locked: l.locked,
            });
        }
        let expanded =
            matches!(&l.content, subcraft_doc::LayerContent::Group(group) if group.expanded);
        let children_locked = rows(l.children(), depth + 1, protected, visible && expanded, out);
        let subtree_locked = l.locked || children_locked;
        if let Some(index) = row_index {
            out[index].subtree_locked = subtree_locked;
        }
        any_locked |= subtree_locked;
    }
    any_locked
}

#[derive(Clone)]
struct InlineRename {
    layer: LayerId,
    text: String,
    focus: bool,
}

#[derive(Clone, Copy)]
struct LayerDrag {
    texture_set: TextureSetId,
    layer: LayerId,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DropPosition {
    Above,
    Below,
    Inside,
}

fn subtree_locked(layer: &Layer) -> bool {
    layer.locked || layer.children().iter().any(|child| subtree_locked(child))
}

/// Convert visual (top-to-bottom) drop positions into bottom-to-top tree indices. The engine's
/// move index is measured after removing the source, which matters for same-parent moves.
fn drop_destination(
    ts: &TextureSet,
    source: LayerId,
    target: LayerId,
    position: DropPosition,
) -> Option<LayerLocation> {
    if source == target || ts.locked_by(source).is_some() || subtree_locked(ts.find(source)?) {
        return None;
    }
    let from = ts.location(source)?;
    let target_loc = ts.location(target)?;
    let (parent, mut index) = match position {
        DropPosition::Inside => {
            let folder = ts.find(target)?;
            if !folder.is_folder() {
                return None;
            }
            (Some(target), folder.children().len())
        }
        DropPosition::Above => (target_loc.parent, target_loc.index + 1),
        DropPosition::Below => (target_loc.parent, target_loc.index),
    };
    if let Some(parent) = parent
        && (ts.is_descendant_or_self(parent, source) || ts.locked_by(parent).is_some())
    {
        return None;
    }
    if from.parent == parent && from.index < index {
        index -= 1;
    }
    let destination = LayerLocation { parent, index };
    (destination != from).then_some(destination)
}

fn row_drop_position(rect: egui::Rect, pointer: egui::Pos2, folder: bool) -> DropPosition {
    let relative = (pointer.y - rect.top()) / rect.height();
    if folder && (0.25..=0.75).contains(&relative) {
        DropPosition::Inside
    } else if relative < 0.5 {
        DropPosition::Above
    } else {
        DropPosition::Below
    }
}

fn draw_drop_marker(ui: &Ui, rect: egui::Rect, position: DropPosition) {
    let stroke = egui::Stroke::new(2.0, ACCENT);
    match position {
        DropPosition::Inside => {
            ui.painter()
                .rect_stroke(rect, 0.0, stroke, egui::StrokeKind::Inside);
        }
        DropPosition::Above => {
            ui.painter()
                .line_segment([rect.left_top(), rect.right_top()], stroke);
        }
        DropPosition::Below => {
            ui.painter()
                .line_segment([rect.left_bottom(), rect.right_bottom()], stroke);
        }
    }
}

pub fn show(state: &mut UiState, ui: &mut Ui) {
    let now = ui.input(|i| i.time);
    let mut cmds = Vec::new();
    let sel = state.session.selection().layer;
    let pad = egui::Margin::symmetric(PAD, 8);
    let rename_id = egui::Id::new("layer_inline_rename");
    let mut rename = ui
        .ctx()
        .data_mut(|data| data.get_temp::<InlineRename>(rename_id));
    if rename.as_ref().is_some_and(|edit| {
        state
            .session
            .selected_texture_set()
            .find(edit.layer)
            .is_none()
    }) {
        rename = None;
    }
    if !ui.ctx().egui_wants_keyboard_input()
        && ui.input(|input| input.key_pressed(egui::Key::F2))
        && let Some(layer) = state.session.selected_layer()
        && state
            .session
            .selected_texture_set()
            .locked_by(layer.id)
            .is_none()
    {
        rename = Some(InlineRename {
            layer: layer.id,
            text: layer.name.clone(),
            focus: true,
        });
    }
    let selected_editable = state.session.selected_layer().is_some_and(|layer| {
        state
            .session
            .selected_texture_set()
            .locked_by(layer.id)
            .is_none()
            && !subtree_locked(layer)
    });

    // Toolbar: create on the left, arrange on the right.
    egui::Frame::new().inner_margin(pad).show(ui, |ui| {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            if icon_button(ui, icons::PAINT_BRUSH, "New paint layer (Ctrl+Shift+N)").clicked() {
                cmds.push(Command::AddLayer(NewLayer::Paint));
            }
            if icon_button(ui, icons::PAINT_BUCKET, "New fill layer (Ctrl+Shift+F)").clicked() {
                cmds.push(Command::AddLayer(NewLayer::Fill));
            }
            if icon_button(ui, icons::FOLDER, "New folder (Ctrl+G)").clicked() {
                cmds.push(Command::AddLayer(NewLayer::Folder));
            }
            if icon_button(ui, icons::IMAGE, "Import texture set as a fill layer…").clicked() {
                import_texture_set_dialog(ui.ctx());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_enabled_ui(selected_editable, |ui| {
                    let id = sel.unwrap_or_default();
                    if icon_button(ui, icons::TRASH, "Delete (Del)").clicked() {
                        cmds.push(Command::DeleteLayer(id));
                    }
                    if icon_button(ui, icons::COPY, "Duplicate (Ctrl+D)").clicked() {
                        cmds.push(Command::DuplicateLayer(id));
                    }
                    let location = state.session.selected_texture_set().location(id);
                    let can_down = location.is_some_and(|loc| loc.index > 0);
                    let can_up = location.is_some_and(|loc| {
                        let siblings = loc
                            .parent
                            .and_then(|parent| state.session.selected_texture_set().find(parent))
                            .map_or(
                                state.session.selected_texture_set().layers.len(),
                                |parent| parent.children().len(),
                            );
                        loc.index + 1 < siblings
                    });
                    if ui
                        .add_enabled_ui(can_down, |ui| {
                            icon_button(ui, icons::ARROW_DOWN, "Move down")
                        })
                        .inner
                        .clicked()
                    {
                        cmds.push(Command::MoveDown(id));
                    }
                    if ui
                        .add_enabled_ui(can_up, |ui| icon_button(ui, icons::ARROW_UP, "Move up"))
                        .inner
                        .clicked()
                    {
                        cmds.push(Command::MoveUp(id));
                    }
                });
            });
        });
        let channels = state.session.channels();
        if !channels.contains(&state.active_channel) {
            state.active_channel = channels.first().copied().unwrap_or(ChannelKind::BaseColor);
        }
        ui.add_space(4.0);
        prop_row(ui, "Channel", |ui| {
            egui::ComboBox::from_id_salt("active_channel")
                .selected_text(state.active_channel.display_name())
                .show_ui(ui, |ui| {
                    for k in &channels {
                        ui.selectable_value(&mut state.active_channel, *k, k.display_name());
                    }
                });
        });
    });
    divider(ui);

    // Blend controls of the selected layer, pinned to the bottom (one compact row when the panel
    // is short, so at least two layer rows stay visible).
    let compact = ui.available_height() < 200.0;
    let active = state.active_channel;
    if let Some(l) = state.session.selected_layer() {
        let id = l.id;
        let folder = l.is_folder();
        let mut b = l.blend_for(active);
        let editable = state.session.selected_texture_set().locked_by(id).is_none();
        egui::Panel::bottom("layer_blend")
            .resizable(false)
            .show_separator_line(false)
            .frame(
                egui::Frame::new()
                    .fill(theme::BG_2)
                    .inner_margin(egui::Margin::symmetric(PAD, 10)),
            )
            .show(ui, |ui| {
                if !editable {
                    ui.disable();
                }
                let blend_combo = |ui: &mut Ui, cmds: &mut Vec<Command>, width: f32| {
                    egui::ComboBox::from_id_salt("blend_mode")
                        .width(width)
                        .selected_text(b.mode.label())
                        .show_ui(ui, |ui| {
                            for m in BlendMode::available(active == ChannelKind::Normal, folder) {
                                if ui.selectable_label(b.mode == m, m.label()).clicked() {
                                    cmds.push(Command::SetBlendMode {
                                        layer: id,
                                        channel: active,
                                        mode: m,
                                    });
                                }
                            }
                        });
                };
                if compact {
                    // Short panel: one row so the layer list keeps room.
                    ui.horizontal(|ui| {
                        blend_combo(ui, &mut cmds, (ui.available_width() - 70.0).max(60.0));
                        let mut pct = b.opacity * 100.0;
                        let r = ui
                            .add(
                                egui::DragValue::new(&mut pct)
                                    .range(0.0..=100.0)
                                    .speed(0.5)
                                    .suffix(" %")
                                    .max_decimals(0),
                            )
                            .on_hover_text("Opacity");
                        if r.changed() {
                            cmds.push(Command::SetBlendOpacity {
                                layer: id,
                                channel: active,
                                opacity: pct / 100.0,
                            });
                        }
                    });
                } else {
                    prop_row(ui, "Blend", |ui| {
                        blend_combo(ui, &mut cmds, ui.available_width())
                    });
                    if slider_row(
                        ui,
                        "Opacity",
                        egui::Slider::new(&mut b.opacity, 0.0..=1.0).max_decimals(2),
                    )
                    .changed()
                    {
                        cmds.push(Command::SetBlendOpacity {
                            layer: id,
                            channel: active,
                            opacity: b.opacity,
                        });
                    }
                }
            });
    }

    let ts = state.session.selected_texture_set();
    let mut save_request: Option<(subcraft_core::LayerId, bool)> = None;
    let mut drop_request: Option<(subcraft_core::LayerId, crate::panels::shelf::ShelfDrag)> = None;
    let mut list = Vec::new();
    rows(&ts.layers, 0, false, true, &mut list);
    let mut finish_rename: Option<(LayerId, String)> = None;
    let mut cancel_rename =
        rename.is_some() && ui.input(|input| input.key_pressed(egui::Key::Escape));
    // Default min height (64 px) would let rows spill over the pinned blend controls when the
    // panel is short; the list must only use the space that is left (it scrolls).
    let rows_height = ui.available_height().max(0.0);
    egui::ScrollArea::vertical()
        .id_salt("layer_rows")
        .auto_shrink([false, false])
        .min_scrolled_height(0.0)
        .max_height(rows_height)
        .show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 1.0;
        fill_width(ui);
        if list.is_empty() {
            empty_state(ui, icons::STACK, "No layers yet. Add one above.");
        }
        for row in &list {
            let l = row.layer;
            let selected = sel == Some(l.id);
            let b = l.blend_for(active);
            let protected = row.protected;
            let editable_subtree = !protected && !row.subtree_locked;
            let resp = list_row(ui, selected, |ui| {
                ui.add_space(row.depth as f32 * 16.0);
                let handle = ui.add(egui::Label::new(RichText::new(icons::STACK).font(theme::font(11.0)).color(TEXT_FAINT)).sense(egui::Sense::drag()));
                if editable_subtree && rename.as_ref().is_none_or(|edit| edit.layer != l.id) {
                    handle.dnd_set_drag_payload(LayerDrag { texture_set: ts.id, layer: l.id });
                    handle.on_hover_text("Drag to reorder. Drop in a folder's center to move inside it.");
                }
                if let subcraft_doc::LayerContent::Group(group) = &l.content {
                    let icon = if group.expanded { icons::CARET_DOWN } else { icons::CARET_RIGHT };
                    if icon_button(ui, icon, if group.expanded { "Collapse folder" } else { "Expand folder" }).clicked() {
                        cmds.push(Command::SetFolderExpanded { layer: l.id, expanded: !group.expanded });
                    }
                }
                let (eye, tip) = if l.visible { (icons::EYE, "Hide") } else { (icons::EYE_SLASH, "Show") };
                if ui.add_enabled_ui(!protected, |ui| icon_button(ui, eye, tip)).inner.clicked() {
                    cmds.push(Command::SetVisible { layer: l.id, visible: !l.visible });
                }
                let icon = match l.kind() {
                    LayerKind::Paint => icons::PAINT_BRUSH,
                    LayerKind::Fill => icons::PAINT_BUCKET,
                    LayerKind::Folder => if matches!(&l.content, subcraft_doc::LayerContent::Group(group) if group.expanded) { icons::FOLDER_OPEN } else { icons::FOLDER },
                };
                let icon_color = if selected { ACCENT } else { TEXT_DIM };
                ui.label(RichText::new(icon).font(theme::font(14.0)).color(icon_color)).on_hover_text(format!("{:?} layer", l.kind()));
                let mut text = row_text(&l.name, selected || l.visible);
                if !l.visible {
                    text = text.color(TEXT_FAINT);
                }
                if let Some(edit) = rename.as_mut().filter(|edit| edit.layer == l.id) {
                    let response = ui.add(egui::TextEdit::singleline(&mut edit.text).id_salt(("inline_name", l.id)).desired_width((ui.available_width() - 82.0).max(50.0)));
                    if edit.focus {
                        response.request_focus();
                        edit.focus = false;
                    } else if response.lost_focus() || (response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Enter))) {
                        finish_rename = Some((l.id, edit.text.clone()));
                    }
                    if response.has_focus() && ui.input(|input| input.key_pressed(egui::Key::Escape)) {
                        cancel_rename = true;
                    }
                } else {
                    ui.add(egui::Label::new(text).truncate().selectable(false));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(format!("{} {:.0}", b.mode.short_label(), b.opacity * 100.0)).font(theme::font(11.0)).color(TEXT_FAINT));
                    if l.mask.is_some() {
                        badge(ui, "mask", ACCENT).on_hover_text("Has a mask");
                    }
                    if protected {
                        badge(ui, "lock", TEXT_DIM).on_hover_text(if l.locked { "Layer locked. Right-click to unlock." } else { "Protected by a locked parent folder." });
                    }
                });
            });
            if resp.clicked() {
                cmds.push(Command::SelectLayer(Some(l.id)));
            }
            if resp.double_clicked() && !protected {
                rename = Some(InlineRename { layer: l.id, text: l.name.clone(), focus: true });
            }
            if let Some(drag) = resp.dnd_hover_payload::<LayerDrag>()
                && drag.texture_set == ts.id
                && let Some(pointer) = ui.input(|input| input.pointer.interact_pos())
            {
                let position = row_drop_position(resp.rect, pointer, l.is_folder());
                if drop_destination(ts, drag.layer, l.id, position).is_some() {
                    draw_drop_marker(ui, resp.rect, position);
                }
            }
            // egui's take_payload clears a payload even when its type differs. Check first so
            // adding layer dragging cannot swallow the existing shelf/material drop path.
            if egui::DragAndDrop::has_payload_of_type::<LayerDrag>(ui.ctx())
                && let Some(drag) = resp.dnd_release_payload::<LayerDrag>()
                && drag.texture_set == ts.id
                && let Some(pointer) = ui.input(|input| input.pointer.interact_pos())
            {
                let position = row_drop_position(resp.rect, pointer, l.is_folder());
                if let Some(destination) = drop_destination(ts, drag.layer, l.id, position) {
                    cmds.push(Command::MoveLayer { layer: drag.layer, parent: destination.parent, index: destination.index });
                    cmds.push(Command::SelectLayer(Some(drag.layer)));
                }
            }
            if let Some(item) = resp.dnd_release_payload::<crate::panels::shelf::ShelfDrag>() {
                drop_request = Some((l.id, *item));
            }
            if resp.dnd_hover_payload::<crate::panels::shelf::ShelfDrag>().is_some() {
                // Materials land above the row; smart masks go onto the row's layer.
                let r = resp.rect;
                ui.painter().rect_stroke(r, 0.0, egui::Stroke::new(1.5, ACCENT), egui::StrokeKind::Inside);
            }
            let has_effects = l.mask.as_ref().is_some_and(|m| !m.effects.is_empty());
            resp.context_menu(|ui| {
                if ui.add_enabled(!protected, egui::Button::new("Rename").shortcut_text("F2")).clicked() {
                    rename = Some(InlineRename { layer: l.id, text: l.name.clone(), focus: true });
                    cmds.push(Command::SelectLayer(Some(l.id)));
                    ui.close();
                }
                if ui.add_enabled(ts.ancestor_locked_by(l.id).is_none(), egui::Button::new(if l.locked { "Unlock layer" } else { "Lock layer" })).clicked() {
                    cmds.push(Command::SetLocked { layer: l.id, locked: !l.locked });
                    ui.close();
                }
                ui.add_enabled_ui(editable_subtree, |ui| {
                    if ui.button(format!("{}  Group in new folder", icons::FOLDER)).clicked() {
                        cmds.push(Command::GroupLayer(l.id));
                        ui.close();
                    }
                    ui.menu_button("Move into folder", |ui| {
                        let mut folders = Vec::new();
                        ts.walk(|folder, depth| {
                            if folder.is_folder() && !ts.is_descendant_or_self(folder.id, l.id) && ts.locked_by(folder.id).is_none() {
                                folders.push((folder.id, format!("{}{}", "  ".repeat(depth), folder.name)));
                            }
                        });
                        if folders.is_empty() { ui.weak("No available folders"); }
                        for (folder, label) in folders {
                            if ui.button(label).clicked() {
                                if let Some(destination) = drop_destination(ts, l.id, folder, DropPosition::Inside) {
                                    cmds.push(Command::MoveLayer { layer: l.id, parent: destination.parent, index: destination.index });
                                }
                                ui.close();
                            }
                        }
                    });
                    if ts.location(l.id).is_some_and(|location| location.parent.is_some()) && ui.button("Move to top level").clicked() {
                        cmds.push(Command::MoveLayer { layer: l.id, parent: None, index: ts.layers.len() });
                        ui.close();
                    }
                });
                if l.mask.is_none() {
                    ui.add_enabled_ui(!protected, |ui| ui.horizontal(|ui| {
                        for (white, label) in [(true, "Add white mask"), (false, "Add black mask")] {
                            if ui.button(label).clicked() {
                                cmds.push(Command::AddMask { layer: l.id, white });
                                ui.close();
                            }
                        }
                    }));
                }
                ui.separator();
                if ui.button(format!("{}  Save as smart material", icons::SAVE)).clicked() {
                    save_request = Some((l.id, false));
                    ui.close();
                }
                if has_effects && ui.button(format!("{}  Save mask as smart mask", icons::MASK)).clicked() {
                    save_request = Some((l.id, true));
                    ui.close();
                }
                ui.separator();
                if ui.add_enabled(editable_subtree, egui::Button::new(format!("{}  Duplicate", icons::COPY)).shortcut_text("Ctrl+D")).clicked() {
                    cmds.push(Command::DuplicateLayer(l.id));
                    ui.close();
                }
                if ui.add_enabled(editable_subtree, egui::Button::new(format!("{}  Delete", icons::TRASH)).shortcut_text("Del")).clicked() {
                    cmds.push(Command::DeleteLayer(l.id));
                    ui.close();
                }
            });
        }
        if egui::DragAndDrop::payload::<LayerDrag>(ui.ctx()).is_some() {
            let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 42.0), egui::Sense::hover());
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "Drop here to move to the top level", theme::font(11.0), TEXT_DIM);
            if response.dnd_hover_payload::<LayerDrag>().is_some() {
                draw_drop_marker(ui, rect, DropPosition::Inside);
            }
            if let Some(drag) = response.dnd_release_payload::<LayerDrag>()
                && drag.texture_set == ts.id && ts.locked_by(drag.layer).is_none()
            {
                let from = ts.location(drag.layer);
                let index = if from.is_some_and(|location| location.parent.is_none()) { ts.layers.len().saturating_sub(1) } else { ts.layers.len() };
                if from != Some(LayerLocation { parent: None, index }) {
                    cmds.push(Command::MoveLayer { layer: drag.layer, parent: None, index });
                }
            }
        }
    });
    if cancel_rename {
        rename = None;
    } else if let Some((layer, name)) = finish_rename {
        if !name.trim().is_empty() {
            cmds.push(Command::Rename { layer, name });
        }
        rename = None;
        state.rename = None;
    }
    ui.ctx().data_mut(|data| {
        if let Some(rename) = rename {
            data.insert_temp(rename_id, rename);
        } else {
            data.remove::<InlineRename>(rename_id);
        }
    });
    state.run(cmds, now);
    if let Some((layer, item)) = drop_request {
        state.run(vec![Command::SelectLayer(Some(layer))], now);
        crate::panels::shelf::apply_shelf_item(state, item, now);
    }
    if let Some((layer, mask)) = save_request {
        save_to_library(state, layer, mask, now);
    }
}

/// Save a layer as a smart material (or its mask as a smart mask) into the user library and
/// show it in the shelf.
pub fn save_to_library(state: &mut UiState, layer: subcraft_core::LayerId, mask: bool, now: f64) {
    let Some(lib) = state.user_library.clone() else {
        state.set_status(
            "No library folder (Documents not found; set SUBCRAFT_LIBRARY)",
            now,
        );
        return;
    };
    let result = if mask {
        state.session.save_smart_mask(layer, &lib).map(|m| {
            let msg = format!("Saved smart mask \"{}\" to the library", m.name);
            state.library.smart_masks.push(m);
            state.shelf.tab = crate::panels::shelf::ShelfTab::SmartMasks;
            msg
        })
    } else {
        state.session.save_smart_material(layer, &lib).map(|m| {
            let msg = format!("Saved smart material \"{}\" to the library", m.name);
            state.library.smart_materials.push(m);
            state.shelf.tab = crate::panels::shelf::ShelfTab::SmartMaterials;
            msg
        })
    };
    match result {
        Ok(msg) => state.set_status(msg, now),
        Err(e) => state.set_status(format!("Save to library: {e}"), now),
    }
}

/// Pick texture files (base color, roughness, normal…) and import them as one fill layer.
pub fn import_texture_set_dialog(ctx: &egui::Context) {
    crate::files::pick(
        ctx,
        crate::files::Purpose::TextureSet,
        crate::files::IMAGES,
        true,
        Some("Import texture set: select the maps (base color, roughness, normal, …)"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stack() -> (TextureSet, [LayerId; 3]) {
        let mut ts = TextureSet::new("Body", 0, 128);
        let ids = ["Bottom", "Middle", "Top"].map(|name| {
            let layer = Layer::new_paint(name, &ts);
            ts.insert_above(None, layer)
        });
        (ts, ids)
    }

    #[test]
    fn drag_reorders_in_both_directions_without_off_by_one() {
        let (ts, [bottom, middle, top]) = stack();
        assert_eq!(
            drop_destination(&ts, bottom, top, DropPosition::Above),
            Some(LayerLocation {
                parent: None,
                index: 2
            })
        );
        assert_eq!(
            drop_destination(&ts, top, bottom, DropPosition::Below),
            Some(LayerLocation {
                parent: None,
                index: 0
            })
        );
        assert_eq!(
            drop_destination(&ts, bottom, middle, DropPosition::Below),
            None,
            "already immediately below"
        );
        assert_eq!(
            drop_destination(&ts, top, middle, DropPosition::Above),
            None,
            "already immediately above"
        );
        assert_eq!(
            drop_destination(&ts, middle, middle, DropPosition::Above),
            None
        );
    }

    #[test]
    fn drag_into_nested_folder_and_out_again_uses_correct_parent() {
        let (mut ts, [bottom, _, top]) = stack();
        let folder = ts.insert_above(None, Layer::new_folder("Folder", &ts.clone()));
        let nested = ts
            .insert(Some(folder), 0, Layer::new_folder("Nested", &ts.clone()))
            .unwrap();
        let destination = drop_destination(&ts, top, nested, DropPosition::Inside).unwrap();
        assert_eq!(
            destination,
            LayerLocation {
                parent: Some(nested),
                index: 0
            }
        );
        ts.move_to(top, destination.parent, destination.index)
            .unwrap();
        assert_eq!(
            drop_destination(&ts, top, bottom, DropPosition::Above),
            Some(LayerLocation {
                parent: None,
                index: 1
            })
        );
        assert_eq!(
            drop_destination(&ts, folder, nested, DropPosition::Inside),
            None,
            "cannot drop folder into its descendant"
        );
        assert_eq!(
            drop_destination(&ts, folder, top, DropPosition::Above),
            None,
            "cannot move parent alongside a nested descendant"
        );
        assert_eq!(
            drop_destination(&ts, bottom, top, DropPosition::Inside),
            None,
            "paint layers cannot contain children"
        );
    }

    #[test]
    fn drag_refuses_locked_sources_ancestors_and_destination_folders() {
        let (mut ts, [bottom, _, top]) = stack();
        let folder = ts.insert_above(None, Layer::new_folder("Folder", &ts.clone()));
        ts.find_mut(folder).unwrap().locked = true;
        assert_eq!(
            drop_destination(&ts, top, folder, DropPosition::Inside),
            None
        );
        ts.find_mut(folder).unwrap().locked = false;
        ts.move_to(top, Some(folder), 0).unwrap();
        ts.find_mut(folder).unwrap().locked = true;
        assert_eq!(
            drop_destination(&ts, top, bottom, DropPosition::Above),
            None
        );
        ts.find_mut(folder).unwrap().locked = false;
        ts.find_mut(top).unwrap().locked = true;
        assert_eq!(
            drop_destination(&ts, folder, bottom, DropPosition::Below),
            None,
            "cannot move a folder containing locked content"
        );
    }

    #[test]
    fn folder_drop_has_distinct_above_inside_and_below_zones() {
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 32.0));
        assert_eq!(
            row_drop_position(rect, egui::pos2(100.0, 3.0), true),
            DropPosition::Above
        );
        assert_eq!(
            row_drop_position(rect, egui::pos2(100.0, 16.0), true),
            DropPosition::Inside
        );
        assert_eq!(
            row_drop_position(rect, egui::pos2(100.0, 29.0), true),
            DropPosition::Below
        );
        assert_eq!(
            row_drop_position(rect, egui::pos2(100.0, 16.0), false),
            DropPosition::Below
        );
    }

    #[test]
    fn collapsed_folders_keep_hidden_child_locks_and_protect_visible_descendants() {
        let (mut ts, [bottom, _, top]) = stack();
        let folder = ts.insert_above(None, Layer::new_folder("Folder", &ts.clone()));
        ts.move_to(top, Some(folder), 0).unwrap();
        ts.find_mut(top).unwrap().locked = true;
        if let subcraft_doc::LayerContent::Group(group) = &mut ts.find_mut(folder).unwrap().content
        {
            group.expanded = false;
        }
        let mut visible = Vec::new();
        rows(&ts.layers, 0, false, true, &mut visible);
        assert!(!visible.iter().any(|row| row.layer.id == top));
        assert!(
            visible
                .iter()
                .find(|row| row.layer.id == folder)
                .unwrap()
                .subtree_locked
        );
        assert!(
            !visible
                .iter()
                .find(|row| row.layer.id == bottom)
                .unwrap()
                .protected
        );
        ts.find_mut(folder).unwrap().locked = true;
        if let subcraft_doc::LayerContent::Group(group) = &mut ts.find_mut(folder).unwrap().content
        {
            group.expanded = true;
        }
        let mut visible = Vec::new();
        rows(&ts.layers, 0, false, true, &mut visible);
        assert!(
            visible
                .iter()
                .find(|row| row.layer.id == top)
                .unwrap()
                .protected
        );
    }
}
