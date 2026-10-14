//! Numerical checks for non-destructive mask filters, including tile and texture borders.

use std::sync::Arc;

use rayon::prelude::*;

use subcraft_compositor::{
    CompositeCache, EvalContext, TileContext, composite_full,
    mask::{SharedMaskCache, eval_mask, eval_mask_cached},
};
use subcraft_core::{ChannelKind, ChannelValue};
use subcraft_doc::{
    FillChannel, FillSource, Layer, LayerContent, Mask, MaskEffect, MaskEffectKind, ResourceStore,
    TextureSet,
};
use subcraft_materials::{MeshMaps, SampleContext, TextureSetGeometry};
use subcraft_mesh::primitives;

fn effect(mask: &mut Mask, kind: MaskEffectKind) {
    mask.effects.push(Arc::new(MaskEffect::new(kind)));
}

fn evaluate(mask: &Mask) -> Vec<f32> {
    evaluate_batch(mask, None, false).0
}

fn evaluate_batch(mask: &Mask, cache_limit: Option<usize>, parallel: bool) -> (Vec<f32>, usize) {
    let size = mask.base.width();
    let geometry = TextureSetGeometry::new(Arc::new(primitives::plane(2.0, 1)), 0, size);
    let resources = ResourceStore::default();
    let maps = MeshMaps::new();
    let cache = cache_limit.map(SharedMaskCache::with_byte_limit);
    let evaluate_one = |tile| {
        let geo = geometry.tile(tile);
        let tc = TileContext {
            geo: &geo,
            set_geometry: &geometry,
            sample: SampleContext {
                resources: &resources,
                mesh_maps: &maps,
            },
        };
        let values = match &cache {
            Some(cache) => eval_mask_cached(Some(mask), &tc, cache),
            None => eval_mask(Some(mask), &tc),
        }
        .unwrap();
        (tile, values)
    };
    let tiles: Vec<_> = (0..geometry.tile_count()).collect();
    let results: Vec<_> = if parallel {
        tiles.par_iter().map(|&tile| evaluate_one(tile)).collect()
    } else {
        tiles.iter().map(|&tile| evaluate_one(tile)).collect()
    };
    let mut result = vec![0.0; (size * size) as usize];
    for (tile, values) in results {
        let geo = geometry.tile(tile);
        for (i, &value) in values.iter().enumerate() {
            let (x, y) = geo.texel_xy(i);
            if x < size && y < size {
                result[(y * size + x) as usize] = value;
            }
        }
    }
    (
        result,
        cache.as_ref().map_or(0, SharedMaskCache::retained_bytes),
    )
}

#[test]
fn shared_prefix_cache_matches_uncached_with_parallel_workers_and_bounded_fallback() {
    let mut mask = Mask::new(130, false);
    for y in 0..130 {
        for x in 0..130 {
            mask.base
                .set(x, y, [((x * 23 + y * 17 + x * y) % 101) as f32 / 100.0; 4]);
        }
    }
    effect(&mut mask, MaskEffectKind::Dilate { radius: 4 });
    effect(&mut mask, MaskEffectKind::Blur { radius: 3 });
    effect(
        &mut mask,
        MaskEffectKind::Contrast {
            amount: 1.6,
            pivot: 0.4,
        },
    );
    effect(&mut mask, MaskEffectKind::Erode { radius: 2 });
    let expected = evaluate(&mask);
    let workers = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for budget in [0, 32 * 1024, 64 * 1024 * 1024] {
        let (serial, retained) = evaluate_batch(&mask, Some(budget), false);
        assert_eq!(serial, expected, "serial shared cache, budget {budget}");
        assert!(retained <= budget, "retained {retained} exceeds {budget}");
        let (parallel, retained) = workers.install(|| evaluate_batch(&mask, Some(budget), true));
        assert_eq!(parallel, expected, "parallel shared cache, budget {budget}");
        assert!(retained <= budget, "parallel reservations remain bounded");
    }
    mask.base.set(64, 64, [0.0; 4]);
    let (fresh, _) = evaluate_batch(&mask, Some(64 * 1024 * 1024), true);
    assert_eq!(
        fresh,
        evaluate(&mask),
        "a new batch has no stale prefix after an edit"
    );
}

#[test]
fn blur_impulse_crosses_four_tiles_and_preserves_mass() {
    let mut mask = Mask::new(128, false);
    mask.base.set(63, 63, [1.0; 4]);
    effect(&mut mask, MaskEffectKind::Blur { radius: 1 });
    let original = mask.clone();
    let result = evaluate(&mask);
    for y in 0..128 {
        for x in 0..128 {
            let expected = if (62..=64).contains(&x) && (62..=64).contains(&y) {
                1.0 / 9.0
            } else {
                0.0
            };
            assert!((result[y * 128 + x] - expected).abs() < 1e-6, "({x}, {y})");
        }
    }
    assert!((result.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    assert_eq!(
        mask, original,
        "evaluation must not change the editable raster or effects"
    );
}

#[test]
fn morphology_grows_and_shrinks_across_tile_boundaries() {
    let mut grow = Mask::new(128, false);
    grow.base.set(63, 63, [1.0; 4]);
    effect(&mut grow, MaskEffectKind::Dilate { radius: 2 });
    let grown = evaluate(&grow);
    assert_eq!(grown.iter().filter(|&&v| v == 1.0).count(), 25);
    assert_eq!(grown[65 * 128 + 65], 1.0);
    assert_eq!(grown[66 * 128 + 65], 0.0);
    effect(&mut grow, MaskEffectKind::Erode { radius: 2 });
    let closed = evaluate(&grow);
    assert_eq!(closed.iter().filter(|&&v| v == 1.0).count(), 1);
    assert_eq!(closed[63 * 128 + 63], 1.0);

    let mut shrink = Mask::new(128, true);
    shrink.base.set(63, 63, [0.0; 4]);
    effect(&mut shrink, MaskEffectKind::Erode { radius: 1 });
    assert_eq!(evaluate(&shrink).iter().filter(|&&v| v == 0.0).count(), 9);
}

#[test]
fn borders_replicate_edge_texels_and_neutral_filters_are_identity() {
    let mut mask = Mask::new(128, true);
    effect(&mut mask, MaskEffectKind::Blur { radius: 32 });
    effect(&mut mask, MaskEffectKind::Erode { radius: 32 });
    assert!(evaluate(&mask).iter().all(|&v| v == 1.0));
    mask.effects.clear();
    mask.base.set(0, 0, [0.25; 4]);
    effect(&mut mask, MaskEffectKind::Blur { radius: 0 });
    effect(&mut mask, MaskEffectKind::Dilate { radius: 0 });
    effect(&mut mask, MaskEffectKind::Erode { radius: 0 });
    effect(
        &mut mask,
        MaskEffectKind::Contrast {
            amount: 1.0,
            pivot: 0.5,
        },
    );
    assert_eq!(evaluate(&mask)[0], 0.25);
}

#[test]
fn filters_respect_opacity_visibility_and_contrast_pivot() {
    let mut mask = Mask::new(128, false);
    mask.base.set(63, 63, [1.0; 4]);
    let mut blur = MaskEffect::new(MaskEffectKind::Blur { radius: 1 });
    blur.opacity = 0.5;
    mask.effects.push(Arc::new(blur));
    let result = evaluate(&mask);
    assert!((result[63 * 128 + 63] - 5.0 / 9.0).abs() < 1e-6);
    assert!((result[64 * 128 + 64] - 1.0 / 18.0).abs() < 1e-6);
    Arc::make_mut(&mut mask.effects[0]).visible = false;
    assert_eq!(evaluate(&mask)[64 * 128 + 64], 0.0);
    mask.effects.clear();
    mask.base.set(0, 0, [0.25; 4]);
    mask.base.set(1, 0, [0.5; 4]);
    mask.base.set(2, 0, [0.75; 4]);
    effect(
        &mut mask,
        MaskEffectKind::Contrast {
            amount: 2.0,
            pivot: 0.5,
        },
    );
    assert_eq!(&evaluate(&mask)[..3], &[0.0, 0.5, 1.0]);
}

#[test]
fn stacked_filters_match_independent_dense_reference_including_partial_tiles() {
    const SIZE: usize = 130;
    let mut mask = Mask::new(SIZE as u32, false);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let v = ((x * 13 + y * 37 + x * y * 3) % 101) as f32 / 100.0;
            mask.base.set(x as u32, y as u32, [v; 4]);
        }
    }
    let mut expected: Vec<_> = (0..SIZE * SIZE)
        .map(|i| mask.base.get((i % SIZE) as u32, (i / SIZE) as u32)[0])
        .collect();
    let filters = [
        MaskEffectKind::Blur { radius: 2 },
        MaskEffectKind::Dilate { radius: 3 },
        MaskEffectKind::Erode { radius: 1 },
        MaskEffectKind::Contrast {
            amount: 1.7,
            pivot: 0.3,
        },
        MaskEffectKind::Blur { radius: 1 },
    ];
    for kind in filters {
        let mut next = vec![0.0; expected.len()];
        let r = kind.spatial_radius() as i32;
        for y in 0..SIZE {
            for x in 0..SIZE {
                let mut values = Vec::new();
                for dy in -r..=r {
                    for dx in -r..=r {
                        let sx = (x as i32 + dx).clamp(0, SIZE as i32 - 1) as usize;
                        let sy = (y as i32 + dy).clamp(0, SIZE as i32 - 1) as usize;
                        values.push(expected[sy * SIZE + sx]);
                    }
                }
                next[y * SIZE + x] = match kind {
                    MaskEffectKind::Blur { .. } => values.iter().sum::<f32>() / values.len() as f32,
                    MaskEffectKind::Dilate { .. } => {
                        values.into_iter().fold(f32::NEG_INFINITY, f32::max)
                    }
                    MaskEffectKind::Erode { .. } => {
                        values.into_iter().fold(f32::INFINITY, f32::min)
                    }
                    MaskEffectKind::Contrast { amount, pivot } => {
                        ((expected[y * SIZE + x] - pivot) * amount + pivot).clamp(0.0, 1.0)
                    }
                    _ => unreachable!(),
                };
            }
        }
        expected = next;
        effect(&mut mask, kind);
    }
    for (i, (actual, expected)) in evaluate(&mask).iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 2e-5,
            "pixel {i}: {actual} vs {expected}"
        );
    }
}

#[test]
fn filtered_mask_edits_refresh_neighbour_tiles_and_match_full_composite() {
    let mut ts = TextureSet::new("Test", 0, 256);
    ts.channels.retain(|c| c.kind == ChannelKind::Roughness);
    let mut layer = Layer::new_fill("Masked", &ts);
    let LayerContent::Fill(fill) = &mut layer.content else {
        unreachable!()
    };
    fill.channels.insert(
        ChannelKind::Roughness,
        FillChannel::new(FillSource::uniform(ChannelValue::scalar(1.0))),
    );
    let mut mask = Mask::new(256, false);
    effect(&mut mask, MaskEffectKind::Blur { radius: 1 });
    layer.mask = Some(mask);
    let id = ts.insert_above(None, layer);
    let geometry = TextureSetGeometry::new(Arc::new(primitives::plane(2.0, 1)), 0, 256);
    let resources = ResourceStore::default();
    let maps = MeshMaps::new();
    let ctx = EvalContext {
        geometry: &geometry,
        resources: &resources,
        mesh_maps: &maps,
    };
    let mut cache = CompositeCache::new();
    cache.update(&ts, &ctx);
    ts.find_mut(id)
        .unwrap()
        .mask
        .as_mut()
        .unwrap()
        .base
        .set(63, 63, [1.0; 4]);
    assert_eq!(cache.update(&ts, &ctx), vec![0, 1, 4, 5]);
    let full = composite_full(&ts, &ctx);
    assert_eq!(
        cache.composite().unwrap(),
        &full,
        "incremental results include filtered pixels in neighbouring tiles"
    );
    assert!(full.channel(ChannelKind::Roughness).unwrap().get(64, 64)[0] > 0.37);
}

#[test]
fn dirty_halo_uses_sum_of_radii_and_clamps_extreme_values() {
    let mut ts = TextureSet::new("Test", 0, 256);
    ts.channels.clear(); // This check only needs invalidation, not image evaluation.
    let mut layer = Layer::new_fill("Masked", &ts);
    let mut mask = Mask::new(256, false);
    effect(&mut mask, MaskEffectKind::Dilate { radius: 32 });
    effect(&mut mask, MaskEffectKind::Dilate { radius: u32::MAX }); // capped to 32
    effect(&mut mask, MaskEffectKind::Blur { radius: 1 });
    layer.mask = Some(mask);
    let id = ts.insert_above(None, layer);
    let geometry = TextureSetGeometry::new(Arc::new(primitives::plane(2.0, 1)), 0, 256);
    let resources = ResourceStore::default();
    let maps = MeshMaps::new();
    let ctx = EvalContext {
        geometry: &geometry,
        resources: &resources,
        mesh_maps: &maps,
    };
    let mut cache = CompositeCache::new();
    cache.update(&ts, &ctx);
    ts.find_mut(id)
        .unwrap()
        .mask
        .as_mut()
        .unwrap()
        .base
        .set(63, 63, [1.0; 4]);
    assert_eq!(cache.update(&ts, &ctx), vec![0, 1, 2, 4, 5, 6, 8, 9, 10]);
}

#[test]
fn passthrough_folder_partial_opacity_preserves_unit_normal() {
    let mut ts = TextureSet::new("Normal", 0, 128);
    ts.channels.retain(|c| c.kind == ChannelKind::Normal);
    let mut folder = Layer::new_folder("Details", &ts);
    let blend = folder.blend.get_mut(&ChannelKind::Normal).unwrap();
    blend.mode = subcraft_doc::BlendMode::Passthrough;
    blend.opacity = 0.5;
    let mut child = Layer::new_fill("Tilted Normal", &ts);
    let LayerContent::Fill(fill) = &mut child.content else {
        unreachable!()
    };
    fill.channels.insert(
        ChannelKind::Normal,
        FillChannel::new(FillSource::uniform(ChannelValue::rgb(1.0, 0.5, 0.5))),
    );
    folder.children_mut().unwrap().push(Arc::new(child));
    ts.insert_above(None, folder);
    let geometry = TextureSetGeometry::new(Arc::new(primitives::plane(2.0, 1)), 0, 128);
    let resources = ResourceStore::default();
    let maps = MeshMaps::new();
    let ctx = EvalContext {
        geometry: &geometry,
        resources: &resources,
        mesh_maps: &maps,
    };
    let image = composite_full(&ts, &ctx);
    let pixel = image.channel(ChannelKind::Normal).unwrap().get(63, 63);
    let normal = glam::Vec3::new(pixel[0], pixel[1], pixel[2]) * 2.0 - glam::Vec3::ONE;
    assert!(
        (normal.length() - 1.0).abs() < 2e-3,
        "stored normal {normal:?} remains unit length"
    );
    let expected = glam::Vec3::new(1.0, 0.0, 1.0).normalize();
    assert!(
        normal.normalize().dot(expected) > 0.99999,
        "opacity interpolates toward the base normal"
    );
}

#[test]
#[ignore = "Manual timing, not a machine-dependent acceptance threshold"]
fn spatial_filter_benchmark() {
    // Run with --ignored --nocapture. Measures 1024² of a generated mask and two real filters.
    let mut mask = Mask::new(1024, false);
    let mut generator = subcraft_doc::Generator::new(subcraft_doc::GeneratorKind::Directional);
    generator.noise_amount = 0.35;
    effect(&mut mask, MaskEffectKind::Generator(generator));
    effect(&mut mask, MaskEffectKind::Dilate { radius: 3 });
    effect(&mut mask, MaskEffectKind::Blur { radius: 2 });
    let start = std::time::Instant::now();
    let values = evaluate(&mask);
    let uncached = start.elapsed();
    let start = std::time::Instant::now();
    let (cached, bytes) = evaluate_batch(&mask, Some(64 * 1024 * 1024), false);
    let serial = start.elapsed();
    assert_eq!(cached, values);
    let start = std::time::Instant::now();
    let (parallel, parallel_bytes) = evaluate_batch(&mask, Some(64 * 1024 * 1024), true);
    let threaded = start.elapsed();
    assert_eq!(parallel, values);
    eprintln!(
        "1024² generator + dilate3 + blur2, including geometry: uncached sequential {uncached:?}; shared sequential {serial:?} ({:.2}x); shared parallel {threaded:?} ({:.2}x); retained {bytes}/{parallel_bytes} bytes",
        uncached.as_secs_f64() / serial.as_secs_f64(),
        uncached.as_secs_f64() / threaded.as_secs_f64(),
    );
    assert!(
        values
            .iter()
            .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    );
}
