//! Cancellable jump flooding; shares the baker's limited Rayon worker pool.
use crate::{BakeError, Progress};
use rayon::prelude::*;

pub fn dilate_cancellable(
    width: usize,
    height: usize,
    comps: usize,
    data: &mut [f32],
    covered: &[bool],
    progress: &Progress,
) -> Result<(), BakeError> {
    if progress.is_cancelled() {
        return Err(BakeError::Cancelled);
    }
    let n = width * height;
    if covered.iter().all(|&c| c) || !covered.iter().any(|&c| c) {
        return Ok(());
    }
    const NONE: u32 = u32::MAX;
    let mut seed: Vec<u32> = (0..n)
        .map(|i| if covered[i] { i as u32 } else { NONE })
        .collect();
    let mut next = vec![NONE; n];
    let dist2 = |a: usize, b: u32| -> u64 {
        let (ax, ay) = ((a % width) as i64, (a / width) as i64);
        let (bx, by) = ((b as usize % width) as i64, (b as usize / width) as i64);
        ((ax - bx).pow(2) + (ay - by).pow(2)) as u64
    };
    let mut step = (width.max(height).next_power_of_two() / 2).max(1);
    loop {
        next.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
            if progress.is_cancelled() {
                return;
            }
            for (x, output) in row.iter_mut().enumerate() {
                let i = y * width + x;
                let mut best = seed[i];
                let mut best_d = if best == NONE {
                    u64::MAX
                } else {
                    dist2(i, best)
                };
                for dy in [-1i64, 0, 1] {
                    for dx in [-1i64, 0, 1] {
                        let nx = x as i64 + dx * step as i64;
                        let ny = y as i64 + dy * step as i64;
                        if nx < 0 || ny < 0 || nx >= width as i64 || ny >= height as i64 {
                            continue;
                        }
                        let s = seed[ny as usize * width + nx as usize];
                        if s != NONE {
                            let d = dist2(i, s);
                            if d < best_d {
                                best_d = d;
                                best = s;
                            }
                        }
                    }
                }
                *output = best;
            }
        });
        if progress.is_cancelled() {
            return Err(BakeError::Cancelled);
        }
        std::mem::swap(&mut seed, &mut next);
        if step == 1 {
            break;
        }
        step /= 2;
    }
    // Seeds identify covered pixels, so copying never modifies another pixel's source.
    for i in 0..n {
        if i % width == 0 && progress.is_cancelled() {
            return Err(BakeError::Cancelled);
        }
        if !covered[i] && seed[i] != NONE {
            let source = seed[i] as usize;
            for c in 0..comps {
                data[i * comps + c] = data[source * comps + c];
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padding_matches_shared_dilation_and_observes_cancel() {
        let covered: Vec<bool> = (0..35).map(|i| i == 3 || i == 17 || i == 30).collect();
        let mut reference: Vec<f32> = (0..35).map(|i| i as f32 / 35.0).collect();
        let mut output = reference.clone();
        subcraft_image::dilate::dilate_infinite(7, 5, 1, &mut reference, &covered);
        dilate_cancellable(7, 5, 1, &mut output, &covered, &Progress::new()).unwrap();
        assert_eq!(output, reference);
        let progress = Progress::new();
        progress.cancel();
        assert_eq!(
            dilate_cancellable(7, 5, 1, &mut output, &covered, &progress),
            Err(BakeError::Cancelled)
        );
    }
}
