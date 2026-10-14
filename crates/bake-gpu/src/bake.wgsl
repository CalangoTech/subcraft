// Intentionally bounded software BVH traversal on the GPU. No ray-tracing extension required.
struct Params {
    counts: vec4<u32>, // texels, map, ray start, ray batch
    roots: vec4<u32>, // low root/end, high root/end
    options: vec4<u32>, // has high, AO samples, thickness samples, flip normal Y
    distances: vec4<f32>, // diagonal, bias, AO distance, thickness distance
    cage: vec4<f32>, // front, rear, curvature radius
    bounds_min: vec4<f32>,
    bounds_extent: vec4<f32>,
    flags: vec4<u32>, // stage, ID seed, texel offset, cached detail offset
}
struct Sample { position: vec4<f32>, normal: vec4<f32>, tangent: vec4<f32>, geometric: vec4<f32>, data: vec4<u32> }
struct Triangle { a: vec4<f32>, ab: vec4<f32>, ac: vec4<f32>, na: vec4<f32>, nb: vec4<f32>, nc: vec4<f32>, data: vec4<u32> }
struct Node { minimum: vec4<f32>, maximum: vec4<f32>, data: vec4<u32> } // first/count/escape
struct Detail { position: vec4<f32>, normal: vec4<f32>, offset: vec4<f32>, data: vec4<u32> } // tri/root/end/error
struct Hit { distance: f32, u: f32, v: f32, tri: u32, error: u32 }
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> samples: array<Sample>;
@group(0) @binding(2) var<storage, read> triangles: array<Triangle>;
@group(0) @binding(3) var<storage, read> nodes: array<Node>;
@group(0) @binding(4) var<storage, read> tri_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> details: array<Detail>;
@group(0) @binding(6) var<storage, read_write> accum: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> results: array<vec4<f32>>;
const NONE: u32 = 0xffffffffu;
const MAX_VISITS: u32 = 4096u;

fn normalized(v: vec3<f32>, fallback: vec3<f32>) -> vec3<f32> {
    let len = dot(v, v);
    if len < 1e-20 { return fallback; }
    return v * inverseSqrt(len);
}
fn box_hit(n: Node, origin: vec3<f32>, direction: vec3<f32>, max_t: f32) -> bool {
    // Finite reciprocal avoids zero times infinity at coplanar box boundaries.
    let safe_direction = select(vec3<f32>(1e-30), direction, abs(direction) > vec3<f32>(1e-30));
    let inv = 1.0 / safe_direction;
    let t0 = (n.minimum.xyz - origin) * inv;
    let t1 = (n.maximum.xyz - origin) * inv;
    let lower = min(t0, t1);
    let upper = max(t0, t1);
    return max(max(lower.x, lower.y), max(lower.z, 0.0)) <= min(min(upper.x, upper.y), min(upper.z, max_t));
}
fn trace(root: u32, end: u32, origin: vec3<f32>, direction: vec3<f32>, max_t: f32, any_hit: bool) -> Hit {
    var hit = Hit(max_t, 0.0, 0.0, NONE, 0u);
    var node_index = root;
    var visits = 0u;
    loop {
        if node_index >= end { break; }
        if visits >= MAX_VISITS { hit.error = 1u; break; }
        visits += 1u;
        let n = nodes[node_index];
        if !box_hit(n, origin, direction, hit.distance) { node_index = n.data.z; continue; }
        if n.data.y == 0u { node_index += 1u; continue; }
        for (var j = 0u; j < n.data.y; j += 1u) {
            let ti = tri_ids[n.data.x + j];
            let tri = triangles[ti];
            let cross_direction = cross(direction, tri.ac.xyz);
            let det = dot(tri.ab.xyz, cross_direction);
            if abs(det) < 1e-14 { continue; }
            let inverse = 1.0 / det;
            let delta = origin - tri.a.xyz;
            let u = dot(delta, cross_direction) * inverse;
            if u < 0.0 || u > 1.0 { continue; }
            let q = cross(delta, tri.ab.xyz);
            let v = dot(direction, q) * inverse;
            if v < 0.0 || u + v > 1.0 { continue; }
            let distance = dot(tri.ac.xyz, q) * inverse;
            if distance <= 0.0 || distance >= hit.distance { continue; }
            hit.distance = distance;
            hit.u = u;
            hit.v = v;
            hit.tri = ti;
            if any_hit { return hit; }
        }
        node_index = n.data.z;
    }
    return hit;
}
fn cosine_direction(n: vec3<f32>, index: u32, count: u32, rotation: f32) -> vec3<f32> {
    let sign_z = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign_z + n.z);
    let b = n.x * n.y * a;
    let tangent = vec3<f32>(1.0 + sign_z * n.x * n.x * a, sign_z * b, -sign_z * n.x);
    let bitangent = vec3<f32>(b, sign_z + n.y * n.y * a, -n.y);
    let u = (f32(index) + 0.5) / f32(count);
    let v = fract(f32(reverseBits(index)) / 4294967296.0 + rotation);
    let r = sqrt(u);
    let phi = 6.283185307179586 * v;
    return normalized(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + n * sqrt(max(1.0 - u, 0.0)), n);
}
// Closest point expressed as barycentric weights, matching the CPU curvature estimator.
fn closest_bary(point: vec3<f32>, tri: Triangle) -> vec3<f32> {
    let a = tri.a.xyz; let ab = tri.ab.xyz; let ac = tri.ac.xyz;
    let ap = point - a; let d1 = dot(ab, ap); let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 { return vec3<f32>(1.0, 0.0, 0.0); }
    let bp = point - a - ab; let d3 = dot(ab, bp); let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 { return vec3<f32>(0.0, 1.0, 0.0); }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 { let v = d1 / (d1 - d3); return vec3<f32>(1.0 - v, v, 0.0); }
    let cp = point - a - ac; let d5 = dot(ab, cp); let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 { return vec3<f32>(0.0, 0.0, 1.0); }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 { let w = d2 / (d2 - d6); return vec3<f32>(1.0 - w, 0.0, w); }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return vec3<f32>(0.0, 1.0 - w, w);
    }
    let inverse = 1.0 / (va + vb + vc); let v = vb * inverse; let w = vc * inverse;
    return vec3<f32>(1.0 - v - w, v, w);
}
fn curvature(d: Detail) -> vec2<f32> {
    let radius = p.cage.z;
    if radius <= 0.0 { return vec2<f32>(0.5, 0.0); }
    var sum = 0.0; var weight = 0.0; var nearest = 0.0;
    var ni = d.data.y; var visits = 0u;
    loop {
        if ni >= d.data.z { break; }
        if visits >= MAX_VISITS { return vec2<f32>(0.5, 1.0); }
        visits += 1u;
        let n = nodes[ni];
        let delta = clamp(d.position.xyz, n.minimum.xyz, n.maximum.xyz) - d.position.xyz;
        if dot(delta, delta) > radius * radius { ni = n.data.z; continue; }
        if n.data.y == 0u { ni += 1u; continue; }
        for (var j = 0u; j < n.data.y; j += 1u) {
            let tri = triangles[tri_ids[n.data.x + j]];
            let bary = closest_bary(d.position.xyz, tri);
            let point = tri.a.xyz + tri.ab.xyz * bary.y + tri.ac.xyz * bary.z;
            var direction = point - d.position.xyz;
            var distance = length(direction);
            if distance >= radius { continue; }
            if distance <= radius * 1e-3 {
                direction = tri.a.xyz + (tri.ab.xyz + tri.ac.xyz) / 3.0 - d.position.xyz;
                if dot(direction, direction) < 1e-12 { continue; }
                direction = normalize(direction) * radius * 1e-3;
                distance = radius * 1e-3;
            }
            let normal_q = normalized(tri.na.xyz * bary.x + tri.nb.xyz * bary.y + tri.nc.xyz * bary.z, vec3<f32>(0.0));
            let difference = normal_q - d.normal.xyz;
            if dot(difference, difference) < 1e-6 { continue; }
            let w = 1.0 - distance / radius;
            sum += w * dot(difference, direction / distance); weight += w; nearest = max(nearest, w);
        }
        ni = n.data.z;
    }
    if weight <= 0.0 { return vec2<f32>(0.5, 0.0); }
    return vec2<f32>(0.5 + 0.5 * clamp(sum / weight, -1.0, 1.0) * nearest, 0.0);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let ray_stage = p.flags.x == 2u;
    let slot = invocation.x % 4u;
    let i = select(invocation.x, invocation.x / 4u, ray_stage) + p.flags.z;
    if i >= p.counts.x || invocation.x >= select(128u, 512u, ray_stage) { return; }
    let detail_index = i + p.flags.w;
    if p.flags.x == 3u {
        accum[i * 4u] = vec4<f32>(0.0); accum[i * 4u + 1u] = vec4<f32>(0.0);
        accum[i * 4u + 2u] = vec4<f32>(0.0); accum[i * 4u + 3u] = vec4<f32>(0.0);
        return;
    }
    let s = samples[i];
    if p.flags.x == 0u {
        var d = Detail(s.position, vec4<f32>(s.normal.xyz, 0.0), s.geometric, vec4<u32>(s.data.y, p.roots.x, p.roots.y, 0u));
        if p.options.x != 0u {
            let n = normalized(s.normal.xyz, s.geometric.xyz);
            let hit = trace(p.roots.z, p.roots.w, s.position.xyz + n * p.cage.x, -n, p.cage.x + p.cage.y, false);
            d.data.w = hit.error;
            if hit.tri != NONE {
                let tri = triangles[hit.tri];
                let bary = vec3<f32>(1.0 - hit.u - hit.v, hit.u, hit.v);
                var normal = normalized(tri.na.xyz * bary.x + tri.nb.xyz * bary.y + tri.nc.xyz * bary.z, n);
                var geometric = normalized(cross(tri.ab.xyz, tri.ac.xyz), n);
                if dot(normal, n) < 0.0 { normal = -normal; }
                if dot(geometric, n) < 0.0 { geometric = -geometric; }
                d.position = vec4<f32>(tri.a.xyz + tri.ab.xyz * hit.u + tri.ac.xyz * hit.v, 0.0);
                d.normal = vec4<f32>(normal, 0.0); d.offset = vec4<f32>(geometric, 0.0);
                d.data = vec4<u32>(hit.tri, p.roots.z, p.roots.w, hit.error);
            }
        }
        details[detail_index] = d;
        accum[i * 4u] = vec4<f32>(0.0); accum[i * 4u + 1u] = vec4<f32>(0.0);
        accum[i * 4u + 2u] = vec4<f32>(0.0); accum[i * 4u + 3u] = vec4<f32>(0.0);
        return;
    }
    let d = details[detail_index];
    if p.flags.x == 2u {
        var value = accum[i * 4u + slot];
        let ao = p.counts.y == 4u;
        let count = select(p.options.z, p.options.y, ao);
        let n = select(-d.normal.xyz, d.normal.xyz, ao);
        let origin = d.position.xyz + d.offset.xyz * select(-p.distances.y, p.distances.y, ao);
        let max_t = select(p.distances.w, p.distances.z, ao);
        let rotation = select(s.normal.w, s.position.w, ao);
        let ray = p.counts.z + slot;
        if ray >= count { return; }
        let direction = cosine_direction(n, ray, count, rotation);
        if ao && dot(direction, d.offset.xyz) <= 0.0 { return; }
        let hit = trace(d.data.y, d.data.z, origin, direction, max_t, ao);
        value.z = max(value.z, f32(hit.error));
        value.x += select(hit.distance, select(0.0, 1.0, hit.tri != NONE), ao);
        value.y += 1.0;
        accum[i * 4u + slot] = value; return;
    }
    var value = vec3<f32>(0.0); var error = f32(d.data.w);
    let totals = accum[i * 4u] + accum[i * 4u + 1u] + accum[i * 4u + 2u] + accum[i * 4u + 3u];
    error = max(error, totals.z);
    switch p.counts.y {
        case 0u: {
            let n = normalized(s.normal.xyz, s.geometric.xyz);
            let tangent = normalized(s.tangent.xyz - n * dot(n, s.tangent.xyz), vec3<f32>(1.0, 0.0, 0.0));
            let bitangent = cross(n, tangent) * select(1.0, -1.0, s.tangent.w < 0.0);
            var normal = normalized(vec3<f32>(dot(d.normal.xyz, tangent), dot(d.normal.xyz, bitangent), dot(d.normal.xyz, n)), vec3<f32>(0.0, 0.0, 1.0));
            if p.options.w != 0u { normal.y = -normal.y; }
            value = normal * 0.5 + 0.5;
        }
        case 1u: { value = d.normal.xyz * 0.5 + 0.5; }
        case 2u: { value = clamp((s.position.xyz - p.bounds_min.xyz) / p.bounds_extent.xyz, vec3<f32>(0.0), vec3<f32>(1.0)); }
        case 3u: {
            let hue = fract(f32(triangles[d.data.x].data.x + p.flags.y) * 0.618034);
            let chroma = 0.9 * 0.65;
            let h = hue * 6.0; let x = chroma * (1.0 - abs(h % 2.0 - 1.0));
            var rgb = vec3<f32>(chroma, 0.0, x);
            if h < 1.0 { rgb = vec3<f32>(chroma, x, 0.0); }
            else if h < 2.0 { rgb = vec3<f32>(x, chroma, 0.0); }
            else if h < 3.0 { rgb = vec3<f32>(0.0, chroma, x); }
            else if h < 4.0 { rgb = vec3<f32>(0.0, x, chroma); }
            else if h < 5.0 { rgb = vec3<f32>(x, 0.0, chroma); }
            value = rgb + (0.9 - chroma);
        }
        case 4u: { value = vec3<f32>(1.0 - totals.x / max(totals.y, 1.0)); }
        case 5u: { let c = curvature(d); value = vec3<f32>(c.x); error = max(error, c.y); }
        case 6u: { value = vec3<f32>(1.0 - totals.x / max(totals.y, 1.0) / p.distances.w); }
        default: {}
    }
    results[i] = vec4<f32>(value, error);
}
