//! Exact Q6_K weight alignment experiment using checked WebGPU shaders.
//! Usage: q6_layout_lab [N=4096] [K=14336] [COPIES=16] [REPEATS=16]
//! LAB_Q6_LAYOUT=aligned|planes; LAB_Q6_SPLITS=1|2.
use fusor_ir::dtype::QFmt;
#[path = "support/quant_layout_common.rs"]
mod common;
use common::*;

fn word_index(block: usize, word: usize, blocks: usize, layout: &str) -> usize {
    match layout {
        "aligned" => block * 64 + word,
        "planes" => match word {
            0..=31 => block * 32 + word,
            32..=47 => blocks * 32 + block * 16 + word - 32,
            48..=51 => blocks * 48 + block * 4 + word - 48,
            52 => blocks * 52 + block,
            _ => unreachable!(),
        },
        "original" => block * 53 + word,
        _ => unreachable!(),
    }
}

fn shader(n: usize, k: usize, layout: &str, splits: usize) -> String {
    let blocks = n * k / 256;
    let mut accessors = String::new();
    for (name, first, width) in [
        ("low", 0, 32),
        ("high", 32, 16),
        ("scale", 48, 4),
        ("d", 52, 1),
    ] {
        let address = match layout {
            "aligned" => format!("block*64u+{first}u+i"),
            "planes" => format!("{}u+block*{width}u+i", blocks * first),
            _ => format!("block*53u+{first}u+i"),
        };
        accessors += &format!(
            "fn {name}(col:u32,bk:u32,i:u32)->u32{{if(col>={n}u){{return 0u;}}let block=col*{}u+bk;return weights[{address}];}}\n",
            k / 256
        );
    }
    let mut body = String::new();
    for part in 0..2 {
        body += &format!(
            "{{let x=input[xbase+{}u];let mx=x*vec4<f32>(1.0,0.00390625,0.0000152587890625,0.000000059604644775390625);let dx=dot(x,vec4<f32>(1.0));let group=half_index*8u+within/4u+{}u;\n",
            part * 16,
            part * 4
        );
        for column in 0..2 {
            body += &format!(
                "let lo{column}=(low{column}>>{}u)&252645135u;let hi{column}=((high{column}>>(within/8u*2u+{}u))&50529027u)<<4u;let bits{column}=lo{column}|hi{column};let quant{column}=vec4<f32>(f32(bits{column}&63u),f32(bits{column}&16128u),f32(bits{column}&4128768u),f32(bits{column}&1056964608u));let sw{column}=scale(col+{column}u,bk,group/4u);let s{column}=f32(bitcast<i32>(((sw{column}>>(group%4u*8u))&255u)<<24u)>>24u);sum{column}=fma(dot(mx,quant{column})-32.0*dx,s{column}*scale{column},sum{column});\n",
                part * 4,
                part * 4
            );
        }
        body += "}\n";
    }
    let close = if splits == 1 {
        format!(
            "let total0=subgroupAdd(sum0);let total1=subgroupAdd(sum1);if(lane==0u && col<{n}u){{output[col]=total0;if(col+1u<{n}u){{output[col+1u]=total1;}}}}"
        )
    } else {
        format!(
            "let total0=subgroupAdd(sum0);let total1=subgroupAdd(sum1);if(lane==0u){{partials[sg*2u]=total0;partials[sg*2u+1u]=total1;}}workgroupBarrier();if(lane==0u && split==0u && col<{n}u){{output[col]=partials[sg*2u]+partials[sg*2u+2u];if(col+1u<{n}u){{output[col+1u]=partials[sg*2u+1u]+partials[sg*2u+3u];}}}}"
        )
    };
    format!(
        r#"
@group(0) @binding(0) var<storage,read> weights:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
var<workgroup> partials:array<f32,8>;
{accessors}
@compute @workgroup_size({threads})
fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) tid:u32){{
 let sg=tid/32u;let split=sg%{splits}u;let col=wg.x*4u+sg/{splits}u*2u;
 let lane=tid%32u;let half_index=lane/16u;let within=lane%16u;
 var sum0=0.0;var sum1=0.0;
 for(var it=0u;it<{passes}u;it++){{
 let bk=it*{splits}u+split;let xbase=bk*64u+half_index*32u+within;
 let low0=low(col,bk,half_index*16u+within);let high0=high(col,bk,half_index*8u+within%8u);let scale0=bitcast<f32>(d(col,bk,0u));
 let low1=low(col+1u,bk,half_index*16u+within);let high1=high(col+1u,bk,half_index*8u+within%8u);let scale1=bitcast<f32>(d(col+1u,bk,0u));
 {body}
 }}
 {close}
}}
"#,
        threads = 64 * splits,
        passes = k / 256 / splits
    )
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let mut number = |default: usize| -> Result<usize> {
        Ok(args
            .next()
            .map(|s| s.parse())
            .transpose()?
            .unwrap_or(default))
    };
    let (n, k, copies, repeats) = (number(4096)?, number(14336)?, number(16)?, number(16)?);
    let layout = std::env::var("LAB_Q6_LAYOUT").unwrap_or_else(|_| "planes".into());
    let splits = std::env::var("LAB_Q6_SPLITS")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(if n > 8192 { 1 } else { 2 });
    if args.next().is_some()
        || !(1..=131072).contains(&n)
        || !(512..=32768).contains(&k)
        || !k.is_multiple_of(512)
        || !(1..=32).contains(&copies)
        || !(1..=128).contains(&repeats)
        || !matches!(layout.as_str(), "planes" | "aligned")
        || !matches!(splits, 1 | 2)
        || n * k / 256 * 256 * copies * 2 > 5 * 1024 * 1024 * 1024
    {
        return Err("invalid dimensions, layout, splits, or working set".into());
    }
    let gpu = pollster::block_on(fusor_gpu::GpuDevice::request(None))?;
    if gpu.caps().subgroup_width() != 32
        || !gpu
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP)
    {
        return Err("requires subgroup32 and WebGPU timestamps".into());
    }
    let (d, q) = (gpu.device(), gpu.queue());
    let blocks = n * k / 256;
    let mut original = vec![0u8; blocks * 210];
    let mut rng = 0x872b_912du32;
    for block in original.as_chunks_mut::<210>().0 {
        for byte in block.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            *byte = rng as u8;
        }
        block[208..210].copy_from_slice(&(0x1800u16 + (rng as u16 & 1023)).to_le_bytes());
    }
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let layouts = ["original", layout.as_str()];
    let mut arms = Vec::new();
    for (arm, layout) in layouts.iter().enumerate() {
        let words = blocks * if *layout == "aligned" { 64 } else { 53 };
        if words as u64 * 4 > d.limits().max_storage_buffer_binding_size {
            return Err("weight binding exceeds adapter limit".into());
        }
        let mut packed = vec![0u8; words * 4];
        for (block, src) in original.as_chunks::<210>().0.iter().enumerate() {
            for word in 0..53 {
                let bytes = if word == 52 {
                    half::f16::from_bits(u16::from_le_bytes(src[208..210].try_into()?))
                        .to_f32()
                        .to_le_bytes()
                } else {
                    src[word * 4..word * 4 + 4].try_into()?
                };
                let at = word_index(block, word, blocks, layout) * 4;
                packed[at..at + 4].copy_from_slice(&bytes);
            }
        }
        if arm == 1
            && let Ok(v) = std::env::var("LAB_Q6_CORRUPT_COLUMN")
        {
            let column: usize = v.parse()?;
            if column >= n {
                return Err("corrupt column out of range".into());
            }
            let at = word_index(column * k / 256, 52, blocks, layout) * 4;
            packed[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        }
        let buffers: Vec<_> = (0..copies)
            .map(|_| {
                let b = buffer(d, packed.len(), usage);
                q.write_buffer(&b, 0, &packed);
                b
            })
            .collect();
        arms.push(buffers);
    }
    let input = buffer(d, k * 4, usage);
    let output = buffer(d, (n + 64) * 4, usage | wgpu::BufferUsages::COPY_SRC);
    let readback = buffer(
        d,
        (n + 64) * 4,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let canary: Vec<_> = (0..n + 64).flat_map(|_| f32::NAN.to_le_bytes()).collect();
    let pipes = [
        pipeline(d, &shader(n, k, "original", splits))?,
        pipeline(d, &shader(n, k, &layout, splits))?,
    ];
    let bindings: Vec<Vec<_>> = pipes
        .iter()
        .zip(&arms)
        .map(|(p, ws)| {
            ws.iter()
                .map(|w| bind(d, p, &[w, &input, &output]))
                .collect()
        })
        .collect();
    let queries = d.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = buffer(
        d,
        256,
        wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
    );
    let stamps = buffer(
        d,
        16,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut samples: [Vec<f64>; 2] = Default::default();
    let mut ratios = Vec::new();
    for round in 0..15 {
        let x: Vec<f32> = (0..k)
            .map(|i| (((i * 17 + i / 7 + round * 13) % 101) as f32 - 50.) / 29.71)
            .collect();
        q.write_buffer(
            &input,
            0,
            &x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        );
        let reference = expected(QFmt::Q6K, &original, &x, n, k, round == 0);
        let mut first: Option<Vec<f32>> = None;
        let mut times = [0.; 2];
        for step in 0..2 {
            let arm = if round % 2 == 0 { step } else { 1 - step };
            q.write_buffer(&output, 0, &canary);
            let mut e = d.create_command_encoder(&Default::default());
            {
                let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Q6 layout replay"),
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &queries,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                pass.set_pipeline(&pipes[arm]);
                for _ in 0..repeats {
                    for bind in &bindings[arm] {
                        pass.set_bind_group(0, bind, &[]);
                        pass.dispatch_workgroups(n.div_ceil(4) as u32, 1, 1);
                    }
                }
            }
            e.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
            q.submit([e.finish()]);
            let values = check(&read(d, &readback)?, &reference, n)
                .map_err(|e| format!("round={round} arm={arm}: {e}"))?;
            if let Some(other) = &first {
                if values[..n] != other[..n] {
                    return Err("layout changed arithmetic output".into());
                }
            } else {
                first = Some(values);
            }
            let mut e = d.create_command_encoder(&Default::default());
            e.resolve_query_set(&queries, 0..2, &resolved, 0);
            e.copy_buffer_to_buffer(&resolved, 0, &stamps, 0, 16);
            q.submit([e.finish()]);
            let bytes = read(d, &stamps)?;
            let start = u64::from_le_bytes(bytes[..8].try_into()?);
            let end = u64::from_le_bytes(bytes[8..].try_into()?);
            if end <= start {
                return Err("invalid timestamps".into());
            }
            times[arm] = (end - start) as f64 * f64::from(q.get_timestamp_period())
                / 1000.
                / (copies * repeats) as f64;
            if round >= 3 {
                samples[arm].push(times[arm]);
            }
        }
        if round >= 3 {
            ratios.push(times[0] / times[1]);
            eprintln!(
                "Q6_LAYOUT_SAMPLE round={round} baseline_us={} candidate_us={} ratio={}",
                times[0],
                times[1],
                times[0] / times[1]
            );
        }
    }
    ratios.sort_by(f64::total_cmp);
    for (arm, values) in samples.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        println!(
            "Q6_LAYOUT n={n} k={k} layout={} splits={splits} copies={copies} repeats={repeats} gpu_us={:.3} p25_us={:.3} p75_us={:.3} paired_speedup={:.4} cpu_all_columns={n} full_arm_match=true runtime_checks=true",
            layouts[arm],
            (values[5] + values[6]) / 2.,
            values[3],
            values[9],
            (ratios[5] + ratios[6]) / 2.
        );
    }
    Ok(())
}
