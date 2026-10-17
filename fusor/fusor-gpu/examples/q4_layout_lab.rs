//! WebGPU Q4_K column interleaving probe; all arithmetic remains F32.
//! Usage: q4_layout_lab [N=28672] [K=4096] [COPIES=8] [REPEATS=16]
//! Compares native and losslessly interleaved words on identical changing inputs
//! in ABBA order. Reports kernel spans, never model throughput. Runtime shader
//! checks remain enabled. First input checks every column against CPU dequant.
//! LAB_LAYOUT_PRESCALED=1 expands each block to 8 exact F32 scale/offset pairs
//! and 128 unchanged quant bytes, for external-shader scale-decoding probes.
use fusor_ir::dtype::QFmt;
#[path = "support/quant_layout_common.rs"]
mod common;
use common::*;

// Keep the production mapping (four K blocks and two output columns per
// subgroup), but transpose words within each four-block packet. This isolates
// memory ordering from the different column-parallel algorithm below.
fn block_shader(n: usize, k: usize, interleaved: bool) -> String {
    let input_mode = std::env::var("LAB_LAYOUT_INPUT").unwrap_or_default();
    let uniform = interleaved && input_mode == "uniform";
    let shared = interleaved && input_mode == "shared";
    let interleaved = interleaved && input_mode.is_empty();
    let subgroups: usize = std::env::var("LAB_LAYOUT_SUBGROUPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);
    let threads = subgroups * 32;
    let address = if interleaved {
        format!("((col*{}u+bk/4u)*36u+word)*4u+bk%4u", k / 1024)
    } else {
        format!("(col*{}u+bk)*36u+word", k / 256)
    };
    let mapping = if interleaved {
        "let block_lane=lane%4u;let pair=lane/16u;let run=(lane/4u)%4u;"
    } else {
        "let block_lane=lane/8u;let pair=(lane/4u)%2u;let run=lane%4u;"
    };
    let mut body = String::new();
    for group in 0..4 {
        body += &format!(
            "{{let x0=input[xbase+{}u];let x1=input[xbase+{}u];let shift=pair*16u+{}u;\n",
            [0, 8, 32, 40][group],
            [1, 9, 33, 41][group],
            group % 2 * 8
        );
        let nibble = group % 2 * 4;
        let factor = 2f32.powi(-(nibble as i32));
        body += &format!(
            "let factors=vec4<f32>({factor},{},{},{});let mx0=x0*factors;let mx1=x1*factors;let dx=dot(x0+x1,vec4<f32>(1.0));\n",
            factor / 256.,
            factor / 65536.,
            factor / 16777216.
        );
        for column in 0..2 {
            body += &format!(
                "let s{column}=(ps{column}>>shift)&255u;let m{column}=(pm{column}>>shift)&255u;let e{column}=(pe{column}>>shift)&255u;\n"
            );
            if group < 2 {
                body += &format!("let gs{column}=s{column}&63u;let gm{column}=m{column}&63u;\n");
            } else {
                body += &format!(
                    "let gs{column}=(e{column}&15u)|((s{column}&192u)>>2u);let gm{column}=(e{column}>>4u)|((m{column}&192u)>>2u);\n"
                );
            }
            body += &format!(
                "let w0{column}=word_at(col+{column}u,bk,{}u+pair*8u+run*2u);let w1{column}=word_at(col+{column}u,bk,{}u+pair*8u+run*2u);\n",
                4 + group / 2 * 16,
                5 + group / 2 * 16
            );
            for word in 0..2 {
                body += &format!(
                    "let q{word}{column}=vec4<f32>(f32(w{word}{column}&{}u),f32(w{word}{column}&{}u),f32(w{word}{column}&{}u),f32(w{word}{column}&{}u));\n",
                    15u32 << nibble,
                    15u32 << (nibble + 8),
                    15u32 << (nibble + 16),
                    15u32 << (nibble + 24)
                );
            }
            body += &format!(
                "partial{column}+=(dot(mx0,q0{column})+dot(mx1,q1{column}))*(f32(gs{column})*scales{column}.x)-dx*(f32(gm{column})*scales{column}.y);\n"
            );
        }
        body += "}\n";
    }
    let mut source = format!(
        r#"
@group(0) @binding(0) var<storage,read> weights:array<u32,{words}>;
@group(0) @binding(1) var<storage,read> input:array<vec4<f32>,{inputs}>;
@group(0) @binding(2) var<storage,read_write> output:array<f32,{out}>;
fn word_at(col:u32,bk:u32,word:u32)->u32{{if(col>={n}u){{return 0u;}}return weights[{address}];}}
@compute @workgroup_size({threads})
fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) tid:u32){{
 let col=wg.x*{outputs}u+tid/32u*2u;let lane=tid%32u;{mapping}
 var sum0=0.0;var sum1=0.0;
 for(var it=0u;it<{passes}u;it++){{
 let bk=it*4u+block_lane;let xbase=bk*64u+pair*16u+run*2u;
 let scales0=unpack2x16float(word_at(col,bk,0u));let ps0=word_at(col,bk,1u);let pm0=word_at(col,bk,2u);let pe0=word_at(col,bk,3u);
 let scales1=unpack2x16float(word_at(col+1u,bk,0u));let ps1=word_at(col+1u,bk,1u);let pm1=word_at(col+1u,bk,2u);let pe1=word_at(col+1u,bk,3u);
 var partial0=0.0;var partial1=0.0;{body}
 sum0+=partial0;sum1+=partial1;
 }}
 let total0=subgroupAdd(sum0);let total1=subgroupAdd(sum1);
 if(lane==0u && col<{n}u){{output[col]=total0;if(col+1u<{n}u){{output[col+1u]=total1;}}}}
}}
"#,
        words = n * k / 256 * 36,
        inputs = k / 4,
        out = n + 64,
        passes = k / 1024,
        outputs = subgroups * 2,
    );
    if uniform {
        source = source.replace("var<storage,read> input", "var<uniform> input");
    }
    if shared {
        source = source.replace(
            "@compute",
            "var<workgroup> staged:array<vec4<f32>,256>;\n@compute",
        );
        source = source.replace("let bk=it*4u+block_lane;", &format!("for(var i=tid;i<256u;i+={threads}u){{staged[i]=input[it*256u+i];}}workgroupBarrier();let bk=it*4u+block_lane;"));
        source = source.replace("let xbase=bk*64u", "let xbase=block_lane*64u");
        source = source.replace("=input[xbase", "=staged[xbase");
        source = source.replace(
            "sum0+=partial0;sum1+=partial1;",
            "sum0+=partial0;sum1+=partial1;workgroupBarrier();",
        );
    }
    source
}
fn shader(n: usize, k: usize, lanes: usize, interleaved: bool) -> String {
    let columns = 32 / lanes;
    let address = if interleaved {
        format!(
            "((col/{columns}u*{}u+bk)*36u+word)*{columns}u+col%{columns}u",
            k / 256
        )
    } else {
        format!("(col*{}u+bk)*36u+word", k / 256)
    };
    let mut merge = String::new();
    let mut offset = 1;
    while offset < lanes {
        merge += &format!("sum += subgroupShuffleXor(sum,{offset}u);\n");
        offset *= 2;
    }
    let word_body = r#"    let group=wi/8u;let j=wi%8u;let shift=(group%4u)*8u;
    let s=(ps>>shift)&255u;let m=(pm>>shift)&255u;let e=(pe>>shift)&255u;
    let gs=select(s&63u,(e&15u)|((s&192u)>>2u),group>=4u);
    let gm=select(m&63u,(e>>4u)|((m&192u)>>2u),group>=4u);
    let bits=word_at(col,bk,4u+group/2u*8u+j)>>(group%2u*4u);
    let quant=vec4<f32>(f32(bits&15u),f32((bits>>8u)&15u),f32((bits>>16u)&15u),f32((bits>>24u)&15u));
    let x=input[bk*64u+wi];
    sum += dot(x,quant)*(f32(gs)*scales.x)-dot(x,vec4<f32>(1.0))*(f32(gm)*scales.y);
"#;
    let iterations = if std::env::var_os("LAB_LAYOUT_UNROLL").is_some() {
        (0..64 / lanes)
            .map(|i| format!("{{let wi=lane+{}u;{word_body}}}", i * lanes))
            .collect::<String>()
    } else {
        format!("for(var wi=lane;wi<64u;wi+={lanes}u){{{word_body}}}")
    };
    let iterations = if std::env::var_os("LAB_LAYOUT_GROUPED").is_some() {
        let groups_per_lane = 8 / (lanes / 8).max(1);
        let words_per_group = 8 / lanes.min(8);
        let mut body = String::new();
        for group_index in 0..groups_per_lane {
            body += &format!(
                "{{let group={}u+lane/8u;let shift=(group%4u)*8u;",
                group_index * (lanes / 8).max(1)
            );
            body += "let s=(ps>>shift)&255u;let m=(pm>>shift)&255u;let e=(pe>>shift)&255u;\n";
            body += "let gs=select(s&63u,(e&15u)|((s&192u)>>2u),group>=4u);let gm=select(m&63u,(e>>4u)|((m&192u)>>2u),group>=4u);var dq=0.0;var dx=vec4<f32>(0.0);\n";
            for word_index in 0..words_per_group {
                body += &format!("{{let j=lane%8u+{}u;", word_index * lanes.min(8));
                body += "let bits=word_at(col,bk,4u+group/2u*8u+j)>>(group%2u*4u);let x=input[bk*64u+group*8u+j];\n";
                if std::env::var_os("LAB_LAYOUT_MASKED").is_some() {
                    body += "let quant=vec4<f32>(f32(bits&15u),f32(bits&3840u),f32(bits&983040u),f32(bits&251658240u));dq+=dot(x*vec4<f32>(1.0,0.00390625,0.0000152587890625,0.000000059604644775390625),quant);\n";
                } else {
                    body += "let quant=vec4<f32>(f32(bits&15u),f32((bits>>8u)&15u),f32((bits>>16u)&15u),f32((bits>>24u)&15u));dq+=dot(x,quant);\n";
                }
                body += "dx+=x;}\n";
            }
            body += "sum+=dq*(f32(gs)*scales.x)-dot(dx,vec4<f32>(1.0))*(f32(gm)*scales.y); }\n";
        }
        body
    } else {
        iterations
    };
    let source = format!(
        r#"
@group(0) @binding(0) var<storage,read> weights:array<u32>;
@group(0) @binding(1) var<storage,read> input:array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> output:array<f32>;
fn word_at(col:u32,bk:u32,word:u32)->u32{{return weights[{address}];}}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) tid:u32){{
 let col=wg.x*{outputs}u+tid/{lanes}u;
 let lane=tid%{lanes}u;
 // All lanes, including padded columns, reach every subgroup collective.
 var sum=0.0;
 if(col<{n}u){{
  for(var bk=0u;bk<{blocks}u;bk++){{
   let scales=unpack2x16float(word_at(col,bk,0u));
   let ps=word_at(col,bk,1u);let pm=word_at(col,bk,2u);let pe=word_at(col,bk,3u);
   {iterations}
  }}
 }}
 {merge}
 if(lane==0u && col<{n}u){{output[col]=sum;}}
}}
"#,
        outputs = 128 / lanes,
        blocks = k / 256
    );
    if std::env::var_os("LAB_LAYOUT_STATIC").is_some() {
        let stored_columns = if interleaved {
            n.div_ceil(columns) * columns
        } else {
            n
        };
        source
            .replace(
                "weights:array<u32>",
                &format!("weights:array<u32,{}>", stored_columns * k / 256 * 36),
            )
            .replace(
                "input:array<vec4<f32>>",
                &format!("input:array<vec4<f32>,{}>", k / 4),
            )
            .replace(
                "output:array<f32>",
                &format!("output:array<f32,{}>", n + 64),
            )
    } else {
        source
    }
}
// Two outputs per lane amortize input loads and the scale-group input sum.
fn paired_shader(n: usize, k: usize, lanes: usize, interleaved: bool) -> String {
    let columns = 64 / lanes;
    let extent = if interleaved {
        n.div_ceil(columns) * columns
    } else {
        n
    };
    let address = if interleaved {
        format!(
            "((col/{columns}u*{}u+bk)*36u+word)*{columns}u+col%{columns}u",
            k / 256
        )
    } else {
        format!("(col*{}u+bk)*36u+word", k / 256)
    };
    let mut body = String::new();
    for g in 0..8 / (lanes / 8).max(1) {
        body += &format!(
            "{{let group={}u+lane/8u;let shift=(group%4u)*8u;",
            g * (lanes / 8).max(1)
        );
        body += "var dq=vec2<f32>(0.0);var dx=vec4<f32>(0.0);\n";
        for a in 0..2 {
            body += &format!(
                "let s{a}=(ps{a}>>shift)&255u;let m{a}=(pm{a}>>shift)&255u;let e{a}=(pe{a}>>shift)&255u;let gs{a}=select(s{a}&63u,(e{a}&15u)|((s{a}&192u)>>2u),group>=4u);let gm{a}=select(m{a}&63u,(e{a}>>4u)|((m{a}&192u)>>2u),group>=4u);\n"
            );
        }
        for w in 0..8 / lanes.min(8) {
            body += &format!(
                "{{let j=lane%8u+{}u;let x=input[bk*64u+group*8u+j];let mx=x*vec4<f32>(1.0,0.00390625,0.0000152587890625,0.000000059604644775390625);",
                w * lanes.min(8)
            );
            for a in 0..2 {
                body += &format!(
                    "let bits{a}=word_at(col+{a}u,bk,4u+group/2u*8u+j)>>(group%2u*4u);let quant{a}=vec4<f32>(f32(bits{a}&15u),f32(bits{a}&3840u),f32(bits{a}&983040u),f32(bits{a}&251658240u));\n"
                );
            }
            body += "dq+=vec2<f32>(dot(mx,quant0),dot(mx,quant1));dx+=x;}\n";
        }
        body += "sum+=dq*vec2<f32>(f32(gs0)*scales0.x,f32(gs1)*scales1.x)-dot(dx,vec4<f32>(1.0))*vec2<f32>(f32(gm0)*scales0.y,f32(gm1)*scales1.y);}\n";
    }
    let mut merge = String::new();
    for shift in 0..lanes.trailing_zeros() {
        merge += &format!("sum+=subgroupShuffleXor(sum,{}u);\n", 1 << shift);
    }
    format!(
        r#"
@group(0) @binding(0) var<storage,read> weights:array<u32,{words}>;
@group(0) @binding(1) var<storage,read> input:array<vec4<f32>,{inputs}>;
@group(0) @binding(2) var<storage,read_write> output:array<f32,{out}>;
fn word_at(col:u32,bk:u32,word:u32)->u32{{if(col>={n}u){{return 0u;}}return weights[{address}];}}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wg:vec3<u32>,@builtin(local_invocation_index) tid:u32){{
 let col=wg.x*{outputs}u+tid/{lanes}u*2u;let lane=tid%{lanes}u;var sum=vec2<f32>(0.0);
 if(col<{n}u){{for(var bk=0u;bk<{blocks}u;bk++){{
 let scales0=unpack2x16float(word_at(col,bk,0u));let ps0=word_at(col,bk,1u);let pm0=word_at(col,bk,2u);let pe0=word_at(col,bk,3u);
 let scales1=unpack2x16float(word_at(col+1u,bk,0u));let ps1=word_at(col+1u,bk,1u);let pm1=word_at(col+1u,bk,2u);let pe1=word_at(col+1u,bk,3u);
 {body}
 }}}}
 {merge}
 if(lane==0u && col<{n}u){{output[col]=sum.x;if(col+1u<{n}u){{output[col+1u]=sum.y;}}}}
}}
"#,
        words = extent * k / 256 * 36,
        inputs = k / 4,
        out = n + 64,
        outputs = 256 / lanes,
        blocks = k / 256
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
    let (n, k, copies, repeats) = (number(28672)?, number(4096)?, number(8)?, number(16)?);
    if args.next().is_some()
        || !(1..=131072).contains(&n)
        || !(256..=32768).contains(&k)
        || !k.is_multiple_of(256)
        || !(1..=32).contains(&copies)
        || !(1..=128).contains(&repeats)
        || n.div_ceil(64) * 64 * k / 256 * 144 * copies * 2 > 5 * 1024 * 1024 * 1024
    {
        return Err("invalid dimensions or working set exceeds 5 GiB".into());
    }
    let expanded_scales = std::env::var_os("LAB_LAYOUT_EXPANDED_SCALES").is_some();
    let prescaled = std::env::var_os("LAB_LAYOUT_PRESCALED").is_some();
    if expanded_scales
        && (prescaled
            || n * k / 256 * (144 + 148) * copies > 512 * 1024 * 1024
            || std::env::var_os("LAB_LAYOUT_SHADER_BASELINE").is_none()
            || std::env::var_os("LAB_LAYOUT_SHADER_CANDIDATE").is_none())
    {
        return Err(
            "expanded scale probe requires external shaders, no prescaling and <=512 MiB weights"
                .into(),
        );
    }
    if prescaled
        && (n * k / 256 * (144 + 192) * copies > 512 * 1024 * 1024
            || std::env::var_os("LAB_LAYOUT_SHADER_BASELINE").is_none()
            || std::env::var_os("LAB_LAYOUT_SHADER_CANDIDATE").is_none())
    {
        return Err("prescaled probe requires external shaders and at most 512 MiB weights".into());
    }
    let glu_probe = std::env::var_os("LAB_LAYOUT_GLU").is_some();
    if glu_probe
        && (std::env::var_os("LAB_LAYOUT_RMS").is_some()
            || prescaled
            || expanded_scales
            || !n.is_multiple_of(4)
            || n * k / 256 * 144 * copies * 2 > 512 * 1024 * 1024
            || std::env::var_os("LAB_LAYOUT_SHADER_BASELINE").is_none()
            || std::env::var_os("LAB_LAYOUT_SHADER_CANDIDATE").is_none())
    {
        return Err("GLU probe requires external shaders, N divisible by four, no other transform and <=512 MiB weights".into());
    }
    let gpu = pollster::block_on(fusor_gpu::GpuDevice::request(None))?;
    if gpu.caps().subgroup_width() != 32
        || !gpu
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::SUBGROUP)
    {
        return Err("requires 32-wide subgroups and timestamps".into());
    }
    let (d, q) = (gpu.device(), gpu.queue());
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let mut original = vec![0u8; n * k / 256 * 144];
    // Deterministic, nonperiodic coefficients; varying exactly representable
    // half scales ensure every metadata bit participates in CPU validation.
    let mut rng = 0x6197_3521u32;
    for block in original.chunks_exact_mut(144) {
        for byte in block.iter_mut() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            *byte = rng as u8;
        }
        block[..2].copy_from_slice(&(0x1800u16 + (rng as u16 & 1023)).to_le_bytes());
        block[2..4].copy_from_slice(&(0x1400u16 + ((rng >> 10) as u16 & 1023)).to_le_bytes());
    }
    if original.len() as u64 > d.limits().max_storage_buffer_binding_size {
        return Err("weight binding exceeds adapter limit".into());
    }
    let native: Vec<_> = (0..copies)
        .map(|_| {
            let b = buffer(d, original.len(), usage);
            q.write_buffer(&b, 0, &original);
            b
        })
        .collect();
    // External fused-RMS probes keep raw x, gamma, and the CPU-normalized
    // reference in three adjacent ranges. The baseline GEMV reads the third.
    let rms_probe = std::env::var_os("LAB_LAYOUT_RMS").is_some();
    if rms_probe
        && (k != 4096
            || std::env::var_os("LAB_LAYOUT_SHADER_BASELINE").is_none()
            || std::env::var_os("LAB_LAYOUT_SHADER_CANDIDATE").is_none())
    {
        return Err("RMS probe requires K4096 and both external shaders".into());
    }
    let input = buffer(
        d,
        k * 4 * if rms_probe { 3 } else { 1 },
        usage | wgpu::BufferUsages::UNIFORM,
    );
    let output_columns = if glu_probe { n / 2 } else { n };
    let output = buffer(
        d,
        (output_columns + 64) * 4,
        usage | wgpu::BufferUsages::COPY_SRC,
    );
    let projection = glu_probe.then(|| buffer(d, (n + 64) * 4, usage));
    let glu_pipeline = if glu_probe {
        Some(pipeline(
            d,
            &format!(
                r#"
@group(0) @binding(0) var<storage,read> pair:array<f32>;
@group(0) @binding(1) var<storage,read_write> result:array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id:vec3<u32>) {{
 let i=id.x;if(i>={output_columns}u){{return;}}
 let gate=pair[i];result[i]=(gate/(1.0+exp(-gate)))*pair[i+{output_columns}u];
}}
"#
            ),
        )?)
    } else {
        None
    };
    let glu_binding = glu_pipeline
        .as_ref()
        .zip(projection.as_ref())
        .map(|(p, raw)| bind(d, p, &[raw, &output]));
    let readback = buffer(
        d,
        (output_columns + 64) * 4,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let canary: Vec<_> = (0..output_columns + 64)
        .flat_map(|_| f32::NAN.to_le_bytes())
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
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
    );
    let block_layout = std::env::var_os("LAB_LAYOUT_BLOCKS").is_some();
    let input_mode = std::env::var("LAB_LAYOUT_INPUT").unwrap_or_default();
    let subgroups: usize = std::env::var("LAB_LAYOUT_SUBGROUPS")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(2);
    if !matches!(subgroups, 1 | 2 | 4 | 8 | 16 | 32)
        || subgroups * 32 > d.limits().max_compute_invocations_per_workgroup as usize
        || subgroups * 32 > d.limits().max_compute_workgroup_size_x as usize
        || !matches!(input_mode.as_str(), "" | "uniform" | "shared")
        || (!input_mode.is_empty() && !block_layout)
        || (input_mode == "uniform" && k * 4 > d.limits().max_uniform_buffer_binding_size as usize)
    {
        return Err("invalid input mode, subgroup count, or uniform extent".into());
    }
    if block_layout && !k.is_multiple_of(1024) {
        return Err("four-block layout requires K divisible by 1024".into());
    }
    let selected = if block_layout {
        "8".into()
    } else {
        std::env::var("LAB_LAYOUT_LANES").unwrap_or_else(|_| "1,2,4,8,16,32".into())
    };
    for lanes in selected.split(',').map(str::parse::<usize>) {
        let lanes = lanes?;
        if !matches!(lanes, 1 | 2 | 4 | 8 | 16 | 32) {
            return Err("lanes must be 1,2,4,8,16,32".into());
        }
        let paired = std::env::var_os("LAB_LAYOUT_PAIRED").is_some();
        let columns = if block_layout {
            1
        } else {
            (if paired { 64 } else { 32 }) / lanes
        };
        let padded = n.div_ceil(columns) * columns;
        let mut interleaved = vec![0u8; padded * k / 256 * 144];
        for col in 0..n {
            for bk in 0..k / 256 {
                for word in 0..36 {
                    let src = (col * k / 256 + bk) * 144 + word * 4;
                    let dst = if block_layout && !input_mode.is_empty() {
                        src
                    } else if block_layout {
                        ((col * (k / 1024) + bk / 4) * 36 + word) * 16 + bk % 4 * 4
                    } else {
                        (((col / columns * (k / 256) + bk) * 36 + word) * columns + col % columns)
                            * 4
                    };
                    interleaved[dst..dst + 4].copy_from_slice(&original[src..src + 4]);
                }
            }
        }
        if expanded_scales {
            if !block_layout || input_mode.is_empty() {
                return Err("expanded scales require block input mode".into());
            }
            interleaved = Vec::with_capacity(original.len() / 144 * 148);
            for block in original.chunks_exact(144) {
                let (scales, offsets) =
                    fusor_gguf::blocks::unpack_k4_scales_offsets(block[4..16].try_into()?);
                interleaved.extend_from_slice(&block[..4]);
                interleaved.extend_from_slice(&scales);
                interleaved.extend_from_slice(&offsets);
                interleaved.extend_from_slice(&block[16..]);
            }
        }
        if prescaled {
            if !block_layout || input_mode.is_empty() {
                return Err("prescaled probe requires block input mode".into());
            }
            interleaved = Vec::with_capacity(original.len() / 144 * 192);
            for block in original.chunks_exact(144) {
                let d = half::f16::from_le_bytes(block[..2].try_into()?).to_f32();
                let dmin = half::f16::from_le_bytes(block[2..4].try_into()?).to_f32();
                let (scales, offsets) =
                    fusor_gguf::blocks::unpack_k4_scales_offsets(block[4..16].try_into()?);
                for group in 0..8 {
                    interleaved.extend_from_slice(&(f32::from(scales[group]) * d).to_le_bytes());
                    interleaved
                        .extend_from_slice(&(f32::from(offsets[group]) * dmin).to_le_bytes());
                }
                interleaved.extend_from_slice(&block[16..]);
            }
        }
        if let Ok(column) = std::env::var("LAB_LAYOUT_CORRUPT_COLUMN") {
            let col: usize = column.parse()?;
            if col >= n {
                return Err("corrupt column outside output".into());
            }
            let at = ((col / columns * (k / 256)) * 36 * columns + col % columns) * 4;
            interleaved[at..at + 4].copy_from_slice(&0x7e00_7e00u32.to_le_bytes());
        }
        let packed: Vec<_> = (0..copies)
            .map(|_| {
                let b = buffer(d, interleaved.len(), usage);
                q.write_buffer(&b, 0, &interleaved);
                b
            })
            .collect();
        let source = |interleaved| {
            if block_layout {
                block_shader(n, k, interleaved)
            } else if paired {
                paired_shader(n, k, lanes, interleaved)
            } else {
                shader(n, k, lanes, interleaved)
            }
        };
        let sources = match (
            std::env::var_os("LAB_LAYOUT_SHADER_BASELINE"),
            std::env::var_os("LAB_LAYOUT_SHADER_CANDIDATE"),
        ) {
            (Some(a), Some(b)) if block_layout && !input_mode.is_empty() => {
                [std::fs::read_to_string(a)?, std::fs::read_to_string(b)?]
            }
            (None, None) => [source(false), source(true)],
            _ => return Err("external shaders require both paths and block input mode".into()),
        };
        let pipes = [pipeline(d, &sources[0])?, pipeline(d, &sources[1])?];
        let bindings: Vec<Vec<_>> = pipes
            .iter()
            .zip([&native, &packed])
            .enumerate()
            .map(|(arm, (p, ws))| {
                let dst = if arm == 0 {
                    projection.as_ref().unwrap_or(&output)
                } else {
                    &output
                };
                ws.iter().map(|w| bind(d, p, &[w, &input, dst])).collect()
            })
            .collect();
        let groups = n.div_ceil(if block_layout {
            subgroups * 2
        } else {
            (if paired { 256 } else { 128 }) / lanes
        }) as u32;
        let mut samples: [Vec<f64>; 2] = Default::default();
        let mut pairs = Vec::new();
        // Keep timing a finite, CPU-checked GLU probe after an arm mismatch so
        // an unpromising fusion can be rejected without further implementation.
        // The run still fails, and its timing rows explicitly remain invalid.
        let mut glu_mismatch = None;
        for round in 0..15 {
            let x: Vec<f32> = (0..k)
                .map(|i| (((i * 17 + i / 7 + round * 13) % 101) as f32 - 50.) / 29.71)
                .collect();
            let (input_values, reference_x) = if rms_probe {
                let gamma: Vec<f32> = (0..k)
                    .map(|i| 1. + ((i * 13 % 97) as f32 - 48.) / 101.)
                    .collect();
                let norm = ((x.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / k as f64)
                    + 1e-5)
                    .sqrt() as f32;
                let reference_x: Vec<f32> = x
                    .iter()
                    .zip(&gamma)
                    .map(|(&v, &g)| (v / norm) * g)
                    .collect();
                let mut input_values = x.clone();
                input_values.extend(gamma);
                input_values.extend_from_slice(&reference_x);
                (input_values, reference_x)
            } else {
                (x.clone(), x.clone())
            };
            q.write_buffer(
                &input,
                0,
                &input_values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            );
            let reference = expected(
                QFmt::Q4K,
                &original,
                &reference_x,
                n,
                k,
                round == 0 || glu_probe,
            );
            let reference = if glu_probe {
                reference[..output_columns]
                    .iter()
                    .zip(&reference[output_columns..])
                    .map(|(&(i, gate), &(_, up))| (i, (gate / (1. + (-gate).exp())) * up))
                    .collect()
            } else {
                reference
            };
            let mut outputs: Option<Vec<f32>> = None;
            let mut times = [0.; 2];
            for step in 0..2 {
                let arm = if round % 2 == 0 { step } else { 1 - step };
                q.write_buffer(&output, 0, &canary);
                let mut encoder = d.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Q4 layout replay"),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &queries,
                            beginning_of_pass_write_index: Some(0),
                            end_of_pass_write_index: Some(1),
                        }),
                    });
                    pass.set_pipeline(&pipes[arm]);
                    for _ in 0..repeats {
                        for bind in &bindings[arm] {
                            if glu_probe && arm == 0 {
                                pass.set_pipeline(&pipes[arm]);
                            }
                            pass.set_bind_group(0, bind, &[]);
                            pass.dispatch_workgroups(groups, 1, 1);
                            if arm == 0
                                && let Some((p, bg)) =
                                    glu_pipeline.as_ref().zip(glu_binding.as_ref())
                            {
                                pass.set_pipeline(p);
                                pass.set_bind_group(0, bg, &[]);
                                pass.dispatch_workgroups(output_columns.div_ceil(256) as u32, 1, 1);
                            }
                        }
                    }
                }
                encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output.size());
                q.submit([encoder.finish()]);
                let values = check(&read(d, &readback)?, &reference, output_columns)
                    .map_err(|e| format!("round={round} arm={arm}: {e}"))?;
                if let Some(other) = &outputs {
                    if block_layout {
                        for (i, (&a, &b)) in values[..output_columns]
                            .iter()
                            .zip(&other[..output_columns])
                            .enumerate()
                        {
                            if (a - b).abs() > 0.0002 + 0.000002 * a.abs().max(b.abs()) {
                                if glu_probe {
                                    glu_mismatch.get_or_insert_with(|| {
                                        format!("layout arm mismatch at {i}: {a} != {b}")
                                    });
                                    continue;
                                }
                                return Err(
                                    format!("layout arm mismatch at {i}: {a} != {b}").into()
                                );
                            }
                        }
                    } else if values[..output_columns] != other[..output_columns] {
                        return Err("layout arms differ despite identical arithmetic".into());
                    }
                } else {
                    outputs = Some(values);
                }
                let mut encoder = d.create_command_encoder(&Default::default());
                encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
                encoder.copy_buffer_to_buffer(&resolved, 0, &stamps, 0, 16);
                q.submit([encoder.finish()]);
                let bytes = read(d, &stamps)?;
                let start = u64::from_le_bytes(bytes[..8].try_into()?);
                let end = u64::from_le_bytes(bytes[8..].try_into()?);
                if end <= start {
                    return Err("invalid timestamp interval".into());
                }
                let us = (end - start) as f64 * f64::from(q.get_timestamp_period())
                    / 1000.
                    / (repeats * copies) as f64;
                times[arm] = us;
                if round >= 3 {
                    samples[arm].push(us);
                }
            }
            if round >= 3 {
                pairs.push(times[0] / times[1]);
                eprintln!(
                    "Q4_LAYOUT_SAMPLE round={round} baseline_us={} candidate_us={} ratio={}",
                    times[0],
                    times[1],
                    times[0] / times[1]
                );
            }
        }
        pairs.sort_by(f64::total_cmp);
        for (arm, values) in samples.iter_mut().enumerate() {
            values.sort_by(f64::total_cmp);
            println!(
                "Q4_LAYOUT n={n} k={k} lanes={lanes} paired={paired} block_layout={block_layout} input_mode={input_mode:?} subgroups={subgroups} candidate={} copies={copies} repeats={repeats} gpu_us={:.3} p25_us={:.3} p75_us={:.3} paired_speedup={:.4} cpu_all_columns={output_columns} full_arm_match={} runtime_checks=true rms_probe={rms_probe} prescaled={prescaled} glu_probe={glu_probe}",
                arm == 1,
                (values[5] + values[6]) / 2.,
                values[3],
                values[9],
                (pairs[5] + pairs[6]) / 2.,
                glu_mismatch.is_none()
            );
        }
        if let Some(error) = glu_mismatch {
            return Err(error.into());
        }
    }
    Ok(())
}
