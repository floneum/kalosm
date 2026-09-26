use super::plan::{BLOCK, Plan, visit_sources};
use fusor_ir::{
    Result,
    dtype::{Dtype, RoundMode, Splat},
    egraph::Id,
    error::Error,
    ir::logical::{LeafKind, Logical, ScatterCombine},
    scalar::{BinOp, CmpOp, ScalarExpr, ScalarKind, UnOp},
    semantics::children::children_logical,
};
use std::fmt::Write;

fn ty(d: Dtype) -> Result<&'static str> {
    match d {
        Dtype::F32 => Ok("f32"),
        Dtype::U32 => Ok("u32"),
        Dtype::I32 => Ok("i32"),
        _ => Err(Error::Dtype("unsupported program scalar type".into())),
    }
}
fn lit(x: Splat) -> Result<String> {
    // Browser WGSL rejects constant evaluation that produces infinity or NaN,
    // including a constant bitcast. Reduction identities need their exact bits;
    // a function parameter makes the conversion a runtime expression instead.
    if x.dtype() == Dtype::F32 && !f32::from_bits(x.bits()).is_finite() {
        return Ok(format!("f32_bits({}u)", x.bits()));
    }
    Ok(format!("bitcast<{}>({}u)", ty(x.dtype())?, x.bits()))
}
fn stride(shape: &[u32], axis: usize) -> u32 {
    shape[axis + 1..].iter().product()
}
fn coord(at: &str, shape: &[u32], axis: usize) -> String {
    if shape[axis] == 1 {
        "0u".into()
    } else {
        format!("(({at}) / {}u % {}u)", stride(shape, axis), shape[axis])
    }
}
fn sum(xs: impl IntoIterator<Item = String>) -> String {
    let xs: Vec<_> = xs.into_iter().collect();
    if xs.is_empty() {
        "0u".into()
    } else {
        format!("({})", xs.join(" + "))
    }
}
fn scalar_node(
    p: &Plan,
    e: &ScalarExpr,
    args: &[String],
    at: &str,
    shape: &[u32],
    sub: &mut impl FnMut(&ScalarExpr) -> Result<String>,
) -> Result<String> {
    Ok(match e.kind() {
        ScalarKind::Arg(i) => args
            .get(*i as usize)
            .cloned()
            .ok_or_else(|| Error::Plan("scalar operand out of range".into()))?,
        ScalarKind::Lit(x) => lit(x.0)?,
        ScalarKind::IndexOf(axis) => {
            format!("{}({})", ty(e.dtype())?, coord(at, shape, *axis as usize))
        }
        ScalarKind::Uniform(sym) => {
            let offset = p
                .uniforms
                .iter()
                .find(|u| u.sym == *sym)
                .ok_or_else(|| Error::Plan("missing program uniform".into()))?
                .offset;
            let value = format!("bitcast<f32>(arena[{offset}u])");
            if e.dtype() == Dtype::F32 {
                value
            } else {
                format!("{}({value})", ty(e.dtype())?)
            }
        }
        ScalarKind::Un { op, x } => {
            let x = sub(x)?;
            match op {
                UnOp::Neg => format!("(-{x})"),
                UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp => format!("exp({x})"),
                UnOp::Exp2 => format!("exp2({x})"),
                UnOp::Log => format!("log({x})"),
                UnOp::Log2 => format!("log2({x})"),
                UnOp::Sqrt => format!("sqrt({x})"),
                UnOp::InverseSqrt => format!("inverseSqrt({x})"),
                UnOp::Abs => format!("abs({x})"),
                UnOp::Sin => format!("sin({x})"),
                UnOp::Cos => format!("cos({x})"),
                UnOp::Tan => format!("tan({x})"),
                UnOp::Tanh => format!("tanh({x})"),
                UnOp::Asin => format!("asin({x})"),
                UnOp::Acos => format!("acos({x})"),
                UnOp::Atan => format!("atan({x})"),
                UnOp::Sinh => format!("sinh({x})"),
                UnOp::Cosh => format!("cosh({x})"),
                UnOp::Asinh => format!("asinh({x})"),
                UnOp::Acosh => format!("acosh({x})"),
                UnOp::Atanh => format!("atanh({x})"),
                _ => {
                    return Err(Error::Plan(
                        "vector scalar operations are not supported by fixed programs".into(),
                    ));
                }
            }
        }
        ScalarKind::Bin { op, a, b } => {
            let (a, b) = (sub(a)?, sub(b)?);
            match op {
                BinOp::Min => format!("min({a},{b})"),
                BinOp::Max => format!("max({a},{b})"),
                BinOp::Pow => format!("pow({a},{b})"),
                BinOp::LogicalAnd => format!("select(0.0,1.0,({a}!=0.0)&&({b}!=0.0))"),
                BinOp::LogicalOr => format!("select(0.0,1.0,({a}!=0.0)||({b}!=0.0))"),
                _ => format!(
                    "({a} {} {b})",
                    match op {
                        BinOp::Add => "+",
                        BinOp::Sub => "-",
                        BinOp::Mul => "*",
                        BinOp::Div => "/",
                        BinOp::Rem => "%",
                        BinOp::BitAnd => "&",
                        BinOp::BitOr => "|",
                        BinOp::BitXor => "^",
                        BinOp::Shr => ">>",
                        BinOp::Shl => "<<",
                        _ => unreachable!(),
                    }
                ),
            }
        }
        ScalarKind::Cmp { op, a, b } => format!(
            "select({0}(0),{0}(1),{1} {2} {3})",
            ty(e.dtype())?,
            sub(a)?,
            match op {
                CmpOp::Lt => "<",
                CmpOp::Le => "<=",
                CmpOp::Gt => ">",
                CmpOp::Ge => ">=",
                CmpOp::Eq => "==",
                CmpOp::Ne => "!=",
            },
            sub(b)?
        ),
        ScalarKind::Select { c, t, f } => format!(
            "select({},{},{} != {}(0))",
            sub(f)?,
            sub(t)?,
            sub(c)?,
            ty(c.dtype())?
        ),
        ScalarKind::Cast { to, x } => format!("{}({})", ty(*to)?, sub(x)?),
        ScalarKind::Bitcast { to, x } => format!("bitcast<{}>({})", ty(*to)?, sub(x)?),
        ScalarKind::Round { mode, x } => match mode {
            RoundMode::Floor => format!("floor({})", sub(x)?),
            RoundMode::Ceil => format!("ceil({})", sub(x)?),
            RoundMode::Trunc => format!("trunc({})", sub(x)?),
            RoundMode::HalfToEven => format!("round({})", sub(x)?),
            RoundMode::HalfAwayFromZero => format!("(sign({0})*floor(abs({0})+0.5))", sub(x)?),
        },
        _ => {
            return Err(Error::Plan(
                "vector scalar operations are not supported by fixed programs".into(),
            ));
        }
    })
}
fn scalar(p: &Plan, e: &ScalarExpr, args: &[String], at: &str, shape: &[u32]) -> Result<String> {
    scalar_node(p, e, args, at, shape, &mut |x| {
        scalar(p, x, args, at, shape)
    })
}
fn scalar_dag(
    out: &mut String,
    p: &Plan,
    e: &ScalarExpr,
    args: &[String],
    shape: &[u32],
    memo: &mut rustc_hash::FxHashMap<ScalarExpr, String>,
) -> Result<String> {
    if let Some(name) = memo.get(e) {
        return Ok(name.clone());
    }
    let value = scalar_node(p, e, args, "at", shape, &mut |x| {
        scalar_dag(out, p, x, args, shape, memo)
    })?;
    let name = format!("s{}", memo.len());
    writeln!(out, "let {name}={value};").unwrap();
    memo.insert(e.clone(), name.clone());
    Ok(name)
}
fn map_call(p: &Plan, id: Id, at: &str) -> Result<String> {
    let v = p.value(id);
    let Logical::Map { ins, .. } = &v.op else {
        unreachable!()
    };
    let mut args = ins
        .iter()
        .map(|id| load(p, *id, at))
        .collect::<Result<Vec<_>>>()?;
    args.push(at.into());
    Ok(format!("map_{}({})", v.id.0, args.join(",")))
}
fn load(p: &Plan, id: Id, at: &str) -> Result<String> {
    let bounds = super::index::Bounds::from([(0, u64::from(p.value(id).len() - 1))]);
    load_expr(p, id, super::index::Expr::var(0), &[at], &bounds)
}
fn index_wgsl(x: &super::index::Expr, variables: &[&str]) -> String {
    use super::index::Expr;
    match x {
        Expr::Const(c) => format!("{c}u"),
        Expr::Var(v) => format!("({})", variables[*v]),
        Expr::Sum(xs) => sum(xs.iter().map(|(x, n)| {
            if *n == 1 {
                index_wgsl(x, variables)
            } else {
                format!("({}*{n}u)", index_wgsl(x, variables))
            }
        })),
        Expr::Div(x, n) => format!("({}/{n}u)", index_wgsl(x, variables)),
        Expr::Mod(x, n) => format!("({}%{n}u)", index_wgsl(x, variables)),
    }
}
// Every caller supplies an in-range logical index. Gather checks its dynamic
// index before calling this function; matrix loads are guarded at tile tails.
fn load_expr(
    p: &Plan,
    id: Id,
    index: super::index::Expr,
    variables: &[&str],
    bounds: &super::index::Bounds,
) -> Result<String> {
    let v = p.value(id);
    match &v.op {
        Logical::Leaf(LeafKind::Const { value, .. }) => lit(*value),
        Logical::Restride { specs, x, .. } => {
            let source = p.value(*x);
            let mapped = index
                .restride(&v.shape, &source.shape, specs)
                .simplify_digits(bounds);
            load_expr(p, *x, mapped, variables, bounds)
        }
        Logical::Map { ins, .. } if v.forwarded => {
            let mut args = ins
                .iter()
                .map(|id| load_expr(p, *id, index.clone(), variables, bounds))
                .collect::<Result<Vec<_>>>()?;
            args.push(index_wgsl(&index, variables));
            Ok(format!("map_{}({})", v.id.0, args.join(",")))
        }
        _ => Ok(format!(
            "read_{}({})",
            v.id.0,
            index_wgsl(&index, variables)
        )),
    }
}
fn store(out: &mut String, p: &Plan, id: Id, at: &str, value: &str) {
    writeln!(
        out,
        "arena[{}u + ({at})] = bitcast<u32>({value});",
        p.value(id).offset.unwrap()
    )
    .unwrap();
}
pub(crate) fn shader(
    p: &Plan,
    region: usize,
    acceleration: super::ProgramAcceleration,
) -> Result<String> {
    let cooperative = acceleration.cooperative();
    let browser_matrix = acceleration.matrices == super::MatrixInstructions::Browser;
    let commit_job = super::regions::Job {
        stages: vec![],
        groups: p.groups(region, cooperative),
        tiled: false,
        bucketed: false,
    };
    let jobs = p
        .regions
        .get(region)
        .map_or(std::slice::from_ref(&commit_job), |r| r.jobs.as_slice());
    let stages: Vec<_> = jobs.iter().flat_map(|j| j.stages.iter().copied()).collect();
    let commit =
        (p.stats().workgroups == 1 && region + 1 == p.regions.len()) || region == p.regions.len();
    let mut used_maps: rustc_hash::FxHashSet<_> = stages
        .iter()
        .copied()
        .filter(|id| matches!(p.value(*id).op, Logical::Map { .. }))
        .collect();
    let mut used_reads = rustc_hash::FxHashSet::default();
    for id in stages
        .iter()
        .flat_map(|id| children_logical(&p.value(*id).op))
        .chain(
            p.feedback
                .iter()
                .filter(|_| commit)
                .map(|(_, output)| *output),
        )
    {
        visit_sources(&p.values, &p.by_id, id, &mut |v| {
            if v.forwarded {
                used_maps.insert(v.id);
            } else {
                used_reads.insert(v.id);
            }
        });
    }
    // A fixed-size arena: its bounds checks clamp against a constant rather
    // than the runtime buffer length, and binding validates the size once.
    let mut out = format!(
        "@group(0) @binding(0) var<storage,read_write> arena: array<u32,{}>;\n",
        p.stats().arena_bytes / 4
    ) + "var<workgroup> tile_a:array<f32,512>;\nvar<workgroup> tile_b:array<f32,256>;\nvar<workgroup> reduce_scratch:array<u32,256>;\nfn f32_bits(bits:u32)->f32{return bitcast<f32>(bits);}\n";
    if cooperative {
        out.insert_str(
            0,
            if browser_matrix {
                "enable chromium_experimental_subgroup_matrix;\n"
            } else {
                "enable wgpu_cooperative_matrix;\n"
            },
        );
        out.push_str("var<workgroup> tile_c:array<f32,512>;\nvar<workgroup> zeros:array<f32,8>;\n");
    }
    if acceleration.subgroup_scatter && jobs.iter().any(|j| j.bucketed) {
        out.push_str(
            "var<workgroup> scan_counts:array<u32,8>;var<workgroup> positions:array<u32,1024>;\n",
        );
    }
    for v in p.values.iter().filter(|v| used_reads.contains(&v.id)) {
        let dtype = ty(v.dtype)?;
        // Logical ranges are checked by the caller before view composition, and
        // `regions::schedule` admits a same-job read only with a proof that the
        // reading workgroup owns it, so a read needs no mask of its own.
        writeln!(
            out,
            "fn read_{}(i:u32)->{dtype}{{return bitcast<{dtype}>(arena[{}u+i]);}}",
            v.id.0,
            v.offset.unwrap()
        )
        .unwrap();
    }
    for v in p.values.iter().filter(|v| used_maps.contains(&v.id)) {
        if let Logical::Map { expr, ins, outs } = &v.op {
            if *outs != 1 {
                return Err(Error::Plan("multi-output maps are unsupported".into()));
            }
            let args: Vec<_> = ins
                .iter()
                .enumerate()
                .map(|(i, _)| format!("arg{i}"))
                .collect();
            let mut params = ins
                .iter()
                .zip(&args)
                .map(|(id, name)| Ok(format!("{name}:{}", ty(p.value(*id).dtype)?)))
                .collect::<Result<Vec<_>>>()?;
            params.push("at:u32".into());
            writeln!(
                out,
                "fn map_{}({})->{}{{",
                v.id.0,
                params.join(","),
                ty(v.dtype)?
            )
            .unwrap();
            let result = scalar_dag(&mut out, p, expr, &args, &v.shape, &mut Default::default())?;
            writeln!(out, "return {result};}}").unwrap();
        }
    }
    let subgroup = if acceleration.subgroups {
        ",@builtin(subgroup_id) sg:u32,@builtin(subgroup_invocation_id) sl:u32,@builtin(subgroup_size) sw:u32,@builtin(num_subgroups) ns:u32"
    } else {
        ""
    };
    writeln!(out,"@compute @workgroup_size({BLOCK})\nfn main(@builtin(local_invocation_index) lane:u32,@builtin(workgroup_id) group:vec3<u32>{subgroup}){{\n").unwrap();
    let mut group_base = 0;
    for job in jobs {
        let groups = p.job_groups(job, cooperative);
        let tiled = job.tiled;
        writeln!(
            out,
            "if(group.x>={group_base}u){{if(group.x<{}u){{let gid=group.x-{group_base}u;",
            group_base + groups
        )
        .unwrap();
        group_base += groups;
        if cooperative
            && !browser_matrix
            && job
                .stages
                .iter()
                .any(|id| matches!(p.value(*id).op, Logical::Contract { .. }))
        {
            // Read-only accumulator seed: initialize once per job, independent
            // of the output staging tile's reuse between matrix iterations.
            out.push_str("if(lane<8u){zeros[lane]=0.0;}workgroupBarrier();\n");
        }
        for id in &job.stages {
            let v = p.value(*id);
            let n = v.len();
            if job.bucketed && acceleration.subgroup_scatter {
                bucketed_scatter(&mut out, p, *id, groups)?;
                continue;
            }
            if matches!(&v.op, Logical::Contract { .. }) {
                contraction(&mut out, p, *id, groups, acceleration.matrices, tiled)?;
                continue;
            }
            if let Logical::Fold {
                axis, ins, carrier, ..
            } = &v.op
                && carrier.associative
                && p.value(ins[0]).shape[*axis as usize] >= 32
            {
                reduction(&mut out, p, *id, groups, acceleration.subgroups)?;
                continue;
            }
            let share = n.div_ceil(groups);
            writeln!(
            out,
            "// stage {id}: {:?}\n{{\nfor (var at=gid*{share}u+lane; at<min({n}u,(gid+1u)*{share}u); at+={BLOCK}u) {{",
            v.op.tag()
        )
        .unwrap();
            let result = match &v.op {
                Logical::Map { .. } => map_call(p, *id, "at")?,
                Logical::Fold {
                    carrier, axis, ins, ..
                } => {
                    if carrier.width() != 1 || carrier.lanes() != Some(1) {
                        return Err(Error::Plan(
                            "fixed programs currently require scalar reduction carriers".into(),
                        ));
                    }
                    let source = p.value(ins[0]);
                    let axis = *axis as usize;
                    let inner = stride(&source.shape, axis);
                    let width = source.shape[axis];
                    let index = format!(
                        "((at / {inner}u) * {}u + k * {inner}u + at % {inner}u)",
                        width * inner
                    );
                    writeln!(
                        out,
                        "var acc={};\nfor (var k=0u;k<{width}u;k+=1u){{",
                        lit(carrier.identity[0])?
                    )
                    .unwrap();
                    let args = ins
                        .iter()
                        .map(|x| load(p, *x, &index))
                        .collect::<Result<Vec<_>>>()?;
                    let lift = scalar(p, &carrier.lift[0], &args, &index, &source.shape)?;
                    let merged =
                        scalar(p, &carrier.merge[0], &["acc".into(), lift], "at", &v.shape)?;
                    writeln!(out, "acc={merged};\n}}").unwrap();
                    "acc".into()
                }
                Logical::Gather { axis, x, idx } => {
                    let s = p.value(*x);
                    let axis = *axis as usize;
                    let inner = stride(&s.shape, axis);
                    let count = p.value(*idx).len();
                    let index = load(p, *idx, &format!("(at/{inner}u)%{count}u"))?;
                    writeln!(out, "let picked=u32({index});").unwrap();
                    let at = format!(
                        "(at/{}u)*{}u + picked*{inner}u + at%{inner}u",
                        inner * count,
                        inner * s.shape[axis]
                    );
                    writeln!(
                        out,
                        "var gathered={}(0);if(picked<{}u){{gathered={};}}",
                        ty(v.dtype)?,
                        s.shape[axis],
                        load(p, *x, &at)?
                    )
                    .unwrap();
                    "gathered".into()
                }
                Logical::Scatter {
                    axis,
                    base,
                    idx,
                    upd,
                    combine,
                    ..
                } => {
                    let axis = *axis as usize;
                    let inner = stride(&v.shape, axis);
                    let width = v.shape[axis];
                    let count = p.value(*idx).len();
                    writeln!(out,"var acc={};\nfor(var k=0u;k<{count}u;k+=1u){{\nif(u32({})==(at/{inner}u)%{width}u){{",load(p,*base,"at")?,load(p,*idx,"k")?).unwrap();
                    let index = format!(
                        "(at/{}u)*{}u+k*{inner}u+at%{inner}u",
                        inner * width,
                        inner * count
                    );
                    let update = load(p, *upd, &index)?;
                    writeln!(
                        out,
                        "acc={};\n}}\n}}",
                        if *combine == ScatterCombine::Add {
                            format!("acc+{update}")
                        } else {
                            update
                        }
                    )
                    .unwrap();
                    "acc".into()
                }
                _ => {
                    return Err(Error::Plan(format!(
                        "unsupported program stage {:?}",
                        v.op.tag()
                    )));
                }
            };
            store(&mut out, p, *id, "at", &result);
            out.push_str("}\n}\nstorageBarrier();\n");
        }
        // Independent jobs communicate only after dispatch completion. Their
        // final storage barrier has no consumer inside this kernel.
        if p.stats().workgroups > 1 && out.ends_with("storageBarrier();\n") {
            out.truncate(out.len() - "storageBarrier();\n".len());
        }
        // Feedback destinations are persistent, disjoint allocations. All sources
        // stay live through this phase; no parameter is overwritten during backward.
        if commit {
            for (input, output) in &p.feedback {
                let n = p.value(*input).len();
                let share = n.div_ceil(groups);
                writeln!(
                    out,
                    "for(var at=gid*{share}u+lane;at<min({n}u,(gid+1u)*{share}u);at+={BLOCK}u){{"
                )
                .unwrap();
                store(&mut out, p, *input, "at", &load(p, *output, "at")?);
                out.push_str("}\n");
            }
        }
        out.push_str("}}\n");
    }
    out.push_str("}\n");
    Ok(out)
}

fn contraction(
    out: &mut String,
    p: &Plan,
    id: Id,
    groups: u32,
    matrices: super::MatrixInstructions,
    tiled: bool,
) -> Result<()> {
    let cooperative = matrices != super::MatrixInstructions::Portable;
    let browser = matrices == super::MatrixInstructions::Browser;
    use fusor_ir::ir::logical::Label;
    let v = p.value(id);
    let Logical::Contract {
        spec,
        a,
        b,
        acc,
        outs,
    } = &v.op
    else {
        return Err(Error::Plan("expected a contraction stage".into()));
    };
    if *acc != Dtype::F32 || *outs != 1 {
        return Err(Error::Dtype(
            "fixed contractions require one f32 output".into(),
        ));
    }
    let (a, b) = (*a, *b);
    let av = p.value(a);
    let bv = p.value(b);
    let mut dims = std::collections::BTreeMap::new();
    for (labels, value) in [(&spec.a, av), (&spec.b, bv)] {
        for (label, dim) in labels.iter().zip(&value.shape) {
            if dims.insert(*label, *dim).is_some_and(|old| old != *dim) {
                return Err(Error::Shape("contraction dimensions disagree".into()));
            }
        }
    }
    let batch: Vec<_> = spec
        .out
        .iter()
        .filter(|l| spec.a.contains(l) && spec.b.contains(l))
        .copied()
        .collect();
    let rows: Vec<_> = spec
        .out
        .iter()
        .filter(|l| !spec.b.contains(l))
        .copied()
        .collect();
    let cols: Vec<_> = spec
        .out
        .iter()
        .filter(|l| !spec.a.contains(l))
        .copied()
        .collect();
    let red: Vec<_> = dims
        .keys()
        .filter(|l| !spec.out.contains(l))
        .copied()
        .collect();
    let shape = |labels: &[Label]| labels.iter().map(|l| dims[l]).collect::<Vec<_>>();
    let (bs, ms, ns, ks) = (shape(&batch), shape(&rows), shape(&cols), shape(&red));
    let (batch_n, m, n, k): (u32, u32, u32, u32) = (
        bs.iter().product(),
        ms.iter().product(),
        ns.iter().product(),
        ks.iter().product(),
    );
    use super::index::{Bounds, Expr};
    let bounds = Bounds::from([
        (0, u64::from(batch_n - 1)),
        (1, u64::from(m - 1)),
        (2, u64::from(n - 1)),
        (3, u64::from(k - 1)),
    ]);
    // Keep coordinates symbolic through every view. Their ranges are guaranteed
    // by the tile loop and tail guards; composition can eliminate reshape and
    // transpose arithmetic before WGSL hides those relationships from the IR.
    let address_expr = |labels: &[Label], shape: &[u32]| {
        Expr::sum(labels.iter().enumerate().map(|(axis, label)| {
            let (var, dims, i) = if let Some(i) = batch.iter().position(|l| l == label) {
                (0, &bs, i)
            } else if let Some(i) = rows.iter().position(|l| l == label) {
                (1, &ms, i)
            } else if let Some(i) = cols.iter().position(|l| l == label) {
                (2, &ns, i)
            } else {
                (3, &ks, red.iter().position(|l| l == label).unwrap())
            };
            Expr::var(var)
                .coordinate(dims, i)
                .scale(stride(shape, axis) as usize)
        }))
        .simplify_digits(&bounds)
    };
    let address = |labels: &[Label], shape: &[u32], r: &str, c: &str, k: &str| {
        index_wgsl(&address_expr(labels, shape), &["batch", r, c, k])
    };
    let matrix_load = |id: Id, labels: &[Label], shape: &[u32], r: &str, c: &str, k: &str| {
        load_expr(
            p,
            id,
            address_expr(labels, shape),
            &["batch", r, c, k],
            &bounds,
        )
    };
    let share = v.len().div_ceil(groups);
    let tm = if cooperative { 32 } else { 16 };
    let canonical: Vec<_> = batch.iter().chain(&rows).chain(&cols).copied().collect();
    let mt = m.div_ceil(tm);
    let nt = n.div_ceil(16);
    let tile_of = |at: &str| {
        format!(
            "(({at})/{}u*{}u + (({at})/{n}u)%{m}u/{tm}u*{nt}u)",
            m * n,
            mt * nt
        )
    };
    let (begin, end) = if tiled {
        ("gid".into(), format!("{}u", batch_n * mt * nt))
    } else if canonical == spec.out.as_slice() {
        (
            tile_of(&format!("min({}u,gid*{share}u)", v.len() - 1)),
            format!(
                "{}+{nt}u",
                tile_of(&format!("min({}u,(gid+1u)*{share}u-1u)", v.len() - 1))
            ),
        )
    } else {
        ("0u".into(), format!("{}u", batch_n * mt * nt))
    };
    let tile_step = if tiled { groups } else { 1 };
    let accumulator = if browser {
        "var fragment=subgroup_matrix_result<f32,8,8>(0.0);"
    } else if cooperative {
        "var fragment=coopLoad<coop_mat8x8<f32,C>>(&zeros[0],0u);"
    } else {
        "var acc=0.0;"
    };
    writeln!(out,"// tiled contraction {id}\n{{\nlet tr=lane/16u;let tc=lane%16u;\nfor(var tile={begin};tile<{end};tile+={tile_step}u){{\nlet batch=tile/{}u;let mt=tile/{nt}u%{mt}u;let nt=tile%{nt}u;\nlet row=mt*{tm}u+tr;let col=nt*16u+tc;{accumulator}\nfor(var kt=0u;kt<{}u;kt+=1u){{\nlet ka=kt*16u+tc;let kb=kt*16u+tr;\nvar va=0.0;var vb=0.0;",mt*nt,k.div_ceil(16)).unwrap();
    // Tail guards only where an extent leaves a partial tile: a tile never
    // starts past its axis, so a multiple of the tile size needs no test.
    let (m_tail, n_tail, k_tail) = (m % tm != 0, n % 16 != 0, k % 16 != 0);
    writeln!(
        out,
        "if({}){{va={};}}",
        guard(&[(m_tail, format!("row<{m}u")), (k_tail, format!("ka<{k}u"))]),
        matrix_load(a, &spec.a, &av.shape, "row", "0u", "ka")?
    )
    .unwrap();
    writeln!(
        out,
        "if({}){{vb={};}}",
        guard(&[(n_tail, format!("col<{n}u")), (k_tail, format!("kb<{k}u"))]),
        matrix_load(b, &spec.b, &bv.shape, "0u", "col", "kb")?
    )
    .unwrap();
    if cooperative {
        writeln!(
            out,
            "var va2=0.0;if({}){{va2={};}}tile_a[lane+256u]=va2;",
            guard(&[
                (m_tail, format!("row+16u<{m}u")),
                (k_tail, format!("ka<{k}u"))
            ]),
            matrix_load(a, &spec.a, &av.shape, "row+16u", "0u", "ka")?
        )
        .unwrap();
    }
    out.push_str("tile_a[lane]=va;tile_b[lane]=vb;workgroupBarrier();\n");
    // Match emit::coop: the pinned Naga/Metal path holds fragments
    // transposed internally. Non-T loads/stores preserve logical A * B.
    if cooperative {
        out.push_str("{let ar=(sg/2u)*128u;let bc=(sg%2u)*8u;\n");
        for k in [0, 8] {
            if browser {
                writeln!(out,"fragment=subgroupMatrixMultiplyAccumulate(subgroupMatrixLoad<subgroup_matrix_left<f32,8,8>>(&tile_a,ar+{k}u,false,16u),subgroupMatrixLoad<subgroup_matrix_right<f32,8,8>>(&tile_b,{}u+bc,false,16u),fragment);",k*16).unwrap();
            } else {
                writeln!(out,"fragment=coopMultiplyAdd(coopLoad<coop_mat8x8<f32,A>>(&tile_a[ar+{k}u],16u),coopLoad<coop_mat8x8<f32,B>>(&tile_b[{}u+bc],16u),fragment);",k*16).unwrap();
            }
        }
        out.push_str("}\n");
    } else {
        for i in 0..16 {
            writeln!(out, "acc=acc+tile_a[tr*16u+{i}u]*tile_b[{i}u*16u+tc];").unwrap();
        }
    }
    out.push_str("workgroupBarrier();\n}\n");
    if cooperative {
        if browser {
            out.push_str(
                "subgroupMatrixStore(&tile_c,(sg/2u)*128u+(sg%2u)*8u,fragment,false,16u);",
            );
        } else {
            out.push_str("coopStore(fragment,&tile_c[(sg/2u)*128u+(sg%2u)*8u],16u);");
        }
        out.push_str("workgroupBarrier();let acc=tile_c[lane];\n");
    }
    let owned = |index: &str| {
        if tiled {
            "true".into()
        } else {
            format!("({index})/{share}u==gid")
        }
    };
    let output_at = address(&spec.out, &v.shape, "row", "col", "0u");
    let owned_at = owned(&output_at);
    writeln!(
        out,
        "if({}){{",
        guard(&[
            (m_tail, format!("row<{m}u")),
            (n_tail, format!("col<{n}u")),
            (owned_at != "true", owned_at.clone()),
        ])
    )
    .unwrap();
    store(
        out,
        p,
        id,
        &address(&spec.out, &v.shape, "row", "col", "0u"),
        "acc",
    );
    out.push_str("}\n");
    if cooperative {
        let index = address(&spec.out, &v.shape, "row+16u", "col", "0u");
        let owned_index = owned(&index);
        writeln!(
            out,
            "if({}){{",
            guard(&[
                (m_tail, format!("row+16u<{m}u")),
                (n_tail, format!("col<{n}u")),
                (owned_index != "true", owned_index.clone()),
            ])
        )
        .unwrap();
        store(out, p, id, &index, "tile_c[lane+256u]");
        out.push_str("}\n");
    }
    // The next tile's first A/B barrier also retires the current C reads
    // before any lane can overwrite C. Flat ownership may fuse later stages
    // using the same scratch, so retain the barrier on that path.
    if !tiled {
        out.push_str("workgroupBarrier();\n");
    }
    out.push_str("}\n}\nstorageBarrier();\n");
    Ok(())
}
/// The conjunction of the conditions that apply, or `true`.
fn guard(parts: &[(bool, String)]) -> String {
    let parts: Vec<&str> = parts
        .iter()
        .filter(|(on, _)| *on)
        .map(|(_, c)| c.as_str())
        .collect();
    if parts.is_empty() {
        "true".into()
    } else {
        parts.join(" && ")
    }
}

fn reduction(out: &mut String, p: &Plan, id: Id, groups: u32, cooperative: bool) -> Result<()> {
    let v = p.value(id);
    let Logical::Fold {
        carrier, axis, ins, ..
    } = &v.op
    else {
        unreachable!()
    };
    if carrier.width() != 1 || carrier.lanes() != Some(1) {
        return Err(Error::Plan(
            "program reduction requires a scalar carrier".into(),
        ));
    }
    let source = p.value(ins[0]);
    let inner = stride(&source.shape, *axis as usize);
    let width = source.shape[*axis as usize];
    let share = v.len().div_ceil(groups);
    let subgroup = if cooperative && inner == 1 {
        use fusor_ir::scalar::BinOp;
        match carrier.kind() {
            Some(BinOp::Add) => Some("subgroupAdd"),
            Some(BinOp::Mul) => Some("subgroupMul"),
            Some(BinOp::Min) => Some("subgroupMin"),
            Some(BinOp::Max) => Some("subgroupMax"),
            _ => None,
        }
    } else {
        None
    };
    // Contiguous rows give adjacent lanes successive reduction elements.
    // Strided axes put adjacent output columns in adjacent lanes instead.
    // All inactive lanes contribute the carrier identity, including tails.
    let columns = if subgroup.is_some() {
        BLOCK / 32
    } else if inner == 1 {
        BLOCK / width.min(BLOCK).next_power_of_two()
    } else {
        share.min(32).next_power_of_two()
    };
    let lanes = BLOCK / columns;
    let (row, part, delta) = if inner == 1 {
        (format!("lane/{lanes}u"), format!("lane%{lanes}u"), 1)
    } else {
        (
            format!("lane%{columns}u"),
            format!("lane/{columns}u"),
            columns,
        )
    };
    let tree_lanes = lanes;
    let (row, part, columns, lanes) = if subgroup.is_some() {
        ("sg".into(), "sl".into(), "ns".into(), "sw".into())
    } else {
        (row, part, format!("{columns}u"), format!("{lanes}u"))
    };
    let dtype = ty(v.dtype)?;
    writeln!(out,"// collective reduction {id}\n{{let r={row};let part={part};\nfor(var base=gid*{share}u;base<min({}u,(gid+1u)*{share}u);base+={columns}){{let at=base+r;var acc={};\nif(at<min({}u,(gid+1u)*{share}u)){{for(var k=part;k<{width}u;k+={lanes}){{",v.len(),lit(carrier.identity[0])?,v.len()).unwrap();
    let index = format!("((at/{inner}u)*{}u+k*{inner}u+at%{inner}u)", width * inner);
    let args = ins
        .iter()
        .map(|id| load(p, *id, &index))
        .collect::<Result<Vec<_>>>()?;
    let lift = scalar(p, &carrier.lift[0], &args, &index, &source.shape)?;
    let merged = scalar(p, &carrier.merge[0], &["acc".into(), lift], "at", &v.shape)?;
    if let Some(op) = subgroup {
        writeln!(
            out,
            "acc={merged};}}}}let total={op}(acc);if(part==0u && at<min({}u,(gid+1u)*{share}u)){{",
            v.len()
        )
        .unwrap();
        store(out, p, id, "at", "total");
        out.push_str("}}}storageBarrier();\n");
        return Ok(());
    }
    writeln!(
        out,
        "acc={merged};}}}}reduce_scratch[lane]=bitcast<u32>(acc);workgroupBarrier();"
    )
    .unwrap();
    let mut step = tree_lanes / 2;
    while step > 0 {
        let merged = scalar(
            p,
            &carrier.merge[0],
            &[
                format!("bitcast<{dtype}>(reduce_scratch[lane])"),
                format!("bitcast<{dtype}>(reduce_scratch[lane+{}u])", step * delta),
            ],
            "at",
            &v.shape,
        )?;
        writeln!(
            out,
            "if(part<{step}u){{reduce_scratch[lane]=bitcast<u32>({merged});}}workgroupBarrier();"
        )
        .unwrap();
        step /= 2;
    }
    writeln!(
        out,
        "if(part==0u && at<min({}u,(gid+1u)*{share}u)){{",
        v.len()
    )
    .unwrap();
    store(
        out,
        p,
        id,
        "at",
        &format!("bitcast<{dtype}>(reduce_scratch[lane])"),
    );
    out.push_str("}workgroupBarrier();}}storageBarrier();\n");
    Ok(())
}

// Compact matching index positions once per row, then reuse that list for
// every feature. Subgroup rank + subgroup prefix + chunk order keep the exact
// original accumulation order, even for repeated indices. Eight counts are
// initialized before the first read; only the populated positions are read.
fn bucketed_scatter(out: &mut String, p: &Plan, id: Id, groups: u32) -> Result<()> {
    let v = p.value(id);
    let Logical::Scatter {
        axis,
        base,
        idx,
        upd,
        ..
    } = &v.op
    else {
        unreachable!()
    };
    let inner = stride(&v.shape, *axis as usize);
    let width = v.shape[*axis as usize];
    let rows = v.len() / inner;
    let count = p.value(*idx).len();
    writeln!(out,"{{for(var row=gid;row<{rows}u;row+={groups}u){{var total=0u;for(var chunk=0u;chunk<{}u;chunk+=1u){{let k=chunk*256u+sg*32u+sl;var hit=false;if(k<{count}u){{hit=u32({})==row%{width}u;}}let mask=subgroupBallot(hit).x;let rank=countOneBits(mask&((1u<<sl)-1u));if(sl==0u){{scan_counts[sg]=countOneBits(mask);}}workgroupBarrier();var before=0u;var count=0u;",count.div_ceil(256),load(p,*idx,"k")?).unwrap();
    for sg in 0..8 {
        writeln!(
            out,
            "if(sg>{sg}u){{before+=scan_counts[{sg}u];}}count+=scan_counts[{sg}u];"
        )
        .unwrap();
    }
    out.push_str("if(hit){positions[total+before+rank]=k;}total+=count;workgroupBarrier();}\n");
    // Matches are summed in their original order; eight loads issue ahead of
    // their adds so a frequent bucket's long run is not one load at a time.
    let update = |picked: &str| {
        load(
            p,
            *upd,
            &format!("(row/{width}u)*{}u+({picked})*{inner}u+col", count * inner),
        )
    };
    writeln!(out,"for(var col=lane;col<{inner}u;col+=256u){{let at=row*{inner}u+col;var acc={};var i=0u;for(;i+8u<=total;i+=8u){{",load(p,*base,"at")?).unwrap();
    for j in 0..8 {
        writeln!(out, "let u{j}={};", update(&format!("positions[i+{j}u]"))?).unwrap();
    }
    for j in 0..8 {
        writeln!(out, "acc+=u{j};").unwrap();
    }
    writeln!(
        out,
        "}}for(;i<total;i+=1u){{acc+={};}}",
        update("positions[i]")?
    )
    .unwrap();
    store(out, p, id, "at", "acc");
    out.push_str("}workgroupBarrier();}}storageBarrier();\n");
    Ok(())
}
