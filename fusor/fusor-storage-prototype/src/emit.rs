//! Code generators ask for logical loads/stores. Only Access knows placement.
use crate::graph::{Graph, Id, Op, Reduce, stride};
use crate::plan::{BLOCK, Plan, Region, scratch, tree_fold};
use std::fmt::Write;

struct Access<'a> {
    graph: &'a Graph,
    plan: &'a Plan,
    region: &'a Region,
}
impl Access<'_> {
    fn load(&self, id: Id, index: &str) -> String {
        let (base, index) = self.graph.resolve_expr(id, index);
        if self.plan.forwarded.contains(&base) {
            return match &self.graph.values[base].op {
                Op::Point(op, xs) => {
                    op.wgsl(&xs.iter().map(|x| self.load(*x, &index)).collect::<Vec<_>>())
                }
                Op::Reduce(kind, x, axis) => {
                    let shape = &self.graph.values[*x].shape;
                    let inner = stride(shape, *axis);
                    (0..shape[*axis])
                        .map(|k| {
                            self.load(
                                *x,
                                &format!(
                                    "((({index}) / {inner}u) * {}u + {}u + ({index}) % {inner}u)",
                                    shape[*axis] * inner,
                                    k * inner
                                ),
                            )
                        })
                        .reduce(|a, b| match kind {
                            Reduce::Sum => format!("({a} + {b})"),
                            Reduce::Max => format!("max({a}, {b})"),
                        })
                        .unwrap()
                }
                _ => unreachable!("only scalar computations are forwarded"),
            };
        }
        if let Some(off) = self.region.shared.offset(base) {
            let share = self.graph.values[base].len() / self.region.groups;
            format!("local_mem[{off}u + ({index}) - group * {share}u]")
        } else if matches!(self.graph.values[base].op, Op::Input) {
            let off: usize = self.graph.values[..base]
                .iter()
                .filter(|v| matches!(v.op, Op::Input))
                .map(|v| v.len())
                .sum();
            format!("inputs[{off}u + ({index})]")
        } else {
            let off = self
                .plan
                .global
                .offset(base)
                .expect("external value must be allocated");
            format!("global_mem[{off}u + ({index})]")
        }
    }
    fn store(&self, out: &mut String, id: Id, index: &str, value: &str) {
        if let Some(off) = self.region.shared.offset(id) {
            let share = self.graph.values[id].len() / self.region.groups;
            writeln!(
                out,
                "local_mem[{off}u + ({index}) - group * {share}u] = {value};"
            )
            .unwrap();
        }
        if self.region.exports.contains(&id) {
            let off = self.plan.global.offset(id).unwrap();
            writeln!(out, "global_mem[{off}u + ({index})] = {value};").unwrap();
        }
    }
}
pub fn shader(g: &Graph, p: &Plan, r: &Region) -> String {
    let a = Access {
        graph: g,
        plan: p,
        region: r,
    };
    let shared_decl = if r.shared.len == 0 {
        String::new()
    } else {
        format!("var<workgroup> local_mem: array<f32, {}>;\n", r.shared.len)
    };
    let mut out = format!(
        "// THROWAWAY Fusor logical-access prototype\n@group(0) @binding(0) var<storage, read> inputs: array<f32>;\n@group(0) @binding(1) var<storage, read_write> global_mem: array<f32>;\n{shared_decl}@compute @workgroup_size({BLOCK})\nfn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {{\nlet group = wid.x;\n"
    );
    for id in &r.members {
        let v = &g.values[*id];
        let share = v.len() / r.groups;
        writeln!(out, "// %{} {} {:?}\n{{", id, v.name, v.shape).unwrap();
        match &v.op {
            Op::Point(op, xs) => {
                writeln!(out,"for (var i = lane; i < {share}u; i += {BLOCK}u) {{\nlet at = group * {share}u + i;").unwrap();
                let args: Vec<String> = xs.iter().map(|x| a.load(*x, "at")).collect();
                writeln!(out, "let value = {};", op.wgsl(&args)).unwrap();
                a.store(&mut out, *id, "at", "value");
                out.push_str("}\nworkgroupBarrier();\n");
            }
            Op::Reduce(kind, x, axis) => {
                let shape = &g.values[*x].shape;
                let width = shape[*axis];
                let inner = stride(shape, *axis);
                let identity = match kind {
                    Reduce::Sum => "0.0",
                    Reduce::Max => "-3.402823466e+38",
                };
                let combine = |lhs: &str, rhs: &str| match kind {
                    Reduce::Sum => format!("({lhs} + {rhs})"),
                    Reduce::Max => format!("max({lhs}, {rhs})"),
                };
                let tree = tree_fold(g, *id, p.serial_limit);
                if tree {
                    writeln!(out, "for (var i = 0u; i < {share}u; i += 1u) {{").unwrap();
                } else {
                    writeln!(out, "for (var i = lane; i < {share}u; i += {BLOCK}u) {{").unwrap();
                }
                writeln!(out, "let at = group * {share}u + i;\nvar acc = {identity};").unwrap();
                writeln!(
                    out,
                    "for (var k = {}; k < {width}u; k += {}u) {{",
                    if tree { "lane" } else { "0u" },
                    if tree { BLOCK } else { 1 }
                )
                .unwrap();
                let ix = format!(
                    "((at / {inner}u) * {}u + k * {inner}u + at % {inner}u)",
                    width * inner
                );
                writeln!(out, "acc = {};\n}}", combine("acc", &a.load(*x, &ix))).unwrap();
                if tree {
                    let off = r.shared.offset(scratch(*id)).unwrap();
                    writeln!(out,"local_mem[{off}u + lane] = acc;\nworkgroupBarrier();\nfor (var step = {}u; step > 0u; step /= 2u) {{\nif (lane < step) {{",BLOCK/2).unwrap();
                    let left = format!("local_mem[{off}u + lane]");
                    let right = format!("local_mem[{off}u + lane + step]");
                    writeln!(
                        out,
                        "{left} = {};\n}}\nworkgroupBarrier();\n}}\nif (lane == 0u) {{",
                        combine(&left, &right)
                    )
                    .unwrap();
                    a.store(&mut out, *id, "at", &format!("local_mem[{off}u]"));
                    out.push_str("}\nworkgroupBarrier();\n");
                } else {
                    a.store(&mut out, *id, "at", "acc");
                }
                out.push_str("}\nworkgroupBarrier();\n");
            }
            _ => unreachable!("views and inputs do not emit stages"),
        }
        out.push_str("}\n");
    }
    out.push_str("}\n");
    out
}
