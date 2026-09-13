//! Code generators ask for logical loads/stores. Only Access knows placement.
use crate::graph::{Graph, Id, Op, Reduce};
use crate::index::{Expr, OUTPUT};
use crate::plan::{BLOCK, Plan, Region, scratch, tree_fold};
use std::fmt::Write;

struct Access<'a> {
    graph: &'a Graph,
    plan: &'a Plan,
    region: &'a Region,
}
impl Access<'_> {
    fn load(&self, id: Id, index: &Expr) -> String {
        let (base, index) = self.graph.index_map(id, index.clone());
        if self.plan.forwarded.contains(&base) {
            return match &self.graph.values[base].op {
                Op::Point(op, xs) => {
                    op.wgsl(&xs.iter().map(|x| self.load(*x, &index)).collect::<Vec<_>>())
                }
                Op::Reduce(kind, x, axis) => (0..self.graph.values[*x].shape[*axis])
                    .map(|k| {
                        self.load(
                            *x,
                            &self.graph.reduction_index(
                                *x,
                                *axis,
                                index.clone(),
                                Expr::constant(k),
                            ),
                        )
                    })
                    .reduce(|a, b| match kind {
                        Reduce::Sum => format!("({a} + {b})"),
                        Reduce::Max => format!("max({a}, {b})"),
                    })
                    .unwrap(),
                _ => unreachable!("only scalar computations are forwarded"),
            };
        }
        let index = index.wgsl(&|v| if v == OUTPUT { "at".into() } else { "k".into() });
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
    fn scratch(&self, id: Id, index: &str) -> String {
        format!(
            "local_mem[{}u + ({index})]",
            self.region.shared.offset(scratch(id)).unwrap()
        )
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
    let collective = p
        .collective
        .filter(|_| r.members.iter().any(|id| tree_fold(g, *id, p.serial_limit)));
    let enable = if collective.is_some() {
        "// native subgroup collectives\n"
    } else {
        ""
    };
    let subgroup_args = if collective.is_some() {
        ", @builtin(subgroup_id) subgroup_id: u32, @builtin(subgroup_invocation_id) subgroup_lane: u32"
    } else {
        ""
    };
    let mut out = format!(
        "// THROWAWAY Fusor logical-access prototype\n{enable}@group(0) @binding(0) var<storage, read> inputs: array<f32>;\n@group(0) @binding(1) var<storage, read_write> global_mem: array<f32>;\n{shared_decl}@compute @workgroup_size({BLOCK})\nfn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32{subgroup_args}) {{\nlet group = wid.x;\n"
    );
    for id in &r.members {
        let v = &g.values[*id];
        let share = v.len() / r.groups;
        writeln!(out, "// %{} {} {:?}\n{{", id, v.name, v.shape).unwrap();
        match &v.op {
            Op::Point(op, xs) => {
                writeln!(out,"for (var i = lane; i < {share}u; i += {BLOCK}u) {{\nlet at = group * {share}u + i;").unwrap();
                let args: Vec<String> = xs.iter().map(|x| a.load(*x, &Expr::var(OUTPUT))).collect();
                writeln!(out, "let value = {};", op.wgsl(&args)).unwrap();
                a.store(&mut out, *id, "at", "value");
                out.push_str("}\nworkgroupBarrier();\n");
            }
            Op::Reduce(kind, x, axis) => {
                let shape = &g.values[*x].shape;
                let width = shape[*axis];
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
                let ix = g.reduction_index(*x, *axis, Expr::var(OUTPUT), Expr::var(1));
                writeln!(out, "acc = {};\n}}", combine("acc", &a.load(*x, &ix))).unwrap();
                if tree {
                    if let Some(recipe) = collective {
                        let address = |index: &str| a.scratch(*id, index);
                        let total = recipe
                            .emit(
                                &mut WgslCollective {
                                    out: &mut out,
                                    kind: *kind,
                                    address: &address,
                                },
                                "acc".to_string(),
                            )
                            .unwrap();
                        out.push_str("if (lane == 0u) {\n");
                        a.store(&mut out, *id, "at", &total);
                        out.push_str("}\nworkgroupBarrier();\n");
                    } else {
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
                    }
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

struct WgslCollective<'a> {
    out: &'a mut String,
    kind: Reduce,
    address: &'a dyn Fn(&str) -> String,
}
impl fusor_gpu::reduction::CollectiveEmitter for WgslCollective<'_> {
    type Value = String;
    type Error = std::convert::Infallible;
    fn subgroup(&mut self, value: String) -> Result<String, Self::Error> {
        let op = match self.kind {
            Reduce::Sum => "subgroupAdd",
            Reduce::Max => "subgroupMax",
        };
        writeln!(self.out, "let subgroup_partial = {op}({value});").unwrap();
        Ok("subgroup_partial".into())
    }
    fn barrier(&mut self) {
        self.out.push_str("workgroupBarrier();\n");
    }
    fn store_leader(&mut self, value: String) -> Result<(), Self::Error> {
        writeln!(
            self.out,
            "if (subgroup_lane == 0u) {{ {} = {value}; }}",
            (self.address)("subgroup_id")
        )
        .unwrap();
        Ok(())
    }
    fn load_partial(&mut self, index: u32) -> Result<String, Self::Error> {
        Ok((self.address)(&format!("{index}u")))
    }
    fn combine(&mut self, a: String, b: String) -> String {
        match self.kind {
            Reduce::Sum => format!("({a} + {b})"),
            Reduce::Max => format!("max({a}, {b})"),
        }
    }
}
