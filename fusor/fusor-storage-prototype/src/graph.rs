//! THROWAWAY: logical values have shapes and indexing semantics, never storage layouts.
pub type Id = usize;

#[derive(Clone, Copy, Debug)]
pub enum Point {
    Square,
    Neg,
    Sqrt,
    Exp,
    Add,
    Sub,
    Div,
}
impl Point {
    pub fn eval(self, x: &[f32]) -> f32 {
        match self {
            Self::Square => x[0] * x[0],
            Self::Neg => -x[0],
            Self::Sqrt => x[0].sqrt(),
            Self::Exp => x[0].exp(),
            Self::Add => x[0] + x[1],
            Self::Sub => x[0] - x[1],
            Self::Div => x[0] / x[1],
        }
    }
    pub fn wgsl(self, x: &[String]) -> String {
        match self {
            Self::Square => format!("({0} * {0})", x[0]),
            Self::Neg => format!("(-{})", x[0]),
            Self::Sqrt => format!("sqrt({})", x[0]),
            Self::Exp => format!("exp({})", x[0]),
            Self::Add => format!("({} + {})", x[0], x[1]),
            Self::Sub => format!("({} - {})", x[0], x[1]),
            Self::Div => format!("({} / {})", x[0], x[1]),
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub enum Reduce {
    Sum,
    Max,
}
#[derive(Clone, Debug)]
pub enum View {
    Reshape,
    Permute(Vec<usize>),
    Broadcast,
}
#[derive(Clone, Debug)]
pub enum Op {
    Input,
    Point(Point, Vec<Id>),
    View(View, Id),
    Reduce(Reduce, Id, usize),
}
#[derive(Clone, Debug)]
pub struct Value {
    pub name: String,
    pub shape: Vec<usize>,
    pub op: Op,
}
impl Value {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }
}
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub name: String,
    pub values: Vec<Value>,
    pub roots: Vec<Id>,
}
impl Graph {
    fn add(&mut self, name: &str, shape: Vec<usize>, op: Op) -> Id {
        assert!(shape.iter().all(|d| *d > 0));
        let id = self.values.len();
        self.values.push(Value {
            name: name.into(),
            shape,
            op,
        });
        id
    }
    fn input(&mut self, shape: &[usize]) -> Id {
        self.add("input", shape.to_vec(), Op::Input)
    }
    fn point(&mut self, name: &str, op: Point, xs: &[Id]) -> Id {
        let shape = self.values[xs[0]].shape.clone();
        assert!(xs.iter().all(|x| self.values[*x].shape == shape));
        self.add(name, shape, Op::Point(op, xs.to_vec()))
    }
    fn reshape(&mut self, x: Id, shape: &[usize]) -> Id {
        assert_eq!(self.values[x].len(), shape.iter().product::<usize>());
        self.add("reshape", shape.to_vec(), Op::View(View::Reshape, x))
    }
    fn transpose(&mut self, x: Id) -> Id {
        assert_eq!(self.values[x].shape.len(), 2);
        let shape = self.values[x].shape.iter().rev().copied().collect();
        self.add("transpose", shape, Op::View(View::Permute(vec![1, 0]), x))
    }
    fn broadcast(&mut self, x: Id, shape: &[usize]) -> Id {
        assert_eq!(self.values[x].shape.len(), shape.len());
        self.add("broadcast", shape.to_vec(), Op::View(View::Broadcast, x))
    }
    fn reduce(&mut self, name: &str, kind: Reduce, x: Id, axis: usize) -> Id {
        let mut shape = self.values[x].shape.clone();
        shape.remove(axis);
        self.add(name, shape, Op::Reduce(kind, x, axis))
    }
    pub fn stages(&self) -> Vec<Id> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(id, v)| matches!(v.op, Op::Point(..) | Op::Reduce(..)).then_some(id))
            .collect()
    }
    pub fn base(&self, mut id: Id) -> Id {
        while let Op::View(_, x) = self.values[id].op {
            id = x;
        }
        id
    }
    /// Cheap scalar producers with one compute consumer need no storage or
    /// stage. Requested outputs and shared producers retain materialization.
    pub fn forwardable(&self) -> Vec<Id> {
        let stages = self.stages();
        stages
            .iter()
            .copied()
            .filter(|id| {
                let cheap = match self.values[*id].op {
                    Op::Point(..) => true,
                    Op::Reduce(_, x, axis) => self.values[x].shape[axis] <= 8,
                    _ => false,
                };
                cheap
                    && !self.roots.iter().any(|r| self.base(*r) == *id)
                    && stages
                        .iter()
                        .filter(|s| self.inputs(**s).iter().any(|x| self.base(*x) == *id))
                        .count()
                        == 1
            })
            .collect()
    }
    pub fn stage_dependencies(&self, id: Id, forwarded: &[Id]) -> Vec<Id> {
        let mut out = vec![];
        for x in self.inputs(id) {
            let base = self.base(x);
            if forwarded.contains(&base) {
                out.extend(self.stage_dependencies(base, forwarded));
            } else {
                out.push(base);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }
    pub fn stage_reads(&self, id: Id, i: usize, forwarded: &[Id]) -> Vec<(Id, usize)> {
        self.read_indices(id, i)
            .into_iter()
            .flat_map(|(src, j)| {
                if forwarded.contains(&src) {
                    self.stage_reads(src, j, forwarded)
                } else {
                    vec![(src, j)]
                }
            })
            .collect()
    }
    pub fn inputs(&self, id: Id) -> Vec<Id> {
        match &self.values[id].op {
            Op::Input => vec![],
            Op::Point(_, xs) => xs.clone(),
            Op::View(_, x) | Op::Reduce(_, x, _) => vec![*x],
        }
    }
    /// A logical element address. Views compose here, before storage exists.
    pub fn resolve_index(&self, id: Id, i: usize) -> (Id, usize) {
        match &self.values[id].op {
            Op::View(view, x) => {
                let src = &self.values[*x].shape;
                let dst = &self.values[id].shape;
                let j = match view {
                    View::Reshape => i,
                    View::Permute(axes) => axes
                        .iter()
                        .enumerate()
                        .map(|(a, b)| coord(i, dst, a) * stride(src, *b))
                        .sum(),
                    View::Broadcast => src
                        .iter()
                        .enumerate()
                        .map(|(a, d)| {
                            if *d == 1 {
                                0
                            } else {
                                coord(i, dst, a) * stride(src, a)
                            }
                        })
                        .sum(),
                };
                self.resolve_index(*x, j)
            }
            _ => (id, i),
        }
    }
    pub fn resolve_expr(&self, id: Id, i: &str) -> (Id, String) {
        match &self.values[id].op {
            Op::View(view, x) => {
                let src = &self.values[*x].shape;
                let dst = &self.values[id].shape;
                let terms = match view {
                    View::Reshape => return self.resolve_expr(*x, i),
                    View::Permute(axes) => axes
                        .iter()
                        .enumerate()
                        .map(|(a, b)| {
                            format!(
                                "((({i}) / {}u) % {}u) * {}u",
                                stride(dst, a),
                                dst[a],
                                stride(src, *b)
                            )
                        })
                        .collect::<Vec<_>>(),
                    View::Broadcast => src
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d != 1)
                        .map(|(a, _)| {
                            format!(
                                "((({i}) / {}u) % {}u) * {}u",
                                stride(dst, a),
                                dst[a],
                                stride(src, a)
                            )
                        })
                        .collect(),
                };
                self.resolve_expr(
                    *x,
                    &if terms.is_empty() {
                        "0u".into()
                    } else {
                        format!("({})", terms.join(" + "))
                    },
                )
            }
            _ => (id, i.into()),
        }
    }
    pub fn read_indices(&self, id: Id, i: usize) -> Vec<(Id, usize)> {
        match &self.values[id].op {
            Op::Point(_, xs) => xs.iter().map(|x| self.resolve_index(*x, i)).collect(),
            Op::Reduce(_, x, axis) => {
                let shape = &self.values[*x].shape;
                let inner = stride(shape, *axis);
                (0..shape[*axis])
                    .map(|r| {
                        self.resolve_index(
                            *x,
                            (i / inner) * shape[*axis] * inner + r * inner + i % inner,
                        )
                    })
                    .collect()
            }
            _ => vec![],
        }
    }
    pub fn input_data(&self, seed: usize) -> Vec<(Id, Vec<f32>)> {
        self.values
            .iter()
            .enumerate()
            .filter(|(_, v)| matches!(v.op, Op::Input))
            .map(|(id, v)| {
                (
                    id,
                    (0..v.len())
                        .map(|i| (((i * 17 + seed * 13) % 101) as f32 - 50.0) / 60.0)
                        .collect(),
                )
            })
            .collect()
    }
    /// Independent eager execution: every logical operation materializes a dense Vec.
    pub fn reference(&self, seed: usize) -> Vec<Vec<f32>> {
        let inputs = self.input_data(seed);
        let mut all: Vec<Vec<f32>> = vec![];
        for (id, v) in self.values.iter().enumerate() {
            let data = match &v.op {
                Op::Input => inputs.iter().find(|(x, _)| *x == id).unwrap().1.clone(),
                Op::View(view, x) => (0..v.len())
                    .map(|i| {
                        let src = &self.values[*x].shape;
                        let j = match view {
                            View::Reshape => i,
                            View::Permute(axes) => axes
                                .iter()
                                .enumerate()
                                .map(|(a, b)| coord(i, &v.shape, a) * stride(src, *b))
                                .sum(),
                            View::Broadcast => src
                                .iter()
                                .enumerate()
                                .map(|(a, d)| {
                                    if *d == 1 {
                                        0
                                    } else {
                                        coord(i, &v.shape, a) * stride(src, a)
                                    }
                                })
                                .sum(),
                        };
                        all[*x][j]
                    })
                    .collect(),
                Op::Point(op, xs) => (0..v.len())
                    .map(|i| op.eval(&xs.iter().map(|x| all[*x][i]).collect::<Vec<_>>()))
                    .collect(),
                Op::Reduce(kind, x, axis) => {
                    let shape = &self.values[*x].shape;
                    let inner = stride(shape, *axis);
                    (0..v.len())
                        .map(|i| {
                            (0..shape[*axis])
                                .map(|r| {
                                    all[*x]
                                        [(i / inner) * shape[*axis] * inner + r * inner + i % inner]
                                })
                                .fold(
                                    match kind {
                                        Reduce::Sum => 0.0,
                                        Reduce::Max => f32::NEG_INFINITY,
                                    },
                                    |a, b| match kind {
                                        Reduce::Sum => a + b,
                                        Reduce::Max => a.max(b),
                                    },
                                )
                        })
                        .collect()
                }
            };
            all.push(data);
        }
        self.roots.iter().map(|r| all[*r].clone()).collect()
    }
}
pub fn stride(shape: &[usize], axis: usize) -> usize {
    shape[axis + 1..].iter().product()
}
fn coord(i: usize, shape: &[usize], axis: usize) -> usize {
    i / stride(shape, axis) % shape[axis]
}
pub const CASES: &[&str] = &[
    "reshape_reduce",
    "softmax",
    "fanout",
    "transpose",
    "global_reduce",
    "reuse",
];
pub fn case(name: &str, rows: usize, cols: usize) -> Graph {
    assert!(cols >= 2 && cols % 2 == 0 && rows > 0);
    let mut g = Graph {
        name: name.into(),
        ..Graph::default()
    };
    let x = g.input(&[rows, cols]);
    let out = match name {
        "reshape_reduce" => {
            let a = g.point("square", Point::Square, &[x]);
            let v = g.reshape(a, &[rows, cols / 2, 2]);
            let b = g.reduce("pair_sum", Reduce::Sum, v, 2);
            let c = g.point("sqrt", Point::Sqrt, &[b]);
            g.reduce("row_sum", Reduce::Sum, c, 1)
        }
        "softmax" => {
            let m = g.reduce("row_max", Reduce::Max, x, 1);
            let v = g.reshape(m, &[rows, 1]);
            let b = g.broadcast(v, &[rows, cols]);
            let c = g.point("shift", Point::Sub, &[x, b]);
            let e = g.point("exp", Point::Exp, &[c]);
            let s = g.reduce("row_sum", Reduce::Sum, e, 1);
            let v = g.reshape(s, &[rows, 1]);
            let b = g.broadcast(v, &[rows, cols]);
            g.point("normalize", Point::Div, &[e, b])
        }
        "fanout" => {
            let a = g.point("square", Point::Square, &[x]);
            let b = g.point("neg", Point::Neg, &[a]);
            let c = g.point("sqrt", Point::Sqrt, &[a]);
            let d = g.point("join", Point::Add, &[b, c]);
            g.roots.push(a);
            g.reduce("row_sum", Reduce::Sum, d, 1)
        }
        "transpose" => {
            let a = g.point("square", Point::Square, &[x]);
            let v = g.transpose(a);
            let b = g.point("neg", Point::Neg, &[v]);
            g.reduce("row_sum", Reduce::Sum, b, 1)
        }
        "global_reduce" => {
            let a = g.point("square", Point::Square, &[x]);
            let b = g.reduce("row_sum", Reduce::Sum, a, 1);
            g.reduce("global_sum", Reduce::Sum, b, 0)
        }
        "reuse" => {
            let mut a = x;
            for i in 0..5 {
                a = g.transpose(a);
                a = g.point(&format!("neg_{i}"), Point::Neg, &[a]);
            }
            a
        }
        _ => panic!("unknown case {name}"),
    };
    g.roots.push(out);
    g
}
