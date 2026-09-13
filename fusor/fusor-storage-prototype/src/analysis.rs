//! Symbolic access recipes, with a deliberately independent finite audit mode.
use crate::graph::{Graph, Id, Op};
use crate::index::{Bounds, Expr, GROUP, LOCAL, OUTPUT};

#[derive(Clone, Debug)]
pub struct Read {
    pub source: Id,
    pub index: Expr,
    pub loops: Vec<(usize, usize)>,
}
impl Read {
    pub fn count(&self) -> usize {
        self.loops.iter().map(|(_, n)| *n).product()
    }
    fn owned(&self, output_len: usize, source_len: usize, groups: usize) -> bool {
        if groups == 1 {
            return true;
        }
        let share = output_len / groups;
        let mut bounds: Bounds = self
            .loops
            .iter()
            .map(|(v, n)| (*v, (*n - 1) as u64))
            .collect();
        bounds.insert(GROUP, (groups - 1) as u64);
        bounds.insert(LOCAL, (share - 1) as u64);
        let owner = self
            .index
            .substitute(
                OUTPUT,
                &Expr::sum([Expr::var(GROUP).scale(share), Expr::var(LOCAL)]),
            )
            .div(source_len / groups)
            .simplify(&bounds);
        owner == Expr::var(GROUP)
    }
    fn enumerate(&self, output: usize, visit: &mut impl FnMut(usize)) {
        fn recurse(
            read: &Read,
            depth: usize,
            values: &mut Vec<usize>,
            output: usize,
            visit: &mut impl FnMut(usize),
        ) {
            if depth == read.loops.len() {
                visit(read.index.eval(&|v| {
                    if v == OUTPUT {
                        output
                    } else {
                        values[read.loops.iter().position(|(id, _)| *id == v).unwrap()]
                    }
                }));
            } else {
                for i in 0..read.loops[depth].1 {
                    values.push(i);
                    recurse(read, depth + 1, values, output, visit);
                    values.pop();
                }
            }
        }
        recurse(self, 0, &mut vec![], output, visit);
    }
}
pub struct Analysis {
    pub dependencies: Vec<Vec<Id>>,
    reads: Vec<Vec<Read>>,
    pub recipes: usize,
    pub audit: bool,
}
impl Analysis {
    pub fn new(g: &Graph, stages: &[Id], forwarded: &[Id], audit: bool) -> Self {
        fn visit(
            g: &Graph,
            id: Id,
            index: Expr,
            loops: Vec<(usize, usize)>,
            forwarded: &[Id],
            computing: bool,
            out: &mut Vec<Read>,
        ) {
            let (base, index) = g.index_map(id, index);
            if !computing && !forwarded.contains(&base) {
                out.push(Read {
                    source: base,
                    index,
                    loops,
                });
                return;
            }
            match &g.values[base].op {
                Op::Point(_, xs) => {
                    for x in xs {
                        visit(g, *x, index.clone(), loops.clone(), forwarded, false, out);
                    }
                }
                Op::Reduce(_, x, axis) => {
                    let var = loops.len() + 1;
                    let index = g.reduction_index(*x, *axis, index, Expr::var(var));
                    let mut loops = loops;
                    loops.push((var, g.values[*x].shape[*axis]));
                    visit(g, *x, index, loops, forwarded, false, out);
                }
                _ => unreachable!("only compute stages are expanded"),
            }
        }
        let mut out = Self {
            dependencies: vec![vec![]; g.values.len()],
            reads: vec![vec![]; g.values.len()],
            recipes: 0,
            audit,
        };
        for id in stages {
            visit(
                g,
                *id,
                Expr::var(OUTPUT),
                vec![],
                forwarded,
                true,
                &mut out.reads[*id],
            );
            out.recipes += out.reads[*id].len();
            out.dependencies[*id] = out.reads[*id].iter().map(|r| r.source).collect();
            out.dependencies[*id].sort_unstable();
            out.dependencies[*id].dedup();
            if audit {
                assert!(
                    g.values[*id].len() * out.reads[*id].iter().map(Read::count).sum::<usize>()
                        <= 2_000_000,
                    "finite audit is limited to two million reads per stage"
                );
                for i in 0..g.values[*id].len() {
                    let mut actual = vec![];
                    for r in &out.reads[*id] {
                        r.enumerate(i, &mut |j| actual.push((r.source, j)));
                    }
                    let mut expected = g.stage_reads(*id, i, forwarded);
                    actual.sort_unstable();
                    expected.sort_unstable();
                    assert_eq!(
                        actual, expected,
                        "symbolic index map differs from finite reference"
                    );
                }
            }
        }
        out
    }
    pub fn external_reads(&self, g: &Graph, members: &[Id]) -> usize {
        members
            .iter()
            .map(|id| {
                g.values[*id].len()
                    * self.reads[*id]
                        .iter()
                        .filter(|r| !members.contains(&r.source))
                        .map(Read::count)
                        .sum::<usize>()
            })
            .sum()
    }
    pub fn owned(&self, g: &Graph, members: &[Id], groups: usize) -> bool {
        let symbolic = members.iter().all(|id| {
            self.reads[*id]
                .iter()
                .filter(|r| members.contains(&r.source))
                .all(|r| r.owned(g.values[*id].len(), g.values[r.source].len(), groups))
        });
        if self.audit {
            let mut finite = true;
            for id in members {
                for i in 0..g.values[*id].len() {
                    for r in &self.reads[*id] {
                        if members.contains(&r.source) {
                            r.enumerate(i, &mut |j| {
                                finite &= i / (g.values[*id].len() / groups)
                                    == j / (g.values[r.source].len() / groups);
                            });
                        }
                    }
                }
            }
            assert!(
                !symbolic || finite,
                "symbolic ownership accepted an unsafe region"
            );
        }
        // Unproven mappings are rejected conservatively; no tensor-size
        // dependent fallback is hidden in normal compilation.
        symbolic
    }
}
