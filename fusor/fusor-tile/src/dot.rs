//! Algebra over an unrolled dot product. Storage formats and lane assignments
//! are deliberately absent: callers supply ordinary scalar expressions.
use crate::build::Kernel;
use fusor_ir::dtype::NumericContract;
use fusor_ir::ir::kernel::{ScalarElement, TileBinaryOp as Op, TileExpr, TileExprKind as K};

mod fields;

/// Accumulate products, extracting factors shared across the unrolled window.
/// Strict arithmetic keeps its original left-to-right evaluation. Expansion is
/// bounded; an oversized or non-f32 window keeps the original expression too.
pub fn accumulate(
    b: &Kernel,
    initial: TileExpr,
    products: &[(TileExpr, TileExpr)],
    numeric: NumericContract,
) -> TileExpr {
    let original = || {
        products.iter().fold(initial.clone(), |sum, (a, c)| {
            let product = b.binary(Op::Mul, a.clone(), c.clone(), numeric);
            b.binary(Op::Add, sum, product, numeric)
        })
    };
    if !numeric.reassoc
        || numeric.min_accum_bits > 32
        || numeric.min_operand_bits > 32
        || initial.element() != ScalarElement::F32.element()
        || products.is_empty()
        || products.len() > 64
        || products
            .iter()
            .any(|(a, c)| a.element() != initial.element() || c.element() != initial.element())
    {
        return original();
    }
    let mut terms = Vec::new();
    for (a, c) in products {
        let product = b.binary(Op::Mul, a.clone(), c.clone(), numeric);
        let Some(expanded) = expand(&product, 128 - terms.len()) else {
            return original();
        };
        terms.extend(expanded);
    }
    let candidate = b.binary(
        Op::Add,
        initial.clone(),
        factor(b, terms.clone(), numeric, 0),
        numeric,
    );
    let original = original();
    // Expansion can increase work when a fused pre-expression has no repeated
    // factor. Keep the original unless the shared DAG actually gets smaller.
    fn work(expr: &TileExpr) -> usize {
        fn visit(expr: &TileExpr, seen: &mut rustc_hash::FxHashSet<TileExpr>) {
            if !seen.insert(expr.clone()) {
                return;
            }
            expr.kind().visit_children(&mut |e| visit(e, seen));
        }
        let mut seen = rustc_hash::FxHashSet::default();
        visit(expr, &mut seen);
        seen.iter()
            .filter(|e| matches!(e.kind(), K::Binary { .. } | K::Unary { .. } | K::Dot { .. }))
            .count()
    }
    if work(&candidate) < work(&original) {
        if fields::pack_casts(b, &mut terms, numeric) {
            let mut expanded = Vec::new();
            for term in terms {
                let value = term
                    .factors
                    .into_iter()
                    .reduce(|a, c| b.binary(Op::Mul, a, c, numeric))
                    .unwrap_or_else(|| b.f32(1.));
                let Some(mut parts) = expand(&value, 128 - expanded.len()) else {
                    return candidate;
                };
                for part in &mut parts {
                    part.negative ^= term.negative;
                }
                expanded.extend(parts);
            }
            b.binary(Op::Add, initial, factor(b, expanded, numeric, 0), numeric)
        } else {
            candidate
        }
    } else {
        original
    }
}

#[derive(Clone)]
struct Term {
    negative: bool,
    factors: Vec<TileExpr>,
}

fn expand(e: &TileExpr, budget: usize) -> Option<Vec<Term>> {
    if budget == 0 {
        return None;
    }
    if let K::Binary {
        op,
        left,
        right,
        numeric,
    } = e.kind()
        && numeric.reassoc
        && e.element() == ScalarElement::F32.element()
    {
        match op {
            Op::Add | Op::Sub => {
                let mut a = expand(left, budget)?;
                let mut c = expand(right, budget.checked_sub(a.len())?)?;
                if *op == Op::Sub {
                    for t in &mut c {
                        t.negative = !t.negative;
                    }
                }
                a.extend(c);
                return Some(a);
            }
            Op::Mul => {
                let a = expand(left, budget)?;
                let c = expand(right, budget)?;
                if a.len().checked_mul(c.len())? > budget {
                    return None;
                }
                return Some(
                    a.iter()
                        .flat_map(|a| {
                            c.iter().map(move |c| {
                                let mut factors = a.factors.clone();
                                factors.extend(c.factors.iter().cloned());
                                Term {
                                    negative: a.negative != c.negative,
                                    factors,
                                }
                            })
                        })
                        .collect(),
                );
            }
            _ => {}
        }
    }
    Some(vec![Term {
        negative: false,
        factors: vec![e.clone()],
    }])
}

fn factor(b: &Kernel, terms: Vec<Term>, numeric: NumericContract, depth: usize) -> TileExpr {
    // A shared negative coefficient negates one sum, not every element.
    // Keeping the sign outside is essential to the operation-count benefit.
    if !terms.is_empty() && terms.iter().all(|term| term.negative) {
        let positive = terms
            .into_iter()
            .map(|mut term| {
                term.negative = false;
                term
            })
            .collect();
        return b.binary(
            Op::Sub,
            b.f32(0.),
            factor(b, positive, numeric, depth + 1),
            numeric,
        );
    }
    // Count a factor once per term, preserving encounter order on ties.
    let mut counts = Vec::<(TileExpr, usize)>::new();
    if depth < 16 {
        for term in &terms {
            for (i, value) in term.factors.iter().enumerate() {
                if term.factors[..i].contains(value) {
                    continue;
                }
                if let Some((_, n)) = counts.iter_mut().find(|(v, _)| v == value) {
                    *n += 1;
                } else {
                    counts.push((value.clone(), 1));
                }
            }
        }
    }
    let best = counts
        .into_iter()
        .rev()
        .filter(|(_, n)| *n > 1)
        .max_by_key(|(_, n)| *n);
    if let Some((common, _)) = best {
        let (mut shared, rest): (Vec<_>, Vec<_>) = terms
            .into_iter()
            .partition(|term| term.factors.contains(&common));
        for term in &mut shared {
            term.factors
                .remove(term.factors.iter().position(|v| v == &common).unwrap());
        }
        let product = b.binary(
            Op::Mul,
            common,
            factor(b, shared, numeric, depth + 1),
            numeric,
        );
        return if rest.is_empty() {
            product
        } else {
            b.binary(
                Op::Add,
                product,
                factor(b, rest, numeric, depth + 1),
                numeric,
            )
        };
    }
    // Keep short independent products explicit as vectors for backend dot
    // lowering. A dot may contract its multiply/adds, so require both freedoms.
    if numeric.contract
        && terms.len() >= 2
        && terms.iter().all(|t| !t.negative && t.factors.len() == 2)
    {
        return terms
            .chunks(4)
            .map(|chunk| {
                if chunk.len() == 1 {
                    return b.binary(
                        Op::Mul,
                        chunk[0].factors[0].clone(),
                        chunk[0].factors[1].clone(),
                        numeric,
                    );
                }
                let vector = |i: usize| {
                    b.vector(
                        ScalarElement::F32,
                        chunk.iter().map(|t| t.factors[i].clone()).collect(),
                    )
                };
                b.dot(vector(0), vector(1))
            })
            .reduce(|a, c| b.binary(Op::Add, a, c, numeric))
            .unwrap();
    }
    terms
        .into_iter()
        .map(|term| {
            let product = term
                .factors
                .into_iter()
                .reduce(|a, c| b.binary(Op::Mul, a, c, numeric))
                .unwrap_or_else(|| b.f32(1.));
            if term.negative {
                b.binary(Op::Sub, b.f32(0.), product, numeric)
            } else {
                product
            }
        })
        .reduce(|a, c| b.binary(Op::Add, a, c, numeric))
        .unwrap_or_else(|| b.f32(0.))
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor_ir::ir::kernel::TileLiteral;
    fn eval(e: &TileExpr, values: &[f32]) -> f32 {
        match e.kind() {
            K::Literal(TileLiteral::F32(bits)) => f32::from_bits(*bits),
            K::LoadLocal(local) => values[local.id() as usize - 1],
            K::Binary {
                op, left, right, ..
            } => {
                let (a, b) = (eval(left, values), eval(right, values));
                match op {
                    Op::Add => a + b,
                    Op::Sub => a - b,
                    Op::Mul => a * b,
                    _ => panic!("unexpected op"),
                }
            }
            K::Dot { left, right } => {
                let (K::Vec { parts: a, .. }, K::Vec { parts: c, .. }) =
                    (left.kind(), right.kind())
                else {
                    panic!("vector dot");
                };
                a.iter()
                    .zip(c)
                    .map(|(a, c)| eval(a, values) * eval(c, values))
                    .sum()
            }
            _ => panic!("unexpected expression"),
        }
    }
    fn multiplies(e: &TileExpr) -> usize {
        let mut n = usize::from(matches!(e.kind(), K::Binary { op: Op::Mul, .. }));
        e.kind().visit_children(&mut |c| n += multiplies(c));
        n
    }
    #[test]
    fn common_scales_and_offsets_are_factored_without_format_metadata() {
        fusor_ir::ir::kernel::reset_decl_ids();
        let b = Kernel::new();
        let locals: Vec<_> = (0..18)
            .map(|_| b.load_local(b.local(ScalarElement::F32.element())))
            .collect();
        let pairs: Vec<_> = (0..8)
            .map(|i| {
                (
                    locals[i].clone(),
                    b.sub(
                        b.mul(locals[8 + i].clone(), locals[16].clone()),
                        locals[17].clone(),
                    ),
                )
            })
            .collect();
        let before = accumulate(&b, b.f32(0.), &pairs, NumericContract::STRICT);
        let after = accumulate(&b, b.f32(0.), &pairs, NumericContract::RELAXED);
        assert!(multiplies(&after) < multiplies(&before));
        for seed in 0..50 {
            let values: Vec<_> = (0..18)
                .map(|i| ((i * 13 + seed * 7) % 53) as f32 / 19. - 1.)
                .collect();
            let a = eval(&before, &values);
            let c = eval(&after, &values);
            assert!((a - c).abs() <= 2e-5 * (1. + a.abs()), "{a} != {c}");
        }
    }
    #[test]
    fn strict_inner_products_are_not_distributed() {
        let b = Kernel::new();
        let scale = b.load_local(b.local(ScalarElement::F32.element()));
        let value = b.binary(Op::Mul, b.f32(3.), scale, NumericContract::STRICT);
        let pairs = vec![(b.f32(2.), value.clone()), (b.f32(5.), value.clone())];
        let after = accumulate(&b, b.f32(0.), &pairs, NumericContract::RELAXED);
        // The strict product remains a single opaque factor.
        fn contains(e: &TileExpr, target: &TileExpr) -> bool {
            let mut hit = e == target;
            e.kind().visit_children(&mut |c| hit |= contains(c, target));
            hit
        }
        assert!(contains(&after, &value));
        let strict = accumulate(&b, b.f32(0.), &pairs, NumericContract::STRICT);
        assert!(matches!(
            strict.kind(),
            K::Binary {
                numeric: NumericContract { reassoc: false, .. },
                ..
            }
        ));
    }

    #[test]
    fn relaxed_products_form_short_vector_dots() {
        fusor_ir::ir::kernel::reset_decl_ids();
        let b = Kernel::new();
        let locals: Vec<_> = (0..14)
            .map(|_| b.load_local(b.local(ScalarElement::F32.element())))
            .collect();
        let pairs: Vec<_> = locals[..7]
            .iter()
            .cloned()
            .zip(locals[7..].iter().cloned())
            .collect();
        let after = accumulate(&b, b.f32(0.), &pairs, NumericContract::RELAXED);
        fn dots(e: &TileExpr) -> usize {
            let mut n = usize::from(matches!(e.kind(), K::Dot { .. }));
            e.kind().visit_children(&mut |c| n += dots(c));
            n
        }
        assert_eq!(dots(&after), 2);
        let strict = accumulate(&b, b.f32(0.), &pairs, NumericContract::STRICT);
        assert_eq!(dots(&strict), 0);
        let values: Vec<_> = (0..14).map(|i| (i as f32 - 6.) / 8.).collect();
        assert_eq!(eval(&after, &values), eval(&strict, &values));
    }
}
