//! `EinSpec` algebra: which labels are contracted, which are batch, and the
//! `(m, n, k, batch)` a `Contract` lowering reads off a spec plus two shapes.
//!
//! `matmul`, `mat_mul_transposed_rhs` and every batched form differ only in
//! the spec.

use crate::error::{Error, Result};
use crate::ir::logical::{EinSpec, Label};
use crate::shape::{Dim, Dims};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// What a label does in a contraction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LabelRole {
    /// In `a`, `b` and `out`.
    Batch,
    /// In `a` and `out`: a free axis of the left operand.
    M,
    /// In `b` and `out`: a free axis of the right operand.
    N,
    /// In `a` and `b` but not `out`: summed.
    K,
}

/// Labels grouped by role, each group in the order the spec writes them
/// (`out` order for `Batch`/`M`/`N`, `a` order for `K`), so `out_shape` and
/// the `mnkb` products agree with the node's declared layout.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EinPartition {
    pub batch: SmallVec<[Label; 4]>,
    pub m: SmallVec<[Label; 4]>,
    pub n: SmallVec<[Label; 4]>,
    pub k: SmallVec<[Label; 4]>,
}

impl EinPartition {
    /// Every label, once, in `batch ++ m ++ n ++ k` order.
    pub fn labels(&self) -> SmallVec<[Label; 8]> {
        self.batch
            .iter()
            .chain(&self.m)
            .chain(&self.n)
            .chain(&self.k)
            .copied()
            .collect()
    }
}

/// Role of one label. A label appearing in fewer than two of `{a, b, out}`
/// has no contraction meaning and is `Error::Shape`.
pub fn role(spec: &EinSpec, l: Label) -> Result<LabelRole> {
    let in_a = spec.a.contains(&l);
    let in_b = spec.b.contains(&l);
    let in_out = spec.out.contains(&l);
    Ok(match (in_a, in_b, in_out) {
        (true, true, true) => LabelRole::Batch,
        (true, false, true) => LabelRole::M,
        (false, true, true) => LabelRole::N,
        (true, true, false) => LabelRole::K,
        _ => {
            return Err(Error::Shape(format!(
                "label {} appears in fewer than two of {{a, b, out}}",
                l.0
            )));
        }
    })
}

/// Partition every label of `spec` by role. A repeated label inside one
/// operand list (a diagonal, which no contraction kernel expresses) is
/// `Error::Shape`.
pub fn partition(spec: &EinSpec) -> Result<EinPartition> {
    for (name, list) in [("a", &spec.a), ("b", &spec.b), ("out", &spec.out)] {
        for (i, l) in list.iter().enumerate() {
            if list[..i].contains(l) {
                return Err(Error::Shape(format!(
                    "label {} is repeated in operand list {name}",
                    l.0
                )));
            }
        }
    }

    let mut part = EinPartition::default();
    // Batch/M/N in `out` order so the output shape reads off directly.
    for &l in &spec.out {
        match role(spec, l)? {
            LabelRole::Batch => part.batch.push(l),
            LabelRole::M => part.m.push(l),
            LabelRole::N => part.n.push(l),
            // A label present in `out` is never classified K.
            LabelRole::K => {
                return Err(Error::Shape(format!(
                    "label {} is both summed and produced",
                    l.0
                )));
            }
        }
    }
    // K in `a` order.
    for &l in &spec.a {
        if role(spec, l)? == LabelRole::K {
            part.k.push(l);
        }
    }
    // Every label of `b` must also have a role; catches a label only in `b`.
    for &l in &spec.b {
        role(spec, l)?;
    }
    Ok(part)
}

/// Bind every label to an extent by zipping each operand list positionally
/// with that operand's shape. A label bound twice must be [`Dim::known_eq`]
/// both times — a symbolic and a constant extent are not decidably equal
/// and are rejected.
pub fn extents(spec: &EinSpec, a: &[Dim], b: &[Dim]) -> Result<FxHashMap<Label, Dim>> {
    let mut map: FxHashMap<Label, Dim> = FxHashMap::default();
    for (name, labels, shape) in [("a", &spec.a, a), ("b", &spec.b, b)] {
        if labels.len() != shape.len() {
            return Err(Error::Shape(format!(
                "contraction operand {name} has rank {} but its spec names {} labels",
                shape.len(),
                labels.len()
            )));
        }
        for (&l, &d) in labels.iter().zip(shape) {
            match map.get(&l) {
                Some(prev) if !prev.known_eq(d) => {
                    return Err(Error::Shape(format!(
                        "contracted extent disagreement on label {}: {prev} vs {d}",
                        l.0
                    )));
                }
                _ => {
                    map.insert(l, d);
                }
            }
        }
    }
    for &l in &spec.out {
        if !map.contains_key(&l) {
            return Err(Error::Shape(format!(
                "output label {} is bound by neither operand",
                l.0
            )));
        }
    }
    Ok(map)
}

/// Output shape: `spec.out` mapped through `extents`.
pub fn out_shape(spec: &EinSpec, extents: &FxHashMap<Label, Dim>) -> Result<Dims> {
    spec.out
        .iter()
        .map(|l| {
            extents
                .get(l)
                .copied()
                .ok_or_else(|| Error::Shape(format!("output label {} has no extent", l.0)))
        })
        .collect()
}

/// `[m, n, k, batch]`, each the collapsed product of its label group.
///
/// Product rule: drop `Const(1)`; all-`Const` ⇒ `Const(product)`; exactly one
/// surviving `Sym` and nothing else ⇒ that `Sym`; empty group ⇒ `Const(1)`;
/// two or more non-collapsible survivors ⇒ `Error::Shape`.
pub fn mnkb(spec: &EinSpec, extents: &FxHashMap<Label, Dim>) -> Result<[Dim; 4]> {
    let part = partition(spec)?;
    Ok([
        collapse(&part.m, extents, "m")?,
        collapse(&part.n, extents, "n")?,
        collapse(&part.k, extents, "k")?,
        collapse(&part.batch, extents, "batch")?,
    ])
}

fn collapse(group: &[Label], extents: &FxHashMap<Label, Dim>, name: &str) -> Result<Dim> {
    let mut product: u64 = 1;
    let mut symbolic: Option<Dim> = None;
    let mut extra_symbols = 0usize;
    for l in group {
        let d = extents
            .get(l)
            .copied()
            .ok_or_else(|| Error::Shape(format!("label {} has no extent", l.0)))?;
        match d {
            Dim::Const(1) => {}
            Dim::Const(v) => {
                product = product
                    .checked_mul(v)
                    .ok_or_else(|| Error::Shape(format!("{name} group extent overflows u64")))?;
            }
            Dim::Sym(_) => {
                if symbolic.is_some() {
                    extra_symbols += 1;
                } else {
                    symbolic = Some(d);
                }
            }
        }
    }
    match symbolic {
        None => Ok(Dim::Const(product)),
        Some(s) if extra_symbols == 0 && product == 1 => Ok(s),
        Some(_) => Err(Error::Shape(format!(
            "symbolic contraction group is not collapsible ({name})"
        ))),
    }
}

/// Assert both adjoint specs of `spec` are themselves well-formed
/// contractions, and that `d_lhs` really maps `out x b -> a` with the
/// original's contracted set becoming `a`'s free set.
pub fn check_adjoint_specs(spec: &EinSpec) -> Result<()> {
    let original = partition(spec)?;

    let d_lhs = spec.d_lhs();
    let d_rhs = spec.d_rhs();
    partition(&d_lhs).map_err(|e| {
        Error::Shape(format!(
            "d_lhs of this contraction is not a contraction: {e}"
        ))
    })?;
    partition(&d_rhs).map_err(|e| {
        Error::Shape(format!(
            "d_rhs of this contraction is not a contraction: {e}"
        ))
    })?;

    // `d_lhs` is `out x b -> a`. Every label the original summed is free in
    // `a` and read from `b`, i.e. an N label of the adjoint.
    for &l in &original.k {
        let r = role(&d_lhs, l)?;
        if r != LabelRole::N {
            return Err(Error::Shape(format!(
                "contracted label {} is {r:?} in d_lhs; it must be a free axis of `a`",
                l.0
            )));
        }
    }
    // Symmetrically for `d_rhs`, which is `a x out -> b`.
    for &l in &original.k {
        let r = role(&d_rhs, l)?;
        if r != LabelRole::M {
            return Err(Error::Shape(format!(
                "contracted label {} is {r:?} in d_rhs; it must be a free axis of `b`",
                l.0
            )));
        }
    }
    Ok(())
}

/// `verify_l0` clause 4's structural half: every label appears in >= 2 of
/// `{a, b, out}`, and no operand repeats a label.
pub fn verify_spec(spec: &EinSpec) -> Result<()> {
    partition(spec).map(|_| ())
}

/// `(batch, m, n, k)` for a `Contract` lowering, from two operand shapes.
pub fn mnk(spec: &EinSpec, a: &[Dim], b: &[Dim]) -> Result<(Dim, Dim, Dim, Dim)> {
    let [m, n, k, batch] = mnkb(spec, &extents(spec, a, b)?)?;
    Ok((batch, m, n, k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::SymId;

    fn spec(a: &[u8], b: &[u8], out: &[u8]) -> EinSpec {
        EinSpec {
            a: a.iter().map(|&c| Label(c)).collect(),
            b: b.iter().map(|&c| Label(c)).collect(),
            out: out.iter().map(|&c| Label(c)).collect(),
        }
    }

    // `bik,bjk->bij`
    fn batched() -> EinSpec {
        spec(b"bik", b"bjk", b"bij")
    }
    // `mk,nk->mn` — `mat_mul_transposed_rhs`.
    fn transposed_rhs() -> EinSpec {
        spec(b"mk", b"nk", b"mn")
    }

    #[test]
    fn roles_and_partition() {
        let s = batched();
        assert_eq!(role(&s, Label(b'b')).unwrap(), LabelRole::Batch);
        assert_eq!(role(&s, Label(b'i')).unwrap(), LabelRole::M);
        assert_eq!(role(&s, Label(b'j')).unwrap(), LabelRole::N);
        assert_eq!(role(&s, Label(b'k')).unwrap(), LabelRole::K);

        let p = partition(&s).unwrap();
        assert_eq!(&p.batch[..], &[Label(b'b')]);
        assert_eq!(&p.m[..], &[Label(b'i')]);
        assert_eq!(&p.n[..], &[Label(b'j')]);
        assert_eq!(&p.k[..], &[Label(b'k')]);
        assert_eq!(p.labels().len(), 4);
    }

    #[test]
    fn a_label_only_in_out_errors() {
        // `z` appears only in `out`.
        let s = spec(b"mk", b"nk", b"mnz");
        assert!(matches!(role(&s, Label(b'z')), Err(Error::Shape(_))));
        assert!(partition(&s).is_err());
    }

    #[test]
    fn repeated_label_in_one_operand_errors() {
        let s = spec(b"mm", b"mn", b"mn");
        assert!(matches!(partition(&s), Err(Error::Shape(_))));
    }

    #[test]
    fn extents_out_shape_and_mnkb() {
        // `bik,bjk->bij` on [2,3,4], [2,5,4].
        let s = batched();
        let a = [Dim::Const(2), Dim::Const(3), Dim::Const(4)];
        let b = [Dim::Const(2), Dim::Const(5), Dim::Const(4)];
        let e = extents(&s, &a, &b).unwrap();
        let out = out_shape(&s, &e).unwrap();
        assert_eq!(&out[..], &[Dim::Const(2), Dim::Const(3), Dim::Const(5)]);
        assert_eq!(
            mnkb(&s, &e).unwrap(),
            [Dim::Const(3), Dim::Const(5), Dim::Const(4), Dim::Const(2)]
        );
    }

    #[test]
    fn disagreeing_contracted_extents_error() {
        let s = batched();
        let a = [Dim::Const(2), Dim::Const(3), Dim::Const(4)];
        let b = [Dim::Const(2), Dim::Const(5), Dim::Const(7)];
        assert!(matches!(extents(&s, &a, &b), Err(Error::Shape(_))));

        // A symbolic and a constant extent are not decidably equal.
        let b2 = [Dim::Const(2), Dim::Const(5), Dim::Sym(SymId(0))];
        assert!(extents(&s, &a, &b2).is_err());
    }

    #[test]
    fn rank_mismatch_errors() {
        let s = batched();
        let a = [Dim::Const(2), Dim::Const(3)];
        let b = [Dim::Const(2), Dim::Const(5), Dim::Const(4)];
        assert!(matches!(extents(&s, &a, &b), Err(Error::Shape(_))));
    }

    #[test]
    fn product_rule() {
        // Two batch labels multiply; `Const(1)` drops; a lone `Sym` survives.
        let s = spec(b"pqik", b"pqjk", b"pqij");
        let a = [
            Dim::Const(2),
            Dim::Const(3),
            Dim::Const(1),
            Dim::Sym(SymId(7)),
        ];
        let b = [
            Dim::Const(2),
            Dim::Const(3),
            Dim::Const(5),
            Dim::Sym(SymId(7)),
        ];
        let e = extents(&s, &a, &b).unwrap();
        assert_eq!(
            mnkb(&s, &e).unwrap(),
            [
                Dim::Const(1),      // m: the only M label is Const(1)
                Dim::Const(5),      // n
                Dim::Sym(SymId(7)), // k
                Dim::Const(6),      // batch: 2*3
            ]
        );
    }

    #[test]
    fn two_symbolic_survivors_do_not_collapse() {
        let s = spec(b"pqik", b"pqjk", b"pqij");
        let a = [
            Dim::Sym(SymId(1)),
            Dim::Sym(SymId(2)),
            Dim::Const(3),
            Dim::Const(4),
        ];
        let b = [
            Dim::Sym(SymId(1)),
            Dim::Sym(SymId(2)),
            Dim::Const(5),
            Dim::Const(4),
        ];
        let e = extents(&s, &a, &b).unwrap();
        let err = mnkb(&s, &e).unwrap_err();
        assert!(format!("{err}").contains("not collapsible"));
    }

    #[test]
    fn adjoint_specs_accepted() {
        check_adjoint_specs(&batched()).unwrap();
        check_adjoint_specs(&transposed_rhs()).unwrap();
    }

    #[test]
    fn adjoint_specs_reject_an_unadjointable_spec() {
        // `q` lives only in `a`, so neither the primal nor its adjoints are
        // well-formed contractions.
        let broken = spec(b"mkq", b"nk", b"mn");
        assert!(check_adjoint_specs(&broken).is_err());
    }

    #[test]
    fn scaffold_spellings_agree() {
        let s = batched();
        verify_spec(&s).unwrap();
        let a = [Dim::Const(2), Dim::Const(3), Dim::Const(4)];
        let b = [Dim::Const(2), Dim::Const(5), Dim::Const(4)];
        assert_eq!(
            mnk(&s, &a, &b).unwrap(),
            (Dim::Const(2), Dim::Const(3), Dim::Const(5), Dim::Const(4))
        );
    }
}
