//! `EinSpec` algebra: label roles and the `(m, n, k, batch)` a `Contract`
//! lowering reads off a spec plus two shapes.

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

/// Labels grouped by role: `out` order for `Batch`/`M`/`N`, `a` order for `K`.
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

/// Role of one label; one in fewer than two of `{a, b, out}` is an error.
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

/// Partition every label of `spec` by role. A label repeated inside one
/// operand list (a diagonal) is an error.
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

/// Bind every label to an extent from the operand shapes; a label bound
/// twice must be [`Dim::known_eq`] both times.
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

/// `[m, n, k, batch]`, each the product of its label group: constant, or a
/// lone `Sym` beside unit extents; anything else is an error.
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

/// Assert both adjoint specs of `spec` are well-formed contractions whose
/// free axes are the original's contracted set.
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

    // `d_lhs` is `out x b -> a`: every summed label is an N label of it.
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
