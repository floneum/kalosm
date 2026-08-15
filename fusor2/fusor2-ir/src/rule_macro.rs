//! The `rule!` macro. Three syntactic forms: the declarative one, which names
//! a rule and its `apply` function, and two structural-pattern forms that
//! destructure an `Op::L0(L0::…)` / `Op::L1(L1::…)` head, bind its fields by
//! name and emit an early `return None` on mismatch. **No proc macro**: four
//! of the interesting rules enumerate integer tuples and a pattern DSL would
//! not earn itself there.
//!
//! Owned by W2.

/// Declare a `pub const` [`crate::egraph::Rule`].
///
/// Three forms.
///
/// **Declarative** — the rule body is a free function elsewhere:
///
/// ```ignore
/// rule!(FOLD_SPLIT, level = Level::L0, head = OpTag::Fold,
///       tag = RuleTag::Additive, apply = fold_split);
/// ```
///
/// **Structural** — the body is inline and the head is destructured, with an
/// implicit `return None` when the node is not that variant:
///
/// ```ignore
/// rule!(UNIT_FOLD_COLLAPSE, level = Level::L0, head = OpTag::Fold,
///       tag = RuleTag::Additive,
///       l0 = Fold { combine, axis, acc, carrier, x },
///       |b, id, node, f| { … Option<Id> });
/// ```
///
/// `l1 = KFold { … }` is the same against [`crate::ir::level1::L1`]. Bound
/// fields are *references* into the node, because the driver hands the rule a
/// borrowed `&Node`.
///
/// The rule's `name` is the identifier, so a conformance case can assert it
/// fired by string without a second registry.
#[macro_export]
macro_rules! rule {
    (
        $name:ident,
        level = $level:expr,
        head  = $head:expr,
        tag   = $tag:expr,
        apply = $apply:path $(,)?
    ) => {
        pub const $name: $crate::egraph::Rule = $crate::egraph::Rule {
            name: stringify!($name),
            level: $level,
            head: $head,
            tag: $tag,
            apply: $apply,
        };
    };

    (
        $name:ident,
        level = $level:expr,
        head  = $head:expr,
        tag   = $tag:expr,
        l0 = $variant:ident { $($field:ident),* $(,)? },
        |$b:ident, $id:ident, $node:ident, $f:ident| $body:block $(,)?
    ) => {
        $crate::rule_structural!(
            $name, $level, $head, $tag,
            $crate::ir::Op::L0($crate::ir::level0::L0::$variant { $($field,)* .. }),
            |$b, $id, $node, $f| $body
        );
    };

    (
        $name:ident,
        level = $level:expr,
        head  = $head:expr,
        tag   = $tag:expr,
        l1 = $variant:ident { $($field:ident),* $(,)? },
        |$b:ident, $id:ident, $node:ident, $f:ident| $body:block $(,)?
    ) => {
        $crate::rule_structural!(
            $name, $level, $head, $tag,
            $crate::ir::Op::L1($crate::ir::level1::L1::$variant { $($field,)* .. }),
            |$b, $id, $node, $f| $body
        );
    };
}

/// Shared expansion of the two structural [`rule!`] arms. Not part of the
/// stable surface; use `rule!` instead.
#[doc(hidden)]
#[macro_export]
macro_rules! rule_structural {
    (
        $name:ident, $level:expr, $head:expr, $tag:expr,
        $pattern:pat,
        |$b:ident, $id:ident, $node:ident, $f:ident| $body:block
    ) => {
        pub const $name: $crate::egraph::Rule = {
            fn apply(
                $b: &mut $crate::egraph::Builder<'_>,
                $id: $crate::egraph::Id,
                $node: &$crate::ir::Node,
                $f: &$crate::egraph::Facts<'_>,
            ) -> Option<$crate::egraph::Id> {
                let _ = (&*$b, $id, $f);
                let $pattern = &$node.op else { return None };
                $body
            }
            $crate::egraph::Rule {
                name: stringify!($name),
                level: $level,
                head: $head,
                tag: $tag,
                apply,
            }
        };
    };
}
