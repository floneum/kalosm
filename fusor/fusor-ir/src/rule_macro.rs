//! The `rule!` macro.

/// Declare a `pub const` [`crate::egraph::Rule`] named by its identifier:
/// `head = tag` or `heads = [tags]`, then either `apply = path`, or
/// `l0 = Variant { fields }` with an inline body that binds the Logical
/// head's fields by reference and returns `None` on any other variant.
#[macro_export]
macro_rules! rule {
    (
        $name:ident,
        level = $level:expr,
        head  = $head:expr,
        tag   = $tag:expr,
        apply = $apply:path $(,)?
    ) => {
        $crate::rule!($name, level = $level, heads = [$head], tag = $tag, apply = $apply);
    };

    (
        $name:ident,
        level = $level:expr,
        heads = [$($head:expr),+ $(,)?],
        tag   = $tag:expr,
        apply = $apply:path $(,)?
    ) => {
        pub const $name: $crate::egraph::Rule = $crate::egraph::Rule {
            name: stringify!($name),
            level: $level,
            heads: &[$($head),+],
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
        pub const $name: $crate::egraph::Rule = {
            fn apply(
                $b: &mut $crate::egraph::Builder<'_>,
                $id: $crate::egraph::Id,
                $node: &$crate::ir::Node,
                $f: &$crate::egraph::Facts<'_>,
            ) -> Option<$crate::egraph::Id> {
                let _ = (&*$b, $id, $f);
                let $crate::ir::Op::Logical($crate::ir::logical::Logical::$variant {
                    $($field,)* ..
                }) = &$node.op
                else {
                    return None;
                };
                $body
            }
            $crate::egraph::Rule {
                name: stringify!($name),
                level: $level,
                heads: &[$head],
                tag: $tag,
                apply,
            }
        };
    };
}
