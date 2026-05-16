macro_rules! id_newtype {
    ($(#[$meta:meta])* $vis:vis $name:ident $(, $derive:ident)*) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, PartialEq, Eq $(, $derive)*)]
        $vis struct $name(pub(crate) u32);

        impl $name {
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_newtype!(
    /// Identifier of a cooperatively-loaded fragment (SSA-cached).
    pub CoopFragmentId, Hash
);

id_newtype!(
    /// Identifier shared by lanes of one fused quantized-block dequant.
    pub BlockDequantId, Hash
);

id_newtype!(
    /// A storage buffer identifier.
    pub BufferId
);

id_newtype!(
    /// A private local identifier.
    pub LocalId, Hash
);

id_newtype!(
    /// A tiny tile identifier for the typed IR.
    pub TileId
);
