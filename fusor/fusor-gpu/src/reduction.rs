//! A shared collective recipe. Planning owns scratch requirements; adapters
//! supply value operations and addresses for Naga or another shader emitter.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CollectivePlan {
    width: u32,
    partials: u32,
}
impl CollectivePlan {
    /// Requires a proven fixed subgroup width and a full-workgroup reduction.
    pub fn new(block: u32, fixed_width: u32) -> Option<Self> {
        (fixed_width > 0 && block >= fixed_width && block.is_multiple_of(fixed_width)).then(|| {
            Self {
                width: fixed_width,
                partials: block / fixed_width,
            }
        })
    }
    pub fn width(self) -> u32 {
        self.width
    }
    pub fn scratch_elements(self) -> u32 {
        if self.partials == 1 { 0 } else { self.partials }
    }
    /// Every lane receives the result. Leading synchronization protects the
    /// previous users of a reused scratch slot, including a loop's back edge.
    pub fn emit<E: CollectiveEmitter>(
        self,
        emitter: &mut E,
        value: E::Value,
    ) -> Result<E::Value, E::Error> {
        let partial = emitter.subgroup(value)?;
        if self.partials == 1 {
            return Ok(partial);
        }
        emitter.barrier();
        emitter.store_leader(partial)?;
        emitter.barrier();
        let mut total = emitter.load_partial(0)?;
        for i in 1..self.partials {
            let next = emitter.load_partial(i)?;
            total = emitter.combine(total, next);
        }
        Ok(total)
    }
}

/// Addressing and backend syntax stay in the adapter. No layout is exposed by
/// the algorithm or its scratch-memory contract.
pub trait CollectiveEmitter {
    type Value;
    type Error;
    fn subgroup(&mut self, value: Self::Value) -> Result<Self::Value, Self::Error>;
    fn barrier(&mut self);
    fn store_leader(&mut self, value: Self::Value) -> Result<(), Self::Error>;
    fn load_partial(&mut self, index: u32) -> Result<Self::Value, Self::Error>;
    fn combine(&mut self, left: Self::Value, right: Self::Value) -> Self::Value;
}
