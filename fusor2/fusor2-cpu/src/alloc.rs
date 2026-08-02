//! 64-byte-aligned buffers, so a `f32x16` load is never split across a cache
//! line and a workgroup tile can be addressed as a raw byte arena.
//!
//! Owned by W10.

use fusor2_ir::error::Error;
use fusor2_ir::Result;
use std::alloc::{alloc_zeroed, dealloc, Layout};

/// A 64-byte-aligned byte buffer.
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `AlignedBuf` owns its allocation exclusively; there is no interior
// mutability and no shared aliasing, so moving it across threads is sound.
unsafe impl Send for AlignedBuf {}
// SAFETY: `&AlignedBuf` only exposes read-only access to owned bytes, plus
// [`AlignedBuf::as_mut_ptr`], whose contract is documented there.
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub const ALIGN: usize = 64;

    pub fn zeroed(len: usize) -> Result<Self> {
        if len == 0 {
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
            });
        }
        let layout = Layout::from_size_align(len, Self::ALIGN)
            .map_err(|e| Error::Device(format!("bad allocation layout: {e}")))?;
        // SAFETY: `layout` has a non-zero size.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return Err(Error::Device(format!("out of memory allocating {len} B")));
        }
        Ok(Self { ptr, len })
    }

    /// Grow in place-ish: reallocate zeroed when `len` exceeds the current
    /// capacity, otherwise keep the existing allocation.
    pub fn ensure(&mut self, len: usize) -> Result<()> {
        if len <= self.len {
            return Ok(());
        }
        *self = Self::zeroed(len)?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `ptr` is a live allocation of `len` initialized bytes.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: `&mut self` proves exclusivity over a live allocation.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// The aliasing escape hatch the launcher needs.
    ///
    /// A dispatch hands the same `&AlignedBuf` to every worker thread and each
    /// writes its own disjoint slice. Disjointness is not a promise this type
    /// can check; it is `verify_l1`'s invariant 3 — *a nest's write map must be
    /// injective unless the nest declares an associative combine* — which the
    /// planner has already discharged before a kernel reaches [`crate::launch`].
    /// Every cross-lane accumulation instead goes through
    /// [`crate::emit::Program`]'s private-accumulate-then-merge path.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.ptr
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        if self.len == 0 || self.ptr.is_null() {
            return;
        }
        // SAFETY: same size and alignment `zeroed` allocated with.
        unsafe {
            let layout = Layout::from_size_align_unchecked(self.len, Self::ALIGN);
            dealloc(self.ptr, layout);
        }
    }
}

impl std::fmt::Debug for AlignedBuf {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AlignedBuf({} B @ {:p})", self.len, self.ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_and_zeroed() {
        let b = AlignedBuf::zeroed(4096).unwrap();
        assert_eq!(b.as_ptr() as usize % AlignedBuf::ALIGN, 0);
        assert!(b.as_slice().iter().all(|&x| x == 0));
    }

    #[test]
    fn empty_is_safe() {
        let b = AlignedBuf::zeroed(0).unwrap();
        assert!(b.is_empty());
        assert!(b.as_slice().is_empty());
    }
}
