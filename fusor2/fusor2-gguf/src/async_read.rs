//! The byte-range source an [`crate::sharded::AsyncShardedVarBuilder`] reads
//! through. Deliberately runtime-agnostic: one boxed future, no executor
//! dependency anywhere in the compiler stack.

use fusor2_ir::Result;
use fusor2_ir::error::Error;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::parse::GgufMetadata;

/// A future returned by [`AsyncReadRange::read_range`].
pub type ReadFuture<'a> = Pin<Box<dyn Future<Output = Result<Vec<u8>>> + Send + 'a>>;

/// Something that can serve `len` bytes starting at `start`.
pub trait AsyncReadRange: Send + Sync {
    fn read_range(&self, start: u64, len: usize) -> ReadFuture<'_>;

    /// Total length, when known. `None` forces the caller to trust the GGUF
    /// directory instead.
    fn len(&self) -> Option<u64> {
        None
    }

    /// `true` when the source is known to be empty. Present because clippy
    /// asks for it next to `len`; a source of unknown length is not empty.
    fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

/// An in-memory source, for tests and for callers that already hold the file.
pub struct BytesRange(pub Arc<[u8]>);

impl AsyncReadRange for BytesRange {
    fn read_range(&self, start: u64, len: usize) -> ReadFuture<'_> {
        let start = start as usize;
        let end = start.saturating_add(len).min(self.0.len());
        let slice = if start >= self.0.len() {
            Vec::new()
        } else {
            self.0[start..end].to_vec()
        };
        Box::pin(async move { Ok(slice) })
    }

    fn len(&self) -> Option<u64> {
        Some(self.0.len() as u64)
    }
}

/// Smallest prefix we ask for when hunting for the end of a GGUF header.
const HEADER_PROBE_BYTES: usize = 1 << 16;
/// Largest prefix we will pull before giving up. A GGUF header is metadata
/// plus a tensor directory; 64 MiB covers a tokenizer vocabulary with room to
/// spare, and refusing past that keeps a corrupt file from pulling the whole
/// model over the wire.
const HEADER_LIMIT_BYTES: usize = 64 << 20;

/// Read just enough of `source` to parse its header.
///
/// The header's length is not known until it has been parsed, so this pulls a
/// prefix and doubles it while the parse runs off the end. Each attempt is one
/// range request; a well-formed file needs exactly one.
pub async fn read_metadata(source: &dyn AsyncReadRange) -> Result<GgufMetadata> {
    let total = source.len();
    let mut want = HEADER_PROBE_BYTES;
    loop {
        let capped = match total {
            Some(n) => want.min(n as usize),
            None => want,
        };
        let bytes = source.read_range(0, capped).await?;
        let mut cursor = std::io::Cursor::new(&bytes[..]);
        match GgufMetadata::read(&mut cursor) {
            Ok(meta) => return Ok(meta),
            Err(e) => {
                let exhausted = total.is_some_and(|n| capped as u64 >= n) || bytes.len() < capped;
                if exhausted || want >= HEADER_LIMIT_BYTES {
                    return Err(e);
                }
                want = (want * 4).min(HEADER_LIMIT_BYTES);
            }
        }
    }
}

/// A minimal executor for callers that have none — the sync-over-async escape
/// hatch the sharded loader's tests use. Spins; only sensible for futures that
/// never actually yield, which is every future this module produces from an
/// in-memory source.
pub fn block_on<F: Future>(future: F) -> F::Output {
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    const VTABLE: RawWakerVTable = RawWakerVTable::new(
        |_| RawWaker::new(std::ptr::null(), &VTABLE),
        |_| {},
        |_| {},
        |_| {},
    );
    // SAFETY: the vtable's clone/wake/drop are all no-ops over a null data
    // pointer, so the waker never dereferences anything.
    let waker = unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) };
    let mut cx = Context::from_waker(&waker);
    let mut future = Box::pin(future);
    loop {
        match future.as_mut().poll(&mut cx) {
            Poll::Ready(v) => return v,
            Poll::Pending => std::hint::spin_loop(),
        }
    }
}

/// Turn a short read into a typed error, so a truncated shard reports the
/// tensor it failed on rather than panicking on a slice.
pub(crate) fn expect_len(bytes: Vec<u8>, want: usize, what: &str) -> Result<Vec<u8>> {
    if bytes.len() == want {
        Ok(bytes)
    } else {
        Err(Error::Io(format!(
            "{what}: wanted {want} bytes, the source served {}",
            bytes.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_range_serves_clamped_slices() {
        let src = BytesRange(Arc::from(vec![1u8, 2, 3, 4].into_boxed_slice()));
        assert_eq!(src.len(), Some(4));
        assert!(!src.is_empty());
        assert_eq!(block_on(src.read_range(1, 2)).unwrap(), vec![2, 3]);
        assert_eq!(block_on(src.read_range(2, 99)).unwrap(), vec![3, 4]);
        assert!(block_on(src.read_range(9, 4)).unwrap().is_empty());
    }
}
