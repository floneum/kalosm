//! Immutable host backing for uploads; a view can retain a read-only file map.
use std::{fmt, ops::Range, sync::Arc};

use crate::{Error, Result};

enum Backing {
    Owned(Vec<u8>),
    Shared(Arc<dyn AsRef<[u8]> + Send + Sync>),
    Segments(Vec<Arc<HostBytes>>),
}

/// Stable host bytes for an external leaf. Sharing a file-backed range avoids
/// keeping a private CPU copy of weights after uploading their GPU copy.
pub struct HostBytes {
    backing: Backing,
    range: Range<usize>,
}

impl HostBytes {
    /// The owner must keep its bytes unchanged for this view's lifetime.
    pub fn shared<T: AsRef<[u8]> + Send + Sync + 'static>(
        owner: Arc<T>,
        range: Range<usize>,
    ) -> Result<Self> {
        if range.start > range.end || range.end > owner.as_ref().as_ref().len() {
            return Err(Error::Shape("host byte view is outside its backing".into()));
        }
        Ok(Self {
            backing: Backing::Shared(owner),
            range,
        })
    }

    /// Concatenate views without retaining a second copy of their bytes.
    pub fn concat(parts: Vec<Arc<Self>>) -> Result<Self> {
        let len = parts
            .iter()
            .try_fold(0usize, |n, part| n.checked_add(part.len()))
            .ok_or_else(|| Error::Shape("host byte concatenation overflow".into()))?;
        Ok(Self {
            backing: Backing::Segments(parts),
            range: 0..len,
        })
    }

    pub fn len(&self) -> usize {
        self.range.len()
    }
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Borrow contiguous backing, or materialize a segmented matrix just for
    /// this call. Uploading many concatenated weights retains no host copies.
    pub fn with_slice<T>(&self, f: impl FnOnce(&[u8]) -> T) -> T {
        let bytes = match &self.backing {
            Backing::Owned(bytes) => bytes.as_slice(),
            Backing::Shared(owner) => owner.as_ref().as_ref(),
            Backing::Segments(parts) => {
                let mut bytes = Vec::with_capacity(self.len());
                for part in parts {
                    part.append_to(&mut bytes);
                }
                return f(&bytes);
            }
        };
        f(&bytes[self.range.clone()])
    }

    fn append_to(&self, out: &mut Vec<u8>) {
        match &self.backing {
            Backing::Owned(bytes) => out.extend_from_slice(&bytes[self.range.clone()]),
            Backing::Shared(owner) => {
                out.extend_from_slice(&owner.as_ref().as_ref()[self.range.clone()])
            }
            Backing::Segments(parts) => {
                for part in parts {
                    part.append_to(out);
                }
            }
        }
    }
}

impl From<Vec<u8>> for HostBytes {
    fn from(bytes: Vec<u8>) -> Self {
        let len = bytes.len();
        Self {
            backing: Backing::Owned(bytes),
            range: 0..len,
        }
    }
}

impl fmt::Debug for HostBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostBytes")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_range_keeps_owner_alive_without_copying() {
        let source = Arc::new(vec![1, 2, 3, 4]);
        let pointer = source[1..].as_ptr();
        let view = HostBytes::shared(source.clone(), 1..3).unwrap();
        view.with_slice(|bytes| assert_eq!(bytes.as_ptr(), pointer));
        drop(source);
        view.with_slice(|bytes| assert_eq!(bytes, &[2, 3]));
        assert!(HostBytes::shared(Arc::new(vec![0]), 0..2).is_err());
    }

    #[test]
    fn segmented_views_retain_sources_without_a_second_allocation() {
        let owner = Arc::new(vec![1, 2, 3, 4]);
        let first = Arc::new(HostBytes::shared(owner.clone(), 0..2).unwrap());
        let second = Arc::new(HostBytes::shared(owner.clone(), 2..4).unwrap());
        let joined = HostBytes::concat(vec![second, first]).unwrap();
        assert_eq!(Arc::strong_count(&owner), 3);
        drop(owner);
        assert_eq!(joined.len(), 4);
        for _ in 0..2 {
            joined.with_slice(|bytes| assert_eq!(bytes, &[3, 4, 1, 2]));
        }
    }
}
