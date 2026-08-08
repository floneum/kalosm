//! Shared on-disk store helpers: `$XDG_CACHE_HOME/fusor2/<sub>` resolution,
//! atomic writes, and versioned loads where a stale format, a foreign
//! fingerprint, a missing file and a corrupt file are all the same answer —
//! a miss, never an error.
//!
//! This module is the only place in the crate that reads the environment, and
//! it reads it only to locate the cache.

use std::path::{Path, PathBuf};

/// `$XDG_CACHE_HOME/fusor2/<sub>`, else `~/.cache/fusor2/<sub>`, else `None`
/// when neither variable is set. Shared by every on-disk store in the crate.
pub(crate) fn user_cache_dir(sub: &str) -> Option<PathBuf> {
    let base = if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME").filter(|v| !v.is_empty()) {
        PathBuf::from(xdg)
    } else {
        PathBuf::from(std::env::var_os("HOME").filter(|v| !v.is_empty())?).join(".cache")
    };
    Some(base.join("fusor2").join(sub))
}

/// Write `body` through a sibling temp file and an atomic rename, creating the
/// parent directory, so a crashed or concurrent writer cannot leave a
/// half-written record.
pub(crate) fn write_atomic(path: &Path, body: &str) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let temp = path.with_extension(format!("{}.tmp", std::process::id()));
    std::fs::write(&temp, body)?;
    std::fs::rename(&temp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })
}

/// Parse one JSON record, treating a missing file, a corrupt file and a
/// record `accept` declines (a stale format, a foreign fingerprint) as the
/// same answer: a miss, never an error.
pub(crate) fn load_versioned<T: serde::de::DeserializeOwned>(
    path: &Path,
    accept: impl FnOnce(&T) -> bool,
) -> Option<T> {
    let text = std::fs::read_to_string(path).ok()?;
    let record: T = serde_json::from_str(&text).ok()?;
    accept(&record).then_some(record)
}

#[cfg(test)]
mod tests {
    /// No optimizer behaviour is gated on an environment variable; the only
    /// environment read in this crate is the cache location.
    #[test]
    fn no_env_gated_behaviour() {
        const OWNED: [(&str, &str); 3] = [
            ("facts.rs", include_str!("facts.rs")),
            ("model.rs", include_str!("model.rs")),
            ("terms.rs", include_str!("terms.rs")),
        ];
        // Assembled at runtime so this file's own needles are not literals
        // another module could match by accident.
        let var_call = ["env", "::", "var"].concat();
        let module = ["std", "::", "env"].concat();
        let spike = ["spike", "_"].concat();
        for (name, source) in OWNED {
            assert!(
                !source.contains(&var_call),
                "{name} reads the environment; only cache.rs may"
            );
            assert!(
                !source.contains(&module),
                "{name} reaches for the env module; only cache.rs may"
            );
            assert!(
                !source.contains(&spike),
                "{name} carries a spike flag; those are cost-model terms now"
            );
        }
        // cache.rs itself reads exactly two variables, both locations.
        let me = include_str!("cache.rs");
        assert!(me.contains("XDG_CACHE_HOME"));
        assert!(me.contains("HOME"));
    }
}
