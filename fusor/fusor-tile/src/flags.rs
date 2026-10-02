//! Measurement switches read from `FUSOR_*` environment variables, parsed once.

use std::sync::OnceLock;

pub(crate) struct Flags {
    /// `FUSOR_PIN_COOP="bm,bn,bk[,..]"`: restrict the coop domain to one geometry.
    pub pin_coop: Option<Vec<u32>>,
    /// `FUSOR_PIN_SGEMV="vector,subgroups,cols[,parts,gap]"`: restrict the
    /// sgemv domain to one cell.
    pub pin_sgemv: Option<Vec<u32>>,
}

pub(crate) fn flags() -> &'static Flags {
    static FLAGS: OnceLock<Flags> = OnceLock::new();
    FLAGS.get_or_init(|| {
        let list = |name| {
            std::env::var(name).ok().map(|v| {
                v.split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect::<Vec<u32>>()
            })
        };
        Flags {
            pin_coop: list("FUSOR_PIN_COOP"),
            pin_sgemv: list("FUSOR_PIN_SGEMV"),
        }
    })
}
