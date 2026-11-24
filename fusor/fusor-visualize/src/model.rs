use serde::{Deserialize, Serialize};

/// One shared, fully associative, LRU cache in a virtual SM. Stores are
/// write-through/no-allocate and invalidate touched lines. Zero capacity bypasses it.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct GpuConfig {
    pub subgroup_size: u32,
    pub transaction_bytes: u32,
    pub cache_line_bytes: u32,
    pub cache_bytes: u32,
    pub banks: u32,
    pub bank_width_bytes: u32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            subgroup_size: 32,
            transaction_bytes: 32,
            cache_line_bytes: 128,
            cache_bytes: 16 * 1024,
            banks: 32,
            bank_width_bytes: 4,
        }
    }
}

impl GpuConfig {
    pub fn validate(&self) -> crate::Result<()> {
        crate::require(
            (1..=1024).contains(&self.subgroup_size),
            "subgroup size must be 1..=1024",
        )?;
        for (name, value) in [
            ("transaction bytes", self.transaction_bytes),
            ("cache line bytes", self.cache_line_bytes),
            ("bank width bytes", self.bank_width_bytes),
            ("banks", self.banks),
        ] {
            crate::require(
                value.is_power_of_two() && value <= 65536,
                format!("{name} must be a power of two in 1..=65536"),
            )?;
        }
        crate::require(
            self.cache_line_bytes >= self.transaction_bytes,
            "cache line must be at least one transaction",
        )?;
        crate::require(
            self.cache_bytes.is_multiple_of(self.cache_line_bytes),
            "cache bytes must be a multiple of cache line bytes",
        )?;
        crate::require(
            self.cache_bytes <= 64 * 1024 * 1024,
            "virtual cache is limited to 64 MiB",
        )
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum Space {
    Storage,
    Workgroup,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Resource {
    pub id: u64,
    pub name: String,
    pub space: Space,
    pub bytes: u64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AccessKind {
    Read,
    Write,
    Atomic,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LaneAccess {
    pub lane: u32,
    pub address: u64,
    pub bytes: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Action {
    Memory {
        resource: u64,
        access: AccessKind,
        lanes: Vec<LaneAccess>,
    },
    Barrier {
        storage: bool,
    },
    Control {
        label: String,
        lanes: Vec<u32>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub workgroup: [u32; 3],
    pub source: String,
    pub action: Action,
}

/// A trace can be produced by the KernelIr adapter or another instrumented emitter.
/// Each memory event is one instruction across a workgroup; the simulator splits
/// it into contiguous subgroups, in ascending order, including partial subgroups.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Trace {
    pub version: u32,
    pub name: String,
    pub workgroup_size: u32,
    /// Set by the IR recorder: lane-address generation depends on subgroup width.
    /// Generic lane-address traces may leave it null and be regrouped on replay.
    #[serde(default)]
    pub recorded_subgroup_size: Option<u32>,
    pub resources: Vec<Resource>,
    pub events: Vec<Event>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct Metrics {
    pub steps: u64,
    pub requested_bytes: u64,
    pub unique_bytes: u64,
    /// Address sectors touched, regardless of cache state (not DRAM transactions).
    pub transactions: u64,
    pub ideal_transactions: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_evictions: u64,
    pub cache_fill_bytes: u64,
    pub within_subgroup_duplicate_bytes: u64,
    /// Unique read bytes seen in an earlier action in the same workgroup.
    pub same_subgroup_reuse_bytes: u64,
    pub cross_subgroup_reuse_bytes: u64,
    pub cross_workgroup_reuse_bytes: u64,
    pub bank_conflict_steps: u64,
    pub max_bank_degree: u64,
    /// Best possible rounds for this many distinct words over the configured banks.
    /// In totals, the largest such lower bound encountered.
    pub ideal_bank_degree: u64,
}

impl Metrics {
    pub(crate) fn add(&mut self, rhs: &Self) {
        macro_rules! add { ($($field:ident),*) => { $(self.$field += rhs.$field;)* }; }
        add!(
            steps,
            requested_bytes,
            unique_bytes,
            transactions,
            ideal_transactions,
            cache_hits,
            cache_misses,
            cache_evictions,
            cache_fill_bytes,
            within_subgroup_duplicate_bytes,
            same_subgroup_reuse_bytes,
            cross_subgroup_reuse_bytes,
            cross_workgroup_reuse_bytes,
            bank_conflict_steps
        );
        self.max_bank_degree = self.max_bank_degree.max(rhs.max_bank_degree);
        self.ideal_bank_degree = self.ideal_bank_degree.max(rhs.ideal_bank_degree);
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct CacheLine {
    pub resource: u64,
    pub line: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Step {
    pub index: usize,
    pub event_index: usize,
    pub workgroup: [u32; 3],
    /// None for a workgroup-wide barrier/control action.
    pub subgroup: Option<u32>,
    pub source: String,
    pub action: Action,
    pub metrics: Metrics,
    pub totals: Metrics,
    pub sectors: Vec<u64>,
    /// Bank -> distinct word addresses (read broadcasts collapse).
    pub bank_words: Vec<Vec<u64>>,
    pub evicted: Vec<CacheLine>,
    pub hints: Vec<String>,
}
