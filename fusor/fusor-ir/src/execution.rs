//! Per-device execution policy. Overrides use the existing diagnostic names
//! so the normal path and performance experiments share one implementation.
//! No process environment is changed; profiles can coexist in one process.

/// Planner and runtime defaults for a device.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum ExecutionProfile {
    /// General tensor workloads and unvalidated adapters.
    #[default]
    General,
    /// Bounded-memory, low-latency inference on supported Apple GPUs.
    LowLatency,
}

impl ExecutionProfile {
    /// A diagnostic override, or this profile's built-in setting.
    pub fn var(self, name: &str) -> Result<String, std::env::VarError> {
        match std::env::var(name) {
            Err(std::env::VarError::NotPresent) => self
                .default_setting(name)
                .map(str::to_owned)
                .ok_or(std::env::VarError::NotPresent),
            value => value,
        }
    }

    /// Boolean override. `0`, `false`, `off` and `no` disable a default.
    pub fn enabled(self, name: &str) -> bool {
        self.var(name)
            .is_ok_and(|v| !matches!(v.as_str(), "0" | "false" | "off" | "no"))
    }

    /// Native defaults, kept independent of environment parsing for tests.
    pub fn default_setting(self, name: &str) -> Option<&'static str> {
        if self == Self::General {
            return None;
        }
        Some(match name {
            "PROFILE_PREFILL_WIDTH" => "64",
            "PROFILE_CONST_PREFILL_KV" => "1",
            "FUSOR_PIN_COOP" => "16,32,8,4,1,2",
            "FUSOR_PIN_COOP_MIN_M" => "16",
            "FUSOR_PIN_CONTRACT_FAMILY" => "coop",
            "FUSOR_PIN_CONTRACT_MIN_M" => "16",
            "FUSOR_PROFILE_SEED_CANDIDATES" => "0",
            "FUSOR_PROFILE_SEED_MIN_M" => "16",
            "FUSOR_PROFILE_NO_WAVEFRONT_MIN_M" => "16",
            "FUSOR_PROFILE_GEMV_PACKED_Q4" => "1",
            "FUSOR_PROFILE_DISABLE_RULES" => "SPLIT_K",
            "FUSOR_PIN_CONTRACT_SMALL_FAMILY" => "sgemv",
            "FUSOR_PIN_SMALL_DENSE_SGEMV" => "1,4,16,1,0",
            "FUSOR_PROFILE_SCATTER_SINGLE" => "1",
            "FUSOR_PROFILE_PACKED_Q4_SPLITS" => "1",
            "FUSOR_PROFILE_PACKED_Q4_SPLIT_MAX_N" => "8192",
            "FUSOR_PROFILE_PACKED_Q4_SPLIT_MAX_K" => "8192",
            "FUSOR_PROFILE_INLINE_MAP_SLAB" => "1",
            "FUSOR_PROFILE_SCATTER_SINGLE_LINEAR" => "4",
            "PROFILE_PADDED_DECODE_ATTENTION" => "1",
            "FUSOR_PROFILE_FUSED_DECODE_ATTENTION" => "1",
            "FUSOR_PROFILE_SUBMIT_SIZE" => "256",
            "FUSOR_PROFILE_NO_DEBUG_GROUPS" => "1",
            "FUSOR_PROFILE_INFLIGHT_CHUNKS" => "32",
            "FUSOR_PROFILE_ATTENTION_CHUNK" => "256",
            "PROFILE_QK_VIEW_ROPE" => "1",
            "PROFILE_QK_PROJECT" => "1",
            "FUSOR_PROFILE_FUSED_RMS_NORM" => "1",
            "FUSOR_PROFILE_INDEX_VIEW_FORWARD" => "1",
            "FUSOR_PROFILE_STREAM_SUBMIT" => "13",
            "FUSOR_PROFILE_Q6_GROUP16" => "1",
            "FUSOR_PROFILE_Q6_COALESCED" => "1",
            "FUSOR_PROFILE_FUSED_DECODE_ROPE" => "1",
            "FUSOR_PROFILE_Q6_SPLITS" => "2",
            "FUSOR_PROFILE_Q6_SPLIT_MAX_N" => "8192",
            "FUSOR_PROFILE_COMPOSITE_VIEW_FORWARD" => "1",
            "FUSOR_PROFILE_FORWARD_PASSES" => "8",
            "FUSOR_PROFILE_LINEAR_VIEW_FORWARD" => "1",
            "FUSOR_PROFILE_SKIP_INITIALIZED_WORKGROUP_ZERO" => "1",
            "FUSOR_PROFILE_RMS_BLOCK" => "512",
            "FUSOR_PROFILE_WORKGROUP_THREADS" => "1024",
            "FUSOR_PROFILE_CACHE_INDEX_UNIFORM" => "1",
            "PROFILE_PROMPT_SIZED_CACHE" => "1",
            "PROFILE_TOP_K_PREFILTER" => "1",
            "FUSOR_PROFILE_RMS_RESIDUAL" => "1",
            "FUSOR_PROFILE_CACHE_DONATION" => "1",
            "FUSOR_PROFILE_ATTENTION_ONLINE" => "1",
            "FUSOR_PROFILE_ATTENTION_BLOCK" => "512",
            "FUSOR_PROFILE_RMS_SUBGROUP_MERGE" => "1",
            "FUSOR_PROFILE_Q4_HALF_UNPACK" => "1",
            "FUSOR_COMPILE_THREADS" => "1",
            "FUSOR_PROFILE_MAPPED_UPLOAD" => "1",
            "FUSOR_PROFILE_MEMORY_BUDGET_BYTES" => "7516192768",
            "FUSOR_PROFILE_LARGE_PLAN_INFLIGHT_CHUNKS" => "1",
            "FUSOR_PROFILE_LARGE_PLAN_SUBMIT_SIZE" => "32",
            "FUSOR_PROFILE_LARGE_PLAN_MIN_DISPATCHES" => "4096",
            "FUSOR_PROFILE_POLL_SPIN_MS" => "32",
            "FUSOR_PROFILE_Q4_VARIANT" => "3",
            "FUSOR_PIN_LARGE_Q4_SGEMV" => "32,2,8,4,32",
            "FUSOR_PROFILE_STREAM_SUBMIT_REST" => "128",
            "FUSOR_PROFILE_WARM_ARTIFACTS" => "1",
            "FUSOR_PROFILE_WARM_RESOLVE" => "1",
            "FUSOR_PROFILE_ATTENTION_EXACT_PARTITION" => "1",
            "FUSOR_PIN_LARGE_Q6_SGEMV" => "32,2,8,4,32",
            "FUSOR_EXPLORE_EPS" => "0",
            _ => return None,
        })
    }
}
