//! A bounded native simulator for the GPU compiler's real [`KernelIr`].

mod interpreter;
mod trace;

pub use interpreter::{
    BindingMemory, Limits, MemoryValue, RunResult, SimError, Simulator, execute, replay,
};
pub use trace::{Choice, ScheduleTrace, TraceError};

use serde::{Deserialize, Serialize};

pub const FAILURE_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct FailureRecord {
    pub version: u32,
    pub kind: String,
    pub recipe: crate::fuzz::Recipe,
    pub inputs: Vec<Vec<f32>>,
    pub caps_profile: String,
    pub limits: Limits,
    pub candidate: String,
    pub selections: Vec<crate::fuzz::CandidateSelection>,
    pub trace: Vec<ScheduleTrace>,
    pub detail: String,
}

mod fuzz;

pub use fuzz::{
    CacheCampaignResult, CacheRecord, CampaignResult, CandidateSelection, Recipe, RecipeKind,
    cache_campaign, campaign, campaign_with_limits, minimize_failure, replay_cache_record,
    replay_failure,
};
