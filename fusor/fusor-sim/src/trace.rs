use serde::{Deserialize, Serialize};

pub const TRACE_VERSION: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Choice {
    pub workgroup: u32,
    pub boundary: String,
    pub runnable: Vec<u32>,
    pub runnable_digest: u64,
    pub lane: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScheduleTrace {
    pub version: u32,
    pub program_digest: u64,
    pub input_digest: u64,
    pub seed: u64,
    pub choices: Vec<Choice>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TraceError {
    Version(u32),
    ProgramMismatch,
    InputMismatch,
    Truncated { at: usize },
    Extra { remaining: usize },
    RunnableMismatch { at: usize },
    ImpossibleChoice { at: usize, lane: u32 },
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Version(v) => write!(f, "unsupported trace version {v}"),
            Self::ProgramMismatch => f.write_str("trace KernelIr digest mismatch"),
            Self::InputMismatch => f.write_str("trace input/config digest mismatch"),
            Self::Truncated { at } => write!(f, "trace ended before choice {at}"),
            Self::Extra { remaining } => write!(f, "trace has {remaining} trailing choices"),
            Self::RunnableMismatch { at } => {
                write!(f, "trace runnable set changed at choice {at}")
            }
            Self::ImpossibleChoice { at, lane } => {
                write!(f, "trace chose non-runnable lane {lane} at choice {at}")
            }
        }
    }
}

impl std::error::Error for TraceError {}

#[derive(Clone, Debug)]
pub(crate) enum ScheduleMode<'a> {
    Seed(u64),
    Replay(&'a ScheduleTrace),
}

pub(crate) struct Scheduler<'a> {
    mode: ScheduleMode<'a>,
    trace: ScheduleTrace,
    cursor: usize,
    rng: SplitMix64,
}

impl<'a> Scheduler<'a> {
    pub(crate) fn new(
        mode: ScheduleMode<'a>,
        program_digest: u64,
        input_digest: u64,
    ) -> Result<Self, TraceError> {
        let seed = match &mode {
            ScheduleMode::Seed(seed) => *seed,
            ScheduleMode::Replay(trace) => {
                if trace.version != TRACE_VERSION {
                    return Err(TraceError::Version(trace.version));
                }
                if trace.program_digest != program_digest {
                    return Err(TraceError::ProgramMismatch);
                }
                if trace.input_digest != input_digest {
                    return Err(TraceError::InputMismatch);
                }
                trace.seed
            }
        };
        Ok(Self {
            mode,
            trace: ScheduleTrace {
                version: TRACE_VERSION,
                program_digest,
                input_digest,
                seed,
                choices: Vec::new(),
            },
            cursor: 0,
            rng: SplitMix64::new(seed),
        })
    }

    pub(crate) fn choose(
        &mut self,
        workgroup: u32,
        boundary: &str,
        runnable: &[u32],
    ) -> Result<u32, TraceError> {
        let digest = runnable_digest(workgroup, boundary, runnable);
        let lane = match &self.mode {
            ScheduleMode::Seed(_) => runnable[self.rng.next() as usize % runnable.len()],
            ScheduleMode::Replay(trace) => {
                let Some(expected) = trace.choices.get(self.cursor) else {
                    return Err(TraceError::Truncated { at: self.cursor });
                };
                if expected.workgroup != workgroup
                    || expected.boundary != boundary
                    || expected.runnable != runnable
                    || expected.runnable_digest != digest
                {
                    return Err(TraceError::RunnableMismatch { at: self.cursor });
                }
                if !runnable.contains(&expected.lane) {
                    return Err(TraceError::ImpossibleChoice {
                        at: self.cursor,
                        lane: expected.lane,
                    });
                }
                expected.lane
            }
        };
        self.trace.choices.push(Choice {
            workgroup,
            boundary: boundary.to_owned(),
            runnable: runnable.to_vec(),
            runnable_digest: digest,
            lane,
        });
        self.cursor += 1;
        Ok(lane)
    }

    pub(crate) fn finish(self) -> Result<ScheduleTrace, TraceError> {
        if let ScheduleMode::Replay(trace) = self.mode {
            let remaining = trace.choices.len().saturating_sub(self.cursor);
            if remaining != 0 {
                return Err(TraceError::Extra { remaining });
            }
        }
        Ok(self.trace)
    }
}

pub(crate) fn stable_digest(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn runnable_digest(workgroup: u32, boundary: &str, runnable: &[u32]) -> u64 {
    let mut bytes = Vec::with_capacity(8 + boundary.len() + runnable.len() * 4);
    bytes.extend_from_slice(&workgroup.to_le_bytes());
    bytes.extend_from_slice(&(boundary.len() as u32).to_le_bytes());
    bytes.extend_from_slice(boundary.as_bytes());
    for lane in runnable {
        bytes.extend_from_slice(&lane.to_le_bytes());
    }
    stable_digest(&bytes)
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        value ^ (value >> 31)
    }
}
