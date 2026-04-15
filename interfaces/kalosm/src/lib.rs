#![warn(missing_docs)]
#![doc = include_str!("../README.md")]

pub use futures_util::StreamExt as _;
pub use kalosm_streams::timed_stream::*;

#[cfg(feature = "language")]
pub mod language {
    #![doc = include_str!("../docs/language.md")]
    pub use kalosm_language::context::*;
    pub use kalosm_language::kalosm_language_model::{
        ChatModel as _, ChatModelExt as _, ChatSession as _, CreateChatSession as _,
        CreateDefaultChatConstraintsForType as _, CreateDefaultCompletionConstraintsForType as _,
        CreateTextCompletionSession as _, Embedder as _, EmbedderCacheExt as _, EmbedderExt as _,
        IntoChatMessage as _, IntoEmbedding as _, ModelConstraints as _, StreamExt as _,
        StructuredChatModel as _, StructuredTextCompletionModel as _, TextCompletionModel as _,
        TextCompletionModelExt as _, TextCompletionSession as _, *,
    };
    #[cfg(feature = "llama")]
    pub use kalosm_language::kalosm_llama::{
        Device, Llama, LlamaBuilder, LlamaChatSession, LlamaSession, LlamaSource,
    };
    pub use kalosm_language::kalosm_sample::{self, *};
    pub use kalosm_language::prelude::Html;
    #[cfg(feature = "bert")]
    pub use kalosm_language::rbert::{Bert, BertBuilder, BertSource};
    pub use kalosm_language::search::*;
    pub use kalosm_language::vector_db::*;
    pub use kalosm_model_types::{
        FileLoadingProgress, FileSource, ModelBuilder as _, ModelLoadingProgress,
    };
    pub use kalosm_streams::text_stream::*;

    #[cfg(feature = "surrealdb")]
    pub use crate::surrealdb_integration::document_table::*;
    #[cfg(feature = "surrealdb")]
    pub use crate::surrealdb_integration::hybrid_search::*;
}
#[cfg(feature = "sound")]
pub mod sound {
    #![doc = include_str!("../docs/sound.md")]
    pub use futures_util::StreamExt as _;
    pub use kalosm_sound::*;
    pub use kalosm_streams::text_stream::*;
    pub use kalosm_streams::timed_stream::*;
}
#[cfg(feature = "vision")]
pub mod vision {
    #![doc = include_str!("../docs/vision.md")]
    pub use futures_util::StreamExt as _;
    pub use kalosm_vision::*;
}

#[cfg(feature = "language")]
mod evaluate;
#[cfg(feature = "language")]
pub use evaluate::*;

#[cfg(feature = "prompt_annealing")]
mod prompt_annealing;
#[cfg(feature = "prompt_annealing")]
pub use prompt_annealing::*;

#[cfg(feature = "surrealdb")]
mod surrealdb_integration;
#[cfg(feature = "surrealdb")]
pub use ::surrealdb;
#[cfg(feature = "surrealdb")]
pub use surrealdb_integration::*;
