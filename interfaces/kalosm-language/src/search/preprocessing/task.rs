use kalosm_language_model::{ChatModel, CreateChatSession, Embedder, StructuredChatModel};
use kalosm_sample::{IndexParser, LiteralParser, Parser, ParserExt, StopOn};

use crate::{
    prelude::{Document, Task},
    search::Chunk,
};

use super::{ChunkStrategy, Chunker};

const SUMMARY_TASK_DESCRIPTION: &str = "You generate summaries of the given text.";

const HYPOTHETICAL_TASK_DESCRIPTION: &str =
    "You generate hypothetical questions that may be answered by the given text. The questions restate any information necessary to understand the question";

const HYPOTHETICAL_EXAMPLES: [(&str, &str); 2] = [
    (
        "A content delivery network or a CDN optimizes the distribution of web content by strategically placing servers worldwide. This reduces latency, accelerates content delivery, and enhances the overall user experience.",
        "What role does a content delivery network play in web performance?",
    ),
    (
        "The Internet of Things or IoT connects everyday devices to the internet, enabling them to send and receive data. This connectivity enhances automation and allows for more efficient monitoring and control of various systems.",
        "What is the purpose of the Internet of Things?",
    ),
];

const QUESTION_STARTERS: [&str; 9] = [
    "Who", "What", "When", "Where", "Why", "How", "Which", "Whom", "Whose",
];

const HYPOTHETICAL_PREFIX: &str = "Questions that are answered by the previous text: ";

/// Constraints for summary generation.
pub type SummaryConstraints = kalosm_sample::SequenceParser<LiteralParser, StopOn<&'static str>>;

/// Constraints for hypothetical question generation.
pub type HypotheticalConstraints = kalosm_sample::SequenceParser<
    LiteralParser,
    kalosm_sample::RepeatParser<
        kalosm_sample::SequenceParser<IndexParser<LiteralParser>, StopOn<&'static str>>,
    >,
>;

fn create_summary_constraints() -> SummaryConstraints {
    LiteralParser::new("One sentence summary: ").then(
        StopOn::new(".")
            .filter_characters(|c| matches!(c, ' ' | '.' | 'a'..='z' | 'A'..='Z' | '0'..='9' | ','))
            .with_length(1..=300),
    )
}

fn create_hypothetical_constraints() -> HypotheticalConstraints {
    LiteralParser::new(HYPOTHETICAL_PREFIX).then(
        IndexParser::new(
            QUESTION_STARTERS
                .iter()
                .copied()
                .map(LiteralParser::new)
                .collect::<Vec<_>>(),
        )
        .then(StopOn::new("?").with_length(1..=75).filter_characters(
            |c| matches!(c, ' ' | '?' | 'a'..='z' | 'A'..='Z' | '0'..='9' | ','),
        ))
        .repeat(1..=5),
    )
}

fn process_summary_output(output: <SummaryConstraints as Parser>::Output) -> Vec<String> {
    vec![output.1]
}

fn process_hypothetical_output(output: <HypotheticalConstraints as Parser>::Output) -> Vec<String> {
    output
        .1
        .into_iter()
        .map(|((i, _), s)| QUESTION_STARTERS[i].to_string() + &s)
        .collect()
}

fn create_summary_prompt(text: &str) -> String {
    format!("Generate a summary of the following text:\n{text}")
}

fn create_hypothetical_prompt(text: &str) -> String {
    text.to_string()
}

/// A unified chunker that runs a task on document chunks and generates embeddings.
///
/// `TaskChunker` can be configured for different strategies:
/// - **Summary**: Generates summaries of text chunks via [`TaskChunker::summary`]
/// - **Hypothetical**: Generates hypothetical questions via [`TaskChunker::hypothetical`]
///
/// # Example
///
/// ```rust,ignore
/// use kalosm_language::prelude::*;
///
/// // Create a summary chunker
/// let summarizer = TaskChunker::summary(model)
///     .with_chunking(ChunkStrategy::Paragraph { paragraph_count: 3, overlap: 1 })
///     .build();
///
/// // Create a hypothetical chunker
/// let hypothetical = TaskChunker::hypothetical(model)
///     .build();
/// ```
pub struct TaskChunker<M: CreateChatSession, C: Parser> {
    chunking: Option<ChunkStrategy>,
    task: Task<M>,
    constraints_fn: fn() -> C,
    process_output_fn: fn(C::Output) -> Vec<String>,
    prompt_fn: fn(&str) -> String,
}

/// A builder for creating a [`TaskChunker`].
pub struct TaskChunkerBuilder<M: CreateChatSession, C: Parser> {
    model: M,
    task_description: String,
    examples: Vec<(String, String)>,
    chunking: Option<ChunkStrategy>,
    constraints_fn: fn() -> C,
    process_output_fn: fn(C::Output) -> Vec<String>,
    prompt_fn: fn(&str) -> String,
}

impl<M: CreateChatSession, C: Parser> TaskChunkerBuilder<M, C> {
    /// Set the chunking strategy for splitting documents before processing.
    pub fn with_chunking(mut self, chunking: ChunkStrategy) -> Self {
        self.chunking = Some(chunking);
        self
    }

    /// Set custom examples for this task.
    ///
    /// Examples help the model perform better by allowing it to mimic the format of the examples.
    pub fn with_examples<S: Into<String>>(
        mut self,
        examples: impl IntoIterator<Item = (S, S)>,
    ) -> Self {
        self.examples = examples
            .into_iter()
            .map(|(a, b)| (a.into(), b.into()))
            .collect();
        self
    }

    /// Set a custom task description.
    pub fn with_task_description(mut self, task_description: impl Into<String>) -> Self {
        self.task_description = task_description.into();
        self
    }

    /// Build the [`TaskChunker`].
    pub fn build(self) -> TaskChunker<M, C> {
        let task = Task::new(self.model, self.task_description).with_examples(self.examples);

        TaskChunker {
            chunking: self.chunking,
            task,
            constraints_fn: self.constraints_fn,
            process_output_fn: self.process_output_fn,
            prompt_fn: self.prompt_fn,
        }
    }
}

impl<M: CreateChatSession + ChatModel> TaskChunker<M, SummaryConstraints> {
    /// Create a new summary chunker builder.
    ///
    /// The summary chunker generates summaries of text chunks and embeds them
    /// for semantic search.
    pub fn summary(model: M) -> TaskChunkerBuilder<M, SummaryConstraints> {
        TaskChunkerBuilder {
            model,
            task_description: SUMMARY_TASK_DESCRIPTION.to_string(),
            examples: vec![],
            chunking: None,
            constraints_fn: create_summary_constraints,
            process_output_fn: process_summary_output,
            prompt_fn: create_summary_prompt,
        }
    }
}

impl<M: CreateChatSession> TaskChunker<M, HypotheticalConstraints> {
    /// Create a new hypothetical chunker builder.
    ///
    /// The hypothetical chunker generates questions that might be answered by
    /// the text chunks and embeds them for semantic search.
    pub fn hypothetical(model: M) -> TaskChunkerBuilder<M, HypotheticalConstraints> {
        TaskChunkerBuilder {
            model,
            task_description: HYPOTHETICAL_TASK_DESCRIPTION.to_string(),
            examples: HYPOTHETICAL_EXAMPLES
                .iter()
                .map(|(a, b)| (a.to_string(), HYPOTHETICAL_PREFIX.to_string() + b))
                .collect(),
            chunking: None,
            constraints_fn: create_hypothetical_constraints,
            process_output_fn: process_hypothetical_output,
            prompt_fn: create_hypothetical_prompt,
        }
    }
}

impl<M, C> TaskChunker<M, C>
where
    M: CreateChatSession,
    C: Parser<Output: Send + 'static> + Clone + Send + Sync + Unpin + 'static,
{
    /// Generate outputs for the given text using the configured task.
    pub async fn generate(&self, text: &str) -> Result<Vec<String>, M::Error>
    where
        M: StructuredChatModel<C> + Send + Sync + Clone + Unpin + 'static,
        M::ChatSession: Clone + Send + Sync + Unpin + 'static,
        M::Error: Send + Sync + Unpin,
    {
        let prompt = (self.prompt_fn)(text);
        tracing::trace!(%prompt, "Running task with prompt");

        let output = self
            .task
            .run(prompt)
            .with_constraints((self.constraints_fn)())
            .await?;

        let results = (self.process_output_fn)(output);
        tracing::trace!(?results, "Task generated outputs");

        Ok(results)
    }
}

/// An error that can occur when chunking a document with [`TaskChunker`].
#[derive(Debug, thiserror::Error)]
pub enum TaskChunkerError<E1: Send + Sync + 'static, E2: Send + Sync + 'static> {
    /// An error from the text generation model.
    #[error("Text generation model error: {0}")]
    TextModelError(#[from] E1),
    /// An error from the embedding model.
    #[error("Embedding model error: {0}")]
    EmbeddingModelError(E2),
}

/// Backwards-compatible type alias for summary chunker errors.
pub type SummaryChunkerError<E1, E2> = TaskChunkerError<E1, E2>;

/// Backwards-compatible type alias for hypothetical chunker errors.
pub type HypotheticalChunkerError<E1, E2> = TaskChunkerError<E1, E2>;

impl<M, C> Chunker for TaskChunker<M, C>
where
    M: StructuredChatModel<C> + Send + Sync + Clone + Unpin + 'static,
    M::ChatSession: Clone + Send + Sync + Unpin + 'static,
    M::Error: Send + Sync + Unpin,
    C: Parser<Output: Send + 'static> + Clone + Send + Sync + Unpin + 'static,
{
    type Error<E: Send + Sync + 'static> = TaskChunkerError<M::Error, E>;

    async fn chunk<E: Embedder + Send>(
        &self,
        document: &Document,
        embedder: &E,
    ) -> Result<Vec<Chunk>, Self::Error<E::Error>> {
        let body = document.body();

        tracing::debug!(
            body_len = body.len(),
            chunking = ?self.chunking,
            "Starting task chunking for document"
        );

        #[allow(clippy::single_range_in_vec_init)]
        let byte_chunks = self
            .chunking
            .map(|chunking| chunking.chunk_str(body))
            .unwrap_or_else(|| vec![0..body.len()]);

        if byte_chunks.is_empty() {
            tracing::debug!("No chunks to process, returning empty result");
            return Ok(vec![]);
        }

        let total_chunks = byte_chunks.len();
        tracing::debug!(chunk_count = total_chunks, "Split document into chunks");

        let mut outputs = Vec::new();
        let mut output_counts = Vec::new();

        for (i, byte_chunk) in byte_chunks.iter().enumerate() {
            let text = &body[byte_chunk.clone()];
            tracing::trace!(
                chunk_index = i,
                chunk_total = total_chunks,
                chunk_len = text.len(),
                byte_range = ?byte_chunk,
                "Processing chunk"
            );

            let mut chunk_outputs = self.generate(text).await?;

            output_counts.push(chunk_outputs.len());
            outputs.append(&mut chunk_outputs);
        }

        let total_outputs = outputs.len();
        tracing::debug!(
            total_outputs = total_outputs,
            chunk_count = total_chunks,
            "Generated all outputs, starting embedding"
        );

        let mut embeddings = embedder
            .embed_vec(outputs)
            .await
            .map_err(TaskChunkerError::EmbeddingModelError)?;

        tracing::debug!(embedding_count = embeddings.len(), "Embeddings generated");

        let mut chunks = Vec::with_capacity(total_chunks);

        for (byte_chunk, output_count) in byte_chunks.iter().zip(output_counts.iter()) {
            let chunk_embeddings = embeddings.drain(0..*output_count);
            let chunk = Chunk {
                byte_range: byte_chunk.clone(),
                embeddings: chunk_embeddings.collect(),
            };
            chunks.push(chunk);
        }

        tracing::debug!(result_count = chunks.len(), "Task chunking complete");

        Ok(chunks)
    }
}

/// A chunker that generates summaries for documents.
///
/// This is a type alias for [`TaskChunker`] with [`SummaryConstraints`].
pub type Summarizer<M> = TaskChunker<M, SummaryConstraints>;

/// A builder for a summary chunker.
pub type SummarizerBuilder<M> = TaskChunkerBuilder<M, SummaryConstraints>;

/// A chunker that generates hypothetical questions for documents.
///
/// This is a type alias for [`TaskChunker`] with [`HypotheticalConstraints`].
pub type Hypothetical<M> = TaskChunker<M, HypotheticalConstraints>;

impl<M: CreateChatSession + ChatModel> Summarizer<M> {
    /// Create a new summary generator builder.
    pub fn builder(model: M) -> SummarizerBuilder<M> {
        TaskChunker::summary(model)
    }

    /// Generate a summary for a document.
    pub async fn generate_summary(&self, text: &str) -> Result<Vec<String>, M::Error>
    where
        M: StructuredChatModel<SummaryConstraints> + Send + Sync + Clone + Unpin + 'static,
        M::ChatSession: Clone + Send + Sync + Unpin + 'static,
        M::Error: Send + Sync + Unpin,
    {
        self.generate(text).await
    }
}

/// A builder for a hypothetical chunker.
pub type HypotheticalBuilder<M> = TaskChunkerBuilder<M, HypotheticalConstraints>;

impl<M: CreateChatSession> Hypothetical<M> {
    /// Create a new hypothetical generator builder.
    pub fn builder(model: M) -> HypotheticalBuilder<M> {
        TaskChunker::hypothetical(model)
    }

    /// Generate a list of hypothetical questions about the given text.
    pub async fn generate_question(&self, text: &str) -> Result<Vec<String>, M::Error>
    where
        M: StructuredChatModel<HypotheticalConstraints> + Send + Sync + Clone + Unpin + 'static,
        M::ChatSession: Clone + Send + Sync + Unpin + 'static,
        M::Error: Send + Sync + Unpin,
    {
        self.generate(text).await
    }
}
