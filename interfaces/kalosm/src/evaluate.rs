use comfy_table::Cell;
use comfy_table::Row;
use comfy_table::Table;
use hdrhistogram::Histogram;
use std::fmt::Display;
use std::future::Future;
use std::ops::RangeInclusive;
use std::sync::OnceLock;

#[cfg(feature = "bert")]
use kalosm_language::prelude::Bert;
#[cfg(feature = "bert")]
use kalosm_language::prelude::Embedder;

/// A metric is a way to compare two pieces of data. It is used to evaluate the performance of a model.
pub trait Metric<T> {
    /// The range of values that this metric can return.
    const RANGE: RangeInclusive<f64> = 0.0..=1.0;

    /// Compute the distance between this piece of data and another piece of data.
    fn distance(&mut self, first: &T, other: &T) -> impl Future<Output = f64> + Send;
}

#[cfg(feature = "bert")]
/// A metric that uses the Bert model to compute the distance between two strings.
pub struct BertDistance {
    bert: Bert,
}

#[cfg(feature = "bert")]
impl BertDistance {
    /// Create a new BertDistance metric.
    pub fn new(model: Bert) -> Self {
        BertDistance { bert: model }
    }
}

#[cfg(feature = "bert")]
impl<S: ToString + Send + Sync> Metric<S> for BertDistance {
    async fn distance(&mut self, first: &S, other: &S) -> f64 {
        let embeddings = self
            .bert
            .embed_vec(vec![first.to_string(), other.to_string()])
            .await
            .expect("Failed to embed text with Bert");
        let [first_embedding, other_embedding] = embeddings
            .try_into()
            .expect("Failed to get two embeddings from the batch of two input texts from Bert");
        first_embedding.cosine_similarity(&other_embedding).into()
    }
}

/// A set of test cases to evaluate a model.
pub struct TestCases<I> {
    name: String,
    tests: Vec<TestCase<I>>,
}

impl<I> Default for TestCases<I> {
    #[track_caller]
    fn default() -> Self {
        Self::new()
    }
}

impl<I> TestCases<I> {
    /// Create a new set of test cases.
    #[track_caller]
    pub fn new() -> Self {
        TestCases {
            name: std::panic::Location::caller().to_string(),
            tests: Vec::new(),
        }
    }

    /// Set the name of this set of test cases.
    pub fn with_name(mut self, name: impl Display) -> Self {
        self.name = name.to_string();
        self
    }

    /// Add a test case to this set of test cases.
    pub fn with_case(mut self, expected: I, actual: I) -> Self {
        self.tests.push(TestCase { expected, actual });
        self
    }

    /// Push a test case to this set of test cases.
    pub fn push_case(&mut self, expected: I, actual: I) {
        self.tests.push(TestCase { expected, actual });
    }

    /// Evaluate a model using this set of test cases.
    pub async fn evaluate<M: Metric<I>>(&mut self, metric: &mut M) -> EvaluationResult<'_, I> {
        let mut values = Vec::new();
        for case in &self.tests {
            let TestCase { expected, actual } = case;
            let distance = metric.distance(expected, actual).await;
            values.push(TestCaseScored {
                case,
                score: distance,
            });
        }
        values.sort_by(|a, b| f64::total_cmp(&b.score, &a.score));
        EvaluationResult {
            name: self.name.clone(),
            histogram: OnceLock::new(),
            tests: values,
            range: M::RANGE,
        }
    }
}

/// The result of evaluating a model using a set of test cases.
#[derive(Clone)]
pub struct EvaluationResult<'a, I> {
    name: String,
    histogram: OnceLock<Histogram<u64>>,
    tests: Vec<TestCaseScored<'a, I>>,
    range: RangeInclusive<f64>,
}

impl<I> EvaluationResult<'_, I> {
    const SCALE_FACTOR: f64 = 10000.0;

    fn histogram_scale_factor(&self) -> f64 {
        let min = self.range.start();
        let max = self.range.end();
        Self::SCALE_FACTOR / (max - min)
    }

    fn scale_value(&self, value: f64) -> f64 {
        let min = self.range.start();
        let scale_factor = self.histogram_scale_factor();
        (value - min) * scale_factor
    }

    fn unscale_value(&self, value: f64) -> f64 {
        let min = self.range.start();
        let scale_factor = self.histogram_scale_factor();
        value / scale_factor + min
    }

    fn histogram(&self) -> &Histogram<u64> {
        self.histogram.get_or_init(|| {
            let mut histogram = Histogram::<u64>::new(3).unwrap();
            for test in &self.tests {
                histogram
                    .record(self.scale_value(test.score) as u64)
                    .expect("Failed to record score");
            }
            histogram
        })
    }

    /// Get the mean score of this EvaluationResult.
    pub fn mean_score(&self) -> f64 {
        self.unscale_value(self.histogram().mean())
    }

    /// Get the median score of this EvaluationResult.
    pub fn median_score(&self) -> f64 {
        self.unscale_value(self.histogram().value_at_percentile(50.0) as f64)
    }

    /// Get the minimum score of this EvaluationResult.
    pub fn min_score(&self) -> f64 {
        self.unscale_value(self.histogram().min() as f64)
    }

    /// Get the maximum score of this EvaluationResult.
    pub fn max_score(&self) -> f64 {
        self.unscale_value(self.histogram().max() as f64)
    }

    /// Get the score at a given quantile of this EvaluationResult.
    pub fn quantile_score(&self, quantile: f64) -> f64 {
        self.unscale_value(self.histogram().value_at_percentile(quantile * 100.0) as f64)
    }

    /// Normalize a single score to a value between 0 and 1.
    pub fn normalize_score(&self, score: f64) -> f64 {
        let min = self.range.start();
        let max = self.range.end();
        (score - min) / (max - min)
    }

    /// Create a new value from a normalized score.
    pub fn denormalize_score(&self, score: f64) -> f64 {
        let min = self.range.start();
        let max = self.range.end();
        score * (max - min) + min
    }

    /// Normalize the EvaluationResult's score to a value between 0 and 1.
    pub fn normalized(self) -> Self {
        let mut normalized_values = self.tests;
        let min = self.range.start();
        let max = self.range.end();
        let range = max - min;
        for test in &mut normalized_values {
            test.score = (test.score - min) / range;
        }
        EvaluationResult {
            name: self.name,
            histogram: OnceLock::new(),
            tests: normalized_values,
            range: 0.0..=1.0,
        }
    }
}

impl<I: Display> std::fmt::Display for EvaluationResult<'_, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let histogram = self.histogram();

        let mut statistics = Table::new();
        statistics.set_header(vec!["Statistic", "Value"]);
        statistics.add_row(vec![
            Cell::new("Mean"),
            Cell::new(format!("{:.2}", self.mean_score())),
        ]);
        statistics.add_row(vec![
            Cell::new("Median"),
            Cell::new(format!("{:.2}", self.median_score())),
        ]);
        statistics.add_row(vec![
            Cell::new("Min"),
            Cell::new(format!("{:.2}", self.min_score())),
        ]);
        statistics.add_row(vec![
            Cell::new("Max"),
            Cell::new(format!("{:.2}", self.max_score())),
        ]);
        statistics.add_row(vec![
            Cell::new("25th Percentile"),
            Cell::new(format!("{:.2}", self.quantile_score(0.25))),
        ]);
        statistics.add_row(vec![
            Cell::new("75th Percentile"),
            Cell::new(format!("{:.2}", self.quantile_score(0.75))),
        ]);

        writeln!(f, "{statistics}")?;

        let mut table = Table::new();
        table.set_header(vec!["Expected Output", "Actual Output", "Score"]);

        let bottom_third_of_metric =
            self.range.start() + (self.range.end() - self.range.start()) / 3.0;
        let bottom_half_of_metric =
            self.range.start() + (self.range.end() - self.range.start()) / 2.0;

        fn create_cell(score: f64, quantile: f64) -> Cell {
            if quantile <= 0.1 {
                Cell::new(format!("{score:.2} (low outlier)"))
            } else if quantile <= 0.9 {
                Cell::new(format!("{score:.2}"))
            } else {
                Cell::new(format!("{score:.2} (high outlier)"))
            }
        }

        let buckets = [
            (comfy_table::Color::Red, bottom_third_of_metric),
            (comfy_table::Color::Yellow, bottom_half_of_metric),
            (comfy_table::Color::Green, f64::INFINITY),
        ];

        let mut test_iter = self.tests.iter().peekable();
        for (color, max) in buckets {
            let mut count = 0;
            while let Some(test) = test_iter.next_if(|test| test.score <= max) {
                let quantile =
                    histogram.percentile_below((test.score * Self::SCALE_FACTOR) as u64) / 100.0;

                let score_cell = create_cell(test.score, quantile).fg(color);
                let mut row = Row::new();
                row.add_cell(Cell::new(&test.case.expected))
                    .add_cell(Cell::new(&test.case.actual))
                    .add_cell(score_cell);
                table.add_row(row);
                count += 1;

                if count >= 5 {
                    let mut remaining_matching_tests = 0;
                    let mut total_score = 0.0;
                    for test in test_iter.by_ref() {
                        if test.score > max {
                            break;
                        }
                        total_score += test.score;
                        remaining_matching_tests += 1;
                    }
                    if remaining_matching_tests > 0 {
                        let mut row = Row::new();
                        row.add_cell(Cell::new(format!("... {remaining_matching_tests} more")))
                            .add_cell(Cell::new(""))
                            .add_cell(
                                Cell::new(format!(
                                    "{:.2} (average)",
                                    total_score / remaining_matching_tests as f64
                                ))
                                .fg(color),
                            );
                        table.add_row(row);
                    }
                    break;
                }
            }
        }

        writeln!(f, "{table}")?;

        let mut buckets = [0; 10];

        for test in &self.tests {
            let normalized_score = self.normalize_score(test.score);
            let bucket = (normalized_score * 10.0) as usize;
            buckets[bucket.min(9)] += 1;
        }

        let max_width = *buckets.iter().max().unwrap();

        // We need to scale the graph to fit in the terminal if it is too wide.
        let scale_factor = if max_width > 50 {
            50.0 / max_width as f64
        } else {
            1.0
        };

        let max_width = ((max_width as f64 * scale_factor) as usize).max(3);

        writeln!(f, "| Score Histogram {} |", " ".repeat(max_width - 3))?;

        for (i, bucket) in buckets.iter().enumerate() {
            let min_bucket = self.denormalize_score(i as f64 / 10.0);
            let max_bucket = self.denormalize_score((i + 1) as f64 / 10.0);
            let bucket = (*bucket as f64 * scale_factor) as usize;
            writeln!(
                f,
                "| {:.2} - {:.2}: {}{} |",
                min_bucket,
                max_bucket,
                "*".repeat(bucket),
                " ".repeat(max_width - bucket)
            )?;
        }

        Ok(())
    }
}

#[derive(Default, Clone)]
struct TestCase<I> {
    expected: I,
    actual: I,
}

#[derive(Clone)]
struct TestCaseScored<'a, I> {
    case: &'a TestCase<I>,
    score: f64,
}
