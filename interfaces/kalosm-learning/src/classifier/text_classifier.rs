use candle_core::Device;
use kalosm_language_model::{Embedder, EmbedderExt, Embedding};

use crate::{
    Class, ClassificationDataset, ClassificationDatasetBuilder, Classifier, ClassifierConfig,
    ClassifierOutput,
};

use super::ClassifierProgress;

/// A builder for [`TextClassifier`].
///
/// # Example
/// ```rust, no_run
/// # use kalosm_learning::*;
/// # use rbert::*;
/// # use std::collections::HashMap;
/// # #[derive(Debug, Copy, Clone, PartialEq, Eq, Class)]
/// # enum MyClass {
/// #     Person,
/// #     Thing,
/// # }
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// // Create a dataset for the classifier
/// let bert = Bert::new().await?;
/// let mut dataset = TextClassifierDatasetBuilder::<MyClass, _>::new(&bert);
/// for question in ["What is the author's name?", "What is the author's age?"] {
///     dataset.add(question, MyClass::Person).await?;
/// }
/// for question in [
///     "What is the capital of France?",
///     "What is the capital of England?",
/// ] {
///     dataset.add(question, MyClass::Thing).await?;
/// }
/// # Ok::<(), anyhow::Error>(())
/// # }
/// ```
pub struct TextClassifierDatasetBuilder<'a, T: Class, E: Embedder> {
    dataset: ClassificationDatasetBuilder<T>,
    embedder: &'a E,
}

impl<'a, T: Class, E: Embedder> TextClassifierDatasetBuilder<'a, T, E> {
    /// Creates a new [`TextClassifierDatasetBuilder`].
    pub fn new(embedder: &'a E) -> Self {
        Self {
            dataset: ClassificationDatasetBuilder::new(),
            embedder,
        }
    }

    /// Adds a new example to the dataset.
    pub async fn add(&mut self, text: impl ToString, class: T) -> Result<(), E::Error> {
        let embedding = self.embedder.embed(text).await?;
        self.dataset
            .add(embedding.vector().to_vec().into_boxed_slice(), class);
        Ok(())
    }

    /// Add many examples to the dataset at once. This may be faster than adding each example individually depending on the embedding model.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm_learning::*;
    /// # use rbert::*;
    /// # use std::collections::HashMap;
    /// # #[derive(Debug, Copy, Clone, PartialEq, Eq, Class)]
    /// # enum MyClass {
    /// #     Person,
    /// #     Thing,
    /// # }
    /// # #[tokio::main]
    /// # async fn main() -> anyhow::Result<()> {
    /// // Create a dataset for the classifier
    /// let bert = Bert::new().await?;
    /// let mut dataset = TextClassifierDatasetBuilder::<MyClass, _>::new(&bert);
    /// dataset
    ///     .extend(
    ///         ["What is the author's name?", "What is the author's age?"]
    ///             .into_iter()
    ///             .map(|q| (q, MyClass::Person)),
    ///     )
    ///     .await?;
    /// dataset
    ///     .extend(
    ///         [
    ///             "What is the capital of France?",
    ///             "What is the capital of England?",
    ///         ]
    ///         .into_iter()
    ///         .map(|q| (q, MyClass::Thing)),
    ///     )
    ///     .await?;
    /// # Ok::<(), anyhow::Error>(())
    /// # }
    /// ```
    pub async fn extend(
        &mut self,
        examples: impl IntoIterator<Item = (impl ToString, T)>,
    ) -> Result<(), E::Error> {
        let (texts, classes): (Vec<_>, Vec<_>) = examples.into_iter().unzip();
        let embeddings = self.embedder.embed_batch(texts).await?;
        for (embedding, class) in embeddings.into_iter().zip(classes) {
            self.dataset
                .add(embedding.vector().to_vec().into_boxed_slice(), class);
        }
        Ok(())
    }

    /// Builds the dataset.
    pub fn build(self, device: &Device) -> candle_core::Result<ClassificationDataset> {
        self.dataset.build(device)
    }
}

/// A text classifier.
///
/// # Example
///
/// ```rust, no_run
/// use candle_core::Device;
/// use kalosm_language_model::{Embedder, EmbedderExt};
/// use kalosm_learning::{
///     Class, Classifier, ClassifierConfig, TextClassifier, TextClassifierDatasetBuilder,
/// };
/// use rbert::Bert;
///
/// #[tokio::main]
/// async fn main() -> anyhow::Result<()> {
///     #[derive(Debug, Copy, Clone, PartialEq, Eq, Class)]
///     enum MyClass {
///         Person,
///         Thing,
///     }
///
///     let mut bert = Bert::builder().build().await?;
///
///     let dev = Device::cuda_if_available(0)?;
///     let person_questions = vec![
///         "What is the author's name?",
///         "What is the author's age?",
///         "Who is the queen of England?",
///         "Who is the president of the United States?",
///         "Who is the president of France?",
///         "Tell me about the CEO of Apple.",
///         "Who is the CEO of Google?",
///         "Who is the CEO of Microsoft?",
///         "What person invented the light bulb?",
///         "What person invented the telephone?",
///         "What is the name of the person who invented the light bulb?",
///         "Who wrote the book 'The Lord of the Rings'?",
///         "Who wrote the book 'The Hobbit'?",
///         "How old is the author of the book 'The Lord of the Rings'?",
///         "How old is the author of the book 'The Hobbit'?",
///         "Who is the best soccer player in the world?",
///         "Who is the best basketball player in the world?",
///         "Who is the best tennis player in the world?",
///         "Who is the best soccer player in the world right now?",
///         "Who is the leader of the United States?",
///         "Who is the leader of France?",
///         "What is the name of the leader of the United States?",
///         "What is the name of the leader of France?",
///     ];
///     let thing_sentences = vec![
///         "What is the capital of France?",
///         "What is the capital of England?",
///         "What is the name of the biggest city in the world?",
///         "What tool do you use to cut a tree?",
///         "What tool do you use to cut a piece of paper?",
///         "What is a good book to read?",
///         "What is a good movie to watch?",
///         "What is a good song to listen to?",
///         "What is the best tool to use to create a website?",
///         "What is the best tool to use to create a mobile app?",
///         "How long does it take to fly from Paris to New York?",
///         "How do you make a cake?",
///         "How do you make a pizza?",
///         "How can you make a website?",
///         "What is the best way to learn a new language?",
///         "What is the best way to learn a new programming language?",
///         "What is a framework?",
///         "What is a library?",
///         "What is a good way to learn a new language?",
///         "What is a good way to learn a new programming language?",
///         "What is the city with the most people in the world?",
///         "What is the most spoken language in the world?",
///         "What is the most spoken language in the United States?",
///     ];
///
///     let mut dataset = TextClassifierDatasetBuilder::<MyClass, _>::new(&mut bert);
///
///     for question in &person_questions {
///         dataset.add(question, MyClass::Person).await?;
///     }
///
///     for sentence in &thing_sentences {
///         dataset.add(sentence, MyClass::Thing).await?;
///     }
///
///     let dataset = dataset.build(&dev)?;
///
///     let mut classifier;
///     let layers = vec![5, 8, 5];
///
///     loop {
///         classifier = TextClassifier::<MyClass>::new(Classifier::new(
///             &dev,
///             ClassifierConfig::new().layers_dims(layers.clone()),
///         )?);
///         if let Err(error) = classifier.train(&dataset, 100, 0.05, 3, |_| {}) {
///             println!("Error: {:?}", error);
///         } else {
///             break;
///         }
///         println!("Retrying...");
///     }
///
///     let config = classifier.config();
///     classifier.save("classifier.safetensors")?;
///     let classifier = Classifier::<MyClass>::load("classifier.safetensors", &dev, config)?;
///
///     let tests = [
///         "Who is the president of Russia?",
///         "What is the capital of Russia?",
///         "Who invented the TV?",
///         "What is the best way to learn a how to ride a bike?",
///     ];
///
///     for test in &tests {
///         let input = bert.embed(test).await?;
///         let class = classifier.run(input.vector())?;
///         println!();
///         println!("{test}");
///         println!("{:?} {:?}", &input.vector()[..5], class);
///     }
///
///     Ok(())
/// }
/// ```
pub struct TextClassifier<T: Class> {
    model: Classifier<T>,
}

impl<T: Class> TextClassifier<T> {
    /// Creates a new [`TextClassifier`].
    pub fn new(model: Classifier<T>) -> Self {
        Self { model }
    }

    /// Runs the classifier on the given input.
    pub fn run(&self, input: Embedding) -> candle_core::Result<ClassifierOutput<T>> {
        self.model.run(input.vector())
    }

    /// Trains the classifier on the given dataset.
    pub fn train(
        &self,
        dataset: &ClassificationDataset,
        epochs: usize,
        learning_rate: f64,
        batch_size: usize,
        progress: impl FnMut(ClassifierProgress),
    ) -> candle_core::Result<f32> {
        self.model
            .train(dataset, epochs, learning_rate, batch_size, progress)
    }

    /// Get the configuration of the classifier.
    pub fn config(&self) -> ClassifierConfig {
        self.model.config()
    }

    /// Saves the classifier to the given path.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> candle_core::Result<()> {
        self.model.save(path)
    }

    /// Loads a classifier from the given path.
    pub fn load<P: AsRef<std::path::Path>>(
        path: P,
        device: &Device,
        config: ClassifierConfig,
    ) -> candle_core::Result<Self> {
        let model = Classifier::load(path, device, config)?;
        Ok(Self::new(model))
    }
}

#[cfg(test)]
#[test]
fn simplified() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(simplified_inner())
}

#[cfg(test)]
async fn simplified_inner() -> Result<(), Box<dyn std::error::Error>> {
    use crate::{Class, Classifier, ClassifierConfig};
    use rbert::{Bert, BertSource};

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Class)]
    enum MyClass {
        Person,
        Thing,
    }

    let bert = Bert::builder()
        .with_source(BertSource::snowflake_arctic_embed_extra_small())
        .build()
        .await?;

    let dev = kalosm_common::accelerated_device_if_available()?;
    let person_questions = [
        "What is the author's name?",
        "What is the author's age?",
        "Who is the queen of England?",
        "Who is the president of the United States?",
        "Who is the president of France?",
        "Tell me about the CEO of Apple.",
        "Who is the CEO of Google?",
        "Who is the CEO of Microsoft?",
        "What person invented the light bulb?",
        "What person invented the telephone?",
        "What is the name of the person who invented the light bulb?",
        "Who wrote the book 'The Lord of the Rings'?",
        "Who wrote the book 'The Hobbit'?",
        "How old is the author of the book 'The Lord of the Rings'?",
        "How old is the author of the book 'The Hobbit'?",
        "Who is the best soccer player in the world?",
        "Who is the best basketball player in the world?",
        "Who is the best tennis player in the world?",
        "Who is the best soccer player in the world right now?",
        "Who is the leader of the United States?",
        "Who is the leader of France?",
        "What is the name of the leader of the United States?",
        "What is the name of the leader of France?",
    ];
    let thing_sentences = [
        "What is the capital of France?",
        "What is the capital of England?",
        "What is the name of the biggest city in the world?",
        "What tool do you use to cut a tree?",
        "What tool do you use to cut a piece of paper?",
        "What is a good book to read?",
        "What is a good movie to watch?",
        "What is a good song to listen to?",
        "What is the best tool to use to create a website?",
        "What is the best tool to use to create a mobile app?",
        "How long does it take to fly from Paris to New York?",
        "How do you make a cake?",
        "How do you make a pizza?",
        "How can you make a website?",
        "What is the best way to learn a new language?",
        "What is the best way to learn a new programming language?",
        "What is a framework?",
        "What is a library?",
        "What is a good way to learn a new language?",
        "What is a good way to learn a new programming language?",
        "What is the city with the most people in the world?",
        "What is the most spoken language in the world?",
        "What is the most spoken language in the United States?",
    ];

    let mut dataset = TextClassifierDatasetBuilder::<MyClass, _>::new(&bert);

    for question in &person_questions {
        dataset.add(question, MyClass::Person).await?;
    }

    for sentence in &thing_sentences {
        dataset.add(sentence, MyClass::Thing).await?;
    }

    let dataset = dataset.build(&dev)?;

    let mut classifier;
    let layers = vec![5, 8, 5];

    loop {
        classifier = TextClassifier::<MyClass>::new(Classifier::new(
            &dev,
            ClassifierConfig::new().layers_dims(layers.clone()),
        )?);
        println!("Training...");
        if let Err(error) = classifier.train(&dataset, 100, 0.05, 100, |_| {}) {
            println!("Error: {error:?}");
        } else {
            break;
        }
        println!("Retrying...");
    }

    let config = classifier.model.config();
    classifier.save("classifier.safetensors")?;
    let classifier = Classifier::<MyClass>::load("classifier.safetensors", &dev, config)?;

    let tests = [
        "Who is the president of Russia?",
        "What is the capital of Russia?",
        "Who invented the TV?",
        "What is the best way to learn a how to ride a bike?",
    ];

    for test in &tests {
        let input = bert.embed(test).await?;
        let class = classifier.run(input.vector())?;
        println!();
        println!("{test}");
        println!("{:?} {:?}", &input.vector()[..5], class);
    }

    Ok(())
}
