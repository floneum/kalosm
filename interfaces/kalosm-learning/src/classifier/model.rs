use rand::prelude::SliceRandom;
use std::{collections::HashMap, sync::OnceLock};

use candle_core::{
    safetensors::{self, Load},
    DType, Device, Result, Tensor, Var, D,
};
use candle_nn::{loss, ops, Dropout, Linear, Module, ModuleT, Optimizer, VarBuilder, VarMap};
use kalosm_common::maybe_autoreleasepool;
use rand::Rng;

/// A class that a [`Classifier`] can predict. You can derive this trait for any enum with only unit values.
///
/// # Example
/// ```rust
/// # use kalosm_learning::*;
/// #[derive(Debug, Clone, Copy, Class)]
/// enum MyClass {
///     Person,
///     Thing,
/// }
/// ```
pub trait Class {
    /// The number of classes.
    const CLASSES: Option<u32>;

    /// Convert the class to a class index.
    fn to_class(&self) -> u32;
    /// Convert a class index to a class.
    fn from_class(class: u32) -> Self;
}

impl Class for u32 {
    const CLASSES: Option<u32> = None;

    fn to_class(&self) -> u32 {
        *self
    }

    fn from_class(class: u32) -> Self {
        class
    }
}

/// A dataset to train a [`Classifier`].
#[derive(Clone, Debug)]
pub struct ClassificationDataset {
    train_inputs: Tensor,
    train_classes: Tensor,
    test_inputs: Tensor,
    test_classes: Tensor,
}

impl ClassificationDataset {
    /// Create a builder for a classification dataset.
    pub fn builder<C: Class>() -> ClassificationDatasetBuilder<C> {
        ClassificationDatasetBuilder::default()
    }

    /// Save the dataset to the given path.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm_learning::*;
    /// let dev = candle_core::Device::Cpu;
    /// let dataset = ClassificationDataset::load("dataset.safetensors", &dev).unwrap();
    /// dataset.save("dataset_copy.safetensors").unwrap();
    /// ```
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let safetensors = HashMap::from([
            ("train_inputs".to_string(), self.train_inputs.clone()),
            ("train_classes".to_string(), self.train_classes.clone()),
            ("test_inputs".to_string(), self.test_inputs.clone()),
            ("test_classes".to_string(), self.test_classes.clone()),
        ]);

        safetensors::save(&safetensors, path)?;
        Ok(())
    }

    /// Load the dataset from the given path.
    ///
    /// # Example
    /// ```rust, no_run
    /// # use kalosm_learning::*;
    /// let dev = candle_core::Device::Cpu;
    /// let dataset = ClassificationDataset::load("dataset.safetensors", &dev).unwrap();
    /// ```
    pub fn load<P: AsRef<std::path::Path>>(path: P, dev: &Device) -> Result<Self> {
        let mut safetensors = safetensors::load(path, dev)?;
        Ok(Self {
            train_inputs: safetensors.remove("train_inputs").unwrap(),
            train_classes: safetensors.remove("train_classes").unwrap(),
            test_inputs: safetensors.remove("test_inputs").unwrap(),
            test_classes: safetensors.remove("test_classes").unwrap(),
        })
    }
}

/// A builder for [`ClassificationDataset`].
pub struct ClassificationDatasetBuilder<C: Class> {
    input_size: Option<usize>,
    inputs: Vec<Box<[f32]>>,
    classes: Vec<C>,
}

impl<C: Class> Default for ClassificationDatasetBuilder<C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<C: Class> ClassificationDatasetBuilder<C> {
    /// Create a new dataset builder.
    pub fn new() -> Self {
        Self {
            input_size: None,
            inputs: Vec::new(),
            classes: Vec::new(),
        }
    }

    /// Adds a pair of input and class to the dataset.
    ///
    /// # Example
    /// ```rust
    /// # use kalosm_learning::*;
    /// # #[derive(Debug, Clone, Copy, Class)]
    /// # enum MyClass {
    /// #     Person,
    /// #     Thing,
    /// # }
    /// let mut dataset = ClassificationDatasetBuilder::new();
    /// dataset.add(vec![1.0, 2.0, 3.0, 4.0], MyClass::Person);
    /// dataset.add(vec![4.0, 3.0, 2.0, 1.0], MyClass::Thing);
    /// ```
    pub fn add(&mut self, input: impl Into<Box<[f32]>>, class: C) {
        let input = input.into();
        if let Some(input_size) = self.input_size {
            debug_assert_eq!(input.len(), input_size, "input size mismatch");
        } else {
            self.input_size = Some(input.len());
        }
        self.inputs.push(input);
        self.classes.push(class);
    }

    /// Builds the dataset and copies the data to the device passed in.
    ///
    /// # Example
    /// ```rust
    /// # use kalosm_learning::*;
    /// # #[derive(Debug, Clone, Copy, Class)]
    /// # enum MyClass {
    /// #     Person,
    /// #     Thing,
    /// # }
    /// let dev = candle_core::Device::Cpu;
    /// let mut dataset = ClassificationDatasetBuilder::new();
    /// dataset.add(vec![1.0, 2.0, 3.0, 4.0], MyClass::Person);
    /// dataset.add(vec![4.0, 3.0, 2.0, 1.0], MyClass::Thing);
    /// let dataset = dataset.build(&dev).unwrap();
    /// ```
    pub fn build(mut self, dev: &Device) -> Result<ClassificationDataset> {
        // split into train and test
        let mut rng = rand::rng();

        // We want to try to maintain a balance of classes in the test and train sets
        let mut class_counts: HashMap<u32, usize> = HashMap::new();
        for class in &self.classes {
            *class_counts.entry(class.to_class()).or_default() += 1;
        }

        // The test classes are 1/4 of the train classes
        let test_class_counts_goal = class_counts
            .iter()
            .map(|(class, count)| (*class, 1.max(*count / 4)))
            .collect::<HashMap<_, _>>();
        let mut test_class_counts = HashMap::new();

        let test_len = test_class_counts_goal.values().copied().sum::<usize>();
        let mut input_test: Vec<f32> = Vec::with_capacity(test_len);
        let mut class_test: Vec<u32> = Vec::with_capacity(test_len);
        let mut inputs: Vec<f32> = Vec::with_capacity(self.inputs.len() - test_len);
        let mut classes: Vec<u32> = Vec::with_capacity(self.classes.len() - test_len);
        while !self.classes.is_empty() {
            let index = rng.random_range(0..self.classes.len());
            let input = self.inputs.remove(index);
            let class = self.classes.remove(index);

            let class_u32 = class.to_class();
            let test_class_counts_goal = test_class_counts_goal.get(&class_u32).unwrap();
            let test_class_count: &mut usize = test_class_counts.entry(class_u32).or_default();
            if *test_class_count < *test_class_counts_goal {
                input_test.extend_from_slice(&input);
                class_test.push(class_u32);
                *test_class_count += 1;
            } else {
                inputs.extend_from_slice(&input);
                classes.push(class_u32);
            }
        }

        println!("{} train/{} tests", classes.len(), class_test.len());

        let input_size = self.input_size.unwrap_or_default();

        // train
        let train_inputs_len = inputs.len() / input_size;
        let train_inputs = Tensor::from_vec(inputs, (train_inputs_len, input_size), dev)?;
        let train_classes_len = classes.len();
        debug_assert_eq!(train_inputs_len, train_classes_len);
        let train_classes = Tensor::from_vec(classes, train_classes_len, dev)?;

        // test
        let test_inputs_len = input_test.len() / input_size;
        let test_inputs = Tensor::from_vec(input_test, (test_inputs_len, input_size), dev)?;
        let test_classes_len = class_test.len();
        debug_assert_eq!(test_inputs_len, test_classes_len);
        let test_classes = Tensor::from_vec(class_test, test_classes_len, dev)?;

        Ok(ClassificationDataset {
            train_inputs,
            train_classes,
            test_inputs,
            test_classes,
        })
    }
}

/// A classifier.
pub struct Classifier<C: Class> {
    device: Device,
    varmap: VarMap,
    layers_dims: Vec<usize>,
    layers: OnceLock<Vec<Linear>>,
    dropout: Dropout,
    dropout_rate: f32,
    classes: u32,
    phantom: std::marker::PhantomData<C>,
}

impl<C: Class> Classifier<C> {
    /// Create a new classifier.
    ///
    /// # Example
    /// ```rust
    /// use kalosm_learning::{Class, Classifier, ClassifierConfig};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let classifier = Classifier::<MyClass>::new(&dev, ClassifierConfig::new()).unwrap();
    /// ```
    pub fn new(dev: &Device, config: ClassifierConfig) -> Result<Self> {
        let varmap = VarMap::new();
        Self::new_inner(dev.clone(), varmap, config)
    }

    /// Get the config of the classifier.
    pub fn config(&self) -> ClassifierConfig {
        ClassifierConfig {
            layers_dims: self.layers_dims.clone(),
            dropout_rate: self.dropout_rate,
            classes: Some(self.classes),
        }
    }

    fn new_inner(dev: Device, varmap: VarMap, config: ClassifierConfig) -> Result<Self> {
        let ClassifierConfig {
            layers_dims,
            dropout_rate,
            classes,
        } = config;
        Ok(Self {
            device: dev,
            layers: OnceLock::new(),
            layers_dims,
            varmap,
            dropout: Dropout::new(dropout_rate),
            dropout_rate,
            classes: classes.or(C::CLASSES).ok_or_else(|| {
                candle_core::Error::Msg("No number of classes specified for classifier".to_string())
            })?,
            phantom: std::marker::PhantomData,
        })
    }

    fn layers(&self, input_dim: usize) -> candle_core::Result<&Vec<Linear>> {
        if let Some(layers) = self.layers.get() {
            return Ok(layers);
        }
        let vs = VarBuilder::from_varmap(&self.varmap, DType::F32, &self.device);
        let output_dim = self.classes;
        let mut layers = Vec::with_capacity(self.layers_dims.len() + 1);
        if self.layers_dims.is_empty() {
            let layer = candle_nn::linear(input_dim, output_dim as usize, vs.pp("ln0"))?;
            layers.push(layer);
        } else {
            layers.push(candle_nn::linear(
                input_dim,
                *self.layers_dims.first().unwrap(),
                vs.pp("ln0"),
            )?);
            for (i, (in_dim, out_dim)) in self
                .layers_dims
                .iter()
                .zip(self.layers_dims.iter().skip(1))
                .enumerate()
            {
                layers.push(candle_nn::linear(
                    *in_dim,
                    *out_dim,
                    vs.pp(format!("ln{}", i + 1)),
                )?);
            }
            layers.push(candle_nn::linear(
                *self.layers_dims.last().unwrap(),
                output_dim as usize,
                vs.pp(format!("ln{}", self.layers_dims.len() + 1)),
            )?);
        }

        Ok(self.layers.get_or_init(|| layers))
    }

    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        let mut xs = xs.clone();
        let input_dim = *xs.dims().last().unwrap();
        for layer in self.layers(input_dim)? {
            xs = self.dropout.forward_t(&xs, train)?;
            xs = layer.forward(&xs)?;
            xs = xs.gelu_erf()?;
        }
        Ok(xs)
    }

    /// Train the model on the given dataset.
    ///
    /// # Example
    /// ```rust, no_run
    /// use kalosm_learning::{Class, ClassificationDatasetBuilder, Classifier, ClassifierConfig};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let classifier = Classifier::<MyClass>::new(&dev, ClassifierConfig::new()).unwrap();
    /// let mut dataset = ClassificationDatasetBuilder::new();
    /// dataset.add(vec![1.0, 2.0, 3.0, 4.0], MyClass::Person);
    /// dataset.add(vec![4.0, 3.0, 2.0, 1.0], MyClass::Thing);
    ///
    /// classifier
    ///     .train(&dataset.build(&dev).unwrap(), 20, 0.05, 3, |_| {})
    ///     .unwrap();
    /// ```
    pub fn train(
        &self,
        m: &ClassificationDataset,
        epochs: usize,
        learning_rate: f64,
        batch_size: usize,
        mut progress: impl FnMut(ClassifierProgress),
    ) -> Result<f32> {
        // unstack both tensors into a list of tensors
        let train_len = m.train_inputs.dims()[0];
        let train_results = m.train_classes.chunk(train_len, 0)?;
        let train_votes = m.train_inputs.chunk(train_len, 0)?;

        // Force the layers to be initialized before we use the varmap
        self.forward_t(&train_votes[0].to_device(&self.device)?, true)?;

        let mut sgd = candle_nn::AdamW::new_lr(self.varmap.all_vars(), learning_rate)?;
        let test_votes = m.test_inputs.to_device(&self.device)?;
        let test_results = m.test_classes.to_device(&self.device)?;
        let mut final_accuracy: f32 = 0.0;
        let mut rng = rand::rng();
        let mut batch = 0;
        for epoch in 1..epochs + 1 {
            // create a random batch of indices
            let mut indices = (0..train_len).collect::<Vec<_>>();
            indices.shuffle(&mut rng);
            maybe_autoreleasepool(|| {
                for indices in indices.chunks(batch_size) {
                    let train_results = Tensor::cat(
                        &indices
                            .iter()
                            .copied()
                            .map(|i| train_results[i].clone())
                            .collect::<Vec<_>>(),
                        0,
                    )?
                    .to_device(&self.device)?;
                    let train_votes = Tensor::cat(
                        &indices
                            .iter()
                            .copied()
                            .map(|i| train_votes[i].clone())
                            .collect::<Vec<_>>(),
                        0,
                    )?
                    .to_device(&self.device)?;

                    let logits = self.forward_t(&train_votes, true)?;
                    let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                    let loss = loss::nll(&log_sm, &train_results)?;
                    sgd.backward_step(&loss)?;
                    progress(ClassifierProgress::BatchFinished {
                        batch,
                        loss: loss.to_scalar::<f32>()?,
                    });
                    batch += 1;
                }
                let test_logits = self.forward_t(&test_votes, false)?;
                let test_cases_passed = test_logits
                    .argmax(D::Minus1)?
                    .eq(&test_results)?
                    .to_dtype(DType::U32)?
                    .sum_all()?
                    .to_scalar::<u32>()?;
                let test_cases = test_results.dims1()?;
                let test_accuracy: f32 = test_cases_passed as f32 / test_cases as f32;
                final_accuracy = f32::from(100u8) * test_accuracy;
                progress(ClassifierProgress::EpochFinished {
                    epoch,
                    accuracy: test_accuracy,
                });
                println!(
                    "Epoch: {epoch:5} Test accuracy: {final_accuracy:5.5}% ({test_cases_passed}/{test_cases})",
                );
                Ok::<_, candle_core::Error>(())
            })?;
        }
        Ok(final_accuracy)
    }

    /// Save the model to a safetensors file at the given path.
    ///
    /// # Example
    ///
    /// ```rust, no_run
    /// use kalosm_learning::{Class, Classifier, ClassifierConfig};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let classifier = Classifier::<MyClass>::new(&dev, ClassifierConfig::new()).unwrap();
    /// classifier.save("classifier.safetensors").unwrap();
    /// ```
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.varmap.save(path)
    }

    /// Load the model from a safetensors file at the given path.
    ///
    /// # Example
    ///
    /// ```rust, no_run
    /// use kalosm_learning::{Class, Classifier, ClassifierConfig};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let classifier =
    ///     Classifier::<MyClass>::load("classifier.safetensors", &dev, ClassifierConfig::new())
    ///         .unwrap();
    /// ```
    pub fn load(
        path: impl AsRef<std::path::Path>,
        dev: &Device,
        config: ClassifierConfig,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        {
            let safetensors = unsafe { candle_core::safetensors::MmapedSafetensors::new(path) }?;
            let tensors = safetensors.tensors();
            let mut tensor_data = varmap.data().lock().unwrap();
            for (name, value) in tensors {
                let tensor = value.load(dev)?;
                tensor_data.insert(name.to_string(), Var::from_tensor(&tensor)?);
            }
        }
        Self::new_inner(dev.clone(), varmap, config)
    }

    /// Run the model on the given input.
    ///
    /// # Example
    ///
    /// ```rust, no_run
    /// use kalosm_learning::{Class, Classifier, ClassifierConfig};
    ///
    /// #[derive(Debug, Clone, Copy, Class)]
    /// enum MyClass {
    ///     Person,
    ///     Thing,
    /// }
    ///
    /// let dev = candle_core::Device::Cpu;
    /// let classifier = Classifier::<MyClass>::new(&dev, ClassifierConfig::new()).unwrap();
    /// let result = classifier.run(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    /// println!("Result: {:?}", result);
    /// ```
    pub fn run(&self, input: &[f32]) -> Result<ClassifierOutput<C>> {
        let input = Tensor::from_vec(input.to_vec(), (1, input.len()), &self.device)?;
        let logits = self.forward_t(&input, false)?;
        let classes = logits.flatten_all()?;
        let classes = ops::softmax(&classes, D::Minus1)?;
        let classes = classes.to_vec1()?;
        Ok(ClassifierOutput {
            classes: classes
                .into_iter()
                .enumerate()
                .map(|(i, c)| (C::from_class(i as u32), c))
                .collect(),
        })
    }
}

/// The output of a classifier.
#[derive(Debug, Clone)]
pub struct ClassifierOutput<C: Class> {
    /// The classes along with their probabilities.
    classes: Box<[(C, f32)]>,
}

impl<C: Class> ClassifierOutput<C> {
    /// Get the probabilities of each class.
    pub fn classes(&self) -> &[(C, f32)] {
        &self.classes
    }

    /// Get the top class with the highest probability.
    pub fn top(&self) -> C
    where
        C: Clone,
    {
        self.classes
            .iter()
            .max_by(|(_, a), (_, b)| f32::total_cmp(a, b))
            .map(|(c, _)| c.clone())
            .unwrap()
    }
}

/// Progress of training a classifier.
#[derive(Debug, Clone, Copy)]
pub enum ClassifierProgress {
    /// Progress after an epoch has finished.
    EpochFinished {
        /// The current epoch.
        epoch: usize,
        /// The test accuracy of the current epoch.
        accuracy: f32,
    },
    /// Progress after a batch has finished.
    BatchFinished {
        /// The current batch.
        batch: usize,
        /// The current loss.
        loss: f32,
    },
}

#[derive(Debug, Clone)]
/// A config for a [`Classifier`].
pub struct ClassifierConfig {
    /// The dimensions of the layers.
    layers_dims: Vec<usize>,
    /// The dropout rate.
    dropout_rate: f32,
    /// The number of classes.
    classes: Option<u32>,
}

impl Default for ClassifierConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl ClassifierConfig {
    /// Create a new config.
    pub fn new() -> Self {
        Self {
            layers_dims: vec![4, 8, 4],
            dropout_rate: 0.1,
            classes: None,
        }
    }

    /// Set the dimensions of the layers.
    pub fn layers_dims(mut self, layers_dims: impl IntoIterator<Item = usize>) -> Self {
        self.layers_dims = layers_dims.into_iter().collect();
        self
    }

    /// Set the dropout rate.
    pub fn dropout_rate(mut self, dropout_rate: f32) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }

    /// Set the number of classes. This is required if [`Class::CLASSES`] is not defined for the type you are classifying.
    pub fn classes(mut self, classes: u32) -> Self {
        self.classes = Some(classes);
        self
    }
}
