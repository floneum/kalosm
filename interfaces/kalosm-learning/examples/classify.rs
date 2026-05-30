use kalosm_common::accelerated_device_if_available;
use kalosm_language_model::EmbedderExt;
use kalosm_learning::{
    Class, Classifier, ClassifierConfig, ClassifierProgress, TextClassifier,
    TextClassifierDatasetBuilder,
};
use rbert::Bert;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Class)]
enum MyClass {
    Person,
    Thing,
}

const PEOPLE_QUESTIONS: &[&str] = &[
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
const THING_QUESTIONS: &[&str] = &[
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(async {
        // Create a new Bert model
        let bert = Bert::new().await?;

        // Create a dataset for the classifier
        let dev = accelerated_device_if_available()?;
        let mut dataset = TextClassifierDatasetBuilder::<MyClass, _>::new(&bert);
        for question in PEOPLE_QUESTIONS {
            dataset.add(question, MyClass::Person).await?;
        }
        for sentence in THING_QUESTIONS {
            dataset.add(sentence, MyClass::Thing).await?;
        }
        let dataset = dataset.build(&dev)?;

        let classifier =
            TextClassifier::<MyClass>::new(Classifier::new(&dev, ClassifierConfig::new())?);

        classifier.train(
            &dataset, // The dataset to train on
            10,       // The number of epochs to train for
            0.003,    // The learning rate
            5,        // The batch size
            |progress| match progress {
                ClassifierProgress::EpochFinished { epoch, accuracy } => {
                    println!("Epoch {epoch} accuracy: {accuracy}");
                }
                ClassifierProgress::BatchFinished { batch, loss } => {
                    println!("Batch {batch} loss: {loss}");
                }
            },
        )?;

        let config = classifier.config();
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
            let input = input.vector();
            let class = classifier.run(input)?;
            println!();
            println!("{test}");
            println!("{:?} {:?}", &input[..5], class);
        }

        Ok(())
    })
}
