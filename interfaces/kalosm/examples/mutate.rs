use kalosm::{language::*, BertDistance, TestCases};
use kalosm_language::search::Hypothetical;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let llm = Llama::new_chat().await.unwrap();

    let mutator = Mutator::builder(llm.clone()).build().unwrap();

    let mut text = "You generate isolated hypothetical questions with any necessary information that may be answered by the given text".to_string();
    let mut max_score = eval_with_prompt(llm.clone(), &text).await.unwrap();
    loop {
        let new_text = mutator.mutate(&text).await.unwrap();
        let new_score = eval_with_prompt(llm.clone(), &new_text).await.unwrap();
        if new_score > max_score {
            println!("new prompt was better: {new_text}");
            println!("new max score: {new_score}");
            max_score = new_score;
            text = new_text;
        }
    }
}

const TASK_DESCRIPTION: &str =
    "You generate variants of instructions that are clear, concise, and easy to follow.";

const EXAMPLES: [(&str, &str); 4] = [
    (
        "Write an essay about Floneum.",
        "Write an paper about Floneum in 500 words.",
    ),
    (
        "Calculate the area of a circle.",
        "Think step by step how to calculate the area of a circle.",
    ),
    (
        "Write a fibonacci function.",
        "Write a function that calculates the fibonacci sequence.",
    ),
    (
        "What is the best and most popular capital of France.",
        "What is the capital of France.",
    ),
];

const PREFIX: &str = "A variant of the previous task: ";

type Constraints = kalosm_sample::SequenceParser<LiteralParser, StopOn<&'static str>>;

fn create_constraints() -> Constraints {
    LiteralParser::new(PREFIX).then(
        StopOn::new(".").filter_characters(
            |c| matches!(c, ' ' | '.' | ':' | '/' | '+' | '-' | '*' | 'a'..='z' | 'A'..='Z' | '0'..='9' | ','),
        ),
    )
}

/// A builder for a Mutator chunker.
pub struct MutatorBuilder {
    model: Llama,
    task_description: Option<String>,
    examples: Option<Vec<(String, String)>>,
}

impl MutatorBuilder {
    /// Set the examples for this task. Each example should include the text and the questions that are answered by the text.
    pub fn with_examples<S: Into<String>>(
        mut self,
        examples: impl IntoIterator<Item = (S, S)>,
    ) -> MutatorBuilder {
        self.examples = Some(
            examples
                .into_iter()
                .map(|(a, b)| (a.into(), b.into()))
                .collect::<Vec<_>>(),
        );
        self
    }

    /// Set the task description. The task description should describe a task of generating Mutator questions that may be answered by the given text.
    pub fn with_task_description(mut self, task_description: String) -> Self {
        self.task_description = Some(task_description);
        self
    }

    /// Build the Mutator chunker.
    pub fn build(self) -> anyhow::Result<Mutator> {
        let task_description = self
            .task_description
            .unwrap_or_else(|| TASK_DESCRIPTION.to_string());
        let examples = self.examples.unwrap_or_else(|| {
            EXAMPLES
                .iter()
                .map(|(a, b)| (a.to_string(), { PREFIX.to_string() + b }))
                .collect::<Vec<_>>()
        });

        let task = self.model.task(task_description).with_examples(examples);

        Ok(Mutator { task })
    }
}

/// Generates embeddings of questions
pub struct Mutator {
    task: Task<Llama>,
}

impl Mutator {
    /// Create a new Mutator chunker.
    pub fn builder(model: Llama) -> MutatorBuilder {
        MutatorBuilder {
            model,
            task_description: None,
            examples: None,
        }
    }

    /// Generate a list of Mutator questions about the given text.
    pub async fn mutate(&self, text: &str) -> anyhow::Result<String> {
        let questions = self
            .task
            .run(text)
            .with_constraints(create_constraints())
            .await?;
        let documents = questions.1;

        Ok(documents)
    }
}

const TEST_PAIRS :&[(&str, &str)]= &[
    ("Cryptocurrencies like Bitcoin operate on a decentralized network using blockchain technology. This decentralized nature eliminates the need for a central authority, providing users with increased security and censorship resistance.", "How does blockchain contribute to the security of cryptocurrencies?"),
    ("Machine learning models require training on labeled datasets to make predictions. Supervised learning, a common approach, involves providing the model with input-output pairs to learn the underlying patterns and relationships.", "What is supervised learning in machine learning?"),
    ("The Internet of Things or IoT connects everyday devices to the internet, enabling them to send and receive data. This connectivity enhances automation and allows for more efficient monitoring and control of various systems.", "What is the purpose of the Internet of Things?"),
    ("Cybersecurity measures, such as firewalls and encryption, play a crucial role in protecting computer systems from unauthorized access and data breaches. These measures help ensure the confidentiality and integrity of sensitive information.", "How do firewalls contribute to cybersecurity?"),
    ("WebAssembly or WASM is a binary instruction format designed for efficient execution on web browsers. It enables running code written in languages like C++ and Rust directly in the browser, expanding the possibilities of web applications.", "What is the purpose of WebAssembly?"),
    ("Quantum computing leverages the principles of quantum mechanics to perform computations at speeds unattainable by classical computers. This emerging technology holds the potential to solve complex problems, such as factorizing large numbers, with unprecedented efficiency.", "What sets quantum computing apart from classical computing?"),
    ("Docker containers provide a lightweight and portable way to package and deploy applications, along with their dependencies. This approach streamlines the deployment process and ensures consistency across different environments.", "What is the purpose of Docker containers in application deployment?"),
    ("Neural networks, inspired by the human brain, form the backbone of deep learning. These interconnected layers of nodes learn to recognize patterns and make predictions, enabling tasks like image recognition and natural language processing.", "What is the role of neural networks in deep learning?"),
    ("Floneum is a user-friendly editor for visual AI workflows. Unlike existing tools that may have a high barrier to entry or allow limited control, Floneum provides a solution that is both easy to use and allows for greater customization.", "What is Floneum?"),
    ("For instance, while the chat GPT interface provides a straightforward entry point, it quickly becomes challenging to create structured workflows. Imagine wanting to search through files to find specific ones, such as all .txt files related to travel, and then upload them. With Floneum, you can achieve this seamlessly within a structured workflow, eliminating the need for manual interaction with external tools.", "What are the tradeoffs of using chat GPT?"),
    ("On the other end of the spectrum, tools like Langchain offer extensive workflow customization but come with more system requirements and potential security concerns. Langchain requires users to install tools like Python and CUDA, making it less accessible to non-developers. In addition to this, building workflows in Python code can be impractical for individuals without programming expertise. Finally, plugins in Langchain are not sandboxed, which can expose users to malware or security risks when incorporating third-party libraries.", "What are the tradeoffs of using Langchain?"),
    ("Floneum is a single executable that runs models locally, eliminating the need for complex installations. The heart of Floneum is its graph-based editor, designed to enable users without programming knowledge to build and manage their AI workflows seamlessly.", "What is Floneum?"),
    ("Embeddings are a way to understand the meaning of text. They provide a representation of the meaning of the words used. It lets us focus on the meaning of the text instead of the specific wording of the text.", "What is an embedding?"),
    ("In the world of programming languages, Rust stands out for its focus on memory safety without sacrificing performance. Its ownership system ensures that memory is managed efficiently, preventing common pitfalls like null pointer dereferencing.", "What is Rust known for?"),
    ("While traditional databases rely on a fixed schema, NoSQL databases like MongoDB offer a flexible structure, allowing you to store and retrieve data in a more dynamic way. This flexibility is particularly beneficial for applications with evolving data requirements.", "How does MongoDB differ from traditional databases?"),
    ("Floneum is designed to support an expanding ecosystem of plugins. In the future, additional plugins will be added to enhance its functionality further. Furthermore, if the built-in plugins don't precisely fit your application, Floneum allows you to extend its capabilities with plugins that are fully sandboxed within their own environment. Through the utilization of a WebAssembly (WASM) compiler, plugins can only access resources within their designated sandbox. This ensures that you can trust Floneum to prevent any malicious activity from impacting your computer.", "What are Floneum plugins?"),
    ("The agile software development methodology emphasizes iterative and collaborative approaches to project management. This flexible framework allows teams to adapt to changing requirements and deliver incremental improvements.", "What are the key principles of agile software development?"),
    ("The concept of virtualization involves creating virtual instances of resources like servers or networks. Virtualization enables better resource utilization, scalability, and isolation in computing environments.", "What is the purpose of virtualization in computer science?"),
    ("Open-source software, like the Linux operating system, is freely accessible and allows users to view, modify, and distribute its source code. This collaborative approach fosters innovation and community-driven development.", "What distinguishes open-source software from proprietary software?"),
    ("A content delivery network or a CDN optimizes the distribution of web content by strategically placing servers worldwide. This reduces latency, accelerates content delivery, and enhances the overall user experience.", "What role does a content delivery network play in web performance?"),
    ("Biometric authentication methods, such as fingerprint recognition and facial recognition, use unique biological characteristics for identity verification. This enhances security by providing a more robust and personalized authentication process.", "How do biometric authentication methods contribute to security?"),
    ("The concept of DevOps integrates software development and IT operations to improve collaboration and productivity. Automation tools play a crucial role in achieving continuous integration and continuous delivery in DevOps practices.", "What is the relationship between DevOps, continuous integration and continuous delivery?"),
    ("Natural Language Processing or NLP involves the interaction between computers and human language. Applications of NLP include language translation, sentiment analysis, and chatbot development.", "What are some applications of Natural Language Processing?"),
    ("The concept of edge computing involves processing data closer to the source rather than relying on a centralized cloud infrastructure. This approach reduces latency and enhances real-time processing in distributed systems.", "What is the significance of edge computing in data processing?"),
    ("APIs facilitate communication between different software systems. They define the methods and data formats applications can use to request and exchange information.", "What is the purpose of APIs in software development?"),
    ("Blockchain technology, beyond cryptocurrencies, is being explored for applications like smart contracts. Smart contracts are self-executing contracts with the terms of the agreement directly written into code.", "How is blockchain technology utilized in the concept of smart contracts?")
];

async fn eval_with_prompt(llm: Llama, prompt: &str) -> anyhow::Result<f64> {
    println!("evaluating prompt: {prompt}");

    let hypothetical = Hypothetical::builder(llm)
        .with_task_description(prompt)
        .build();

    let mut llama_test_cases = TestCases::new();

    for (text, expected) in TEST_PAIRS {
        let actual = &hypothetical.generate_question(text).await.unwrap()[0];

        llama_test_cases.push_case(expected.to_string(), actual.clone());
    }

    let mut bert_distance = BertDistance::new(Bert::new().await.unwrap());
    let llama_distance = llama_test_cases
        .evaluate(&mut bert_distance)
        .await
        .normalized();

    println!("llama_distance: {:#?}", llama_distance.mean_score());

    Ok(llama_distance.mean_score())
}
