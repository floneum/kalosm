use kalosm::language::*;
use std::sync::Arc;

fn main() {
    pollster::block_on(async {
        tracing_subscriber::fmt::init();

        let llm = Llama::new_chat().await.unwrap();

        let constraints =
            RegexParser::new(r"(Step \d: \d+ [+\-*/] \d+ = \d+\n){1,3}Output: \d+").unwrap();

        let task = llm.task("You are an assistant who solves math problems. When solving problems, you will always solve problems step by step with one step per line. Once you have solved the problem, you will output the result in the format 'Output: <result>'.")
            .with_example("What is 1 + 2?", "Step 1: 1 + 2 = 3\nOutput: 3")
            .with_example("What is 3 + 4?", "Step 1: 3 + 4 = 7\nOutput: 7")
            .with_example("What is (4 + 8) / 3?", "Step 1: 4 + 8 = 12\nStep 2: 12 / 3 = 4\nOutput: 4")
            .with_constraints(Arc::new(constraints));

        let start_timestamp = std::time::Instant::now();
        println!("question 1");
        // The first time we use the task, it will load the model and prompt.
        task(&"What is 2 + 2?").to_std_out().await.unwrap();
        println!("\nfirst question took: {:?}", start_timestamp.elapsed());

        let start_timestamp = std::time::Instant::now();
        println!("question 2");
        // After the first time, the model and prompt are cached.
        task(&"What is 4 + 4?").to_std_out().await.unwrap();
        println!("\nsecond question took: {:?}", start_timestamp.elapsed());

        let start_timestamp = std::time::Instant::now();
        println!("question 3");
        task(&"What is (7 + 5)/2?").to_std_out().await.unwrap();
        println!("\nthird question took: {:?}", start_timestamp.elapsed());
    });
}
