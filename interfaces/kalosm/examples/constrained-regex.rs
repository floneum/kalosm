use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        let llm = Llama::new().await.unwrap();
        let prompt = "Five prime numbers: 2, ";

        println!("# with constraints");
        print!("{prompt}");

        let validator = RegexParser::new(r"(\d, ){4}\d").unwrap();
        let mut stream = llm(prompt).with_constraints(validator);

        stream.to_std_out().await.unwrap();

        println!("\n\n# without constraints");
        print!("{prompt}");

        let mut stream = llm(prompt);
        stream.to_std_out().await.unwrap();
    });
}
