use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        let description = prompt_input("What is your character like? ").unwrap();
        let character_name = prompt_input("What is your character's name? ").unwrap();
        let character_description =
            format!("{description} The assistant will never reveal that it is an AI or assistant.");

        let model = Llama::new_chat().await.unwrap();
        // Create constraints that parses (Responding as <character_name>) and then stops on the end of the assistant's response
        let constraints = LiteralParser::new(format!("(Responding as {character_name}) "))
            .then(model.default_assistant_constraints());
        // Create a chat session with the model and the constraints
        let mut chat = model.chat().with_system_prompt(character_description);

        // Chat with the user
        loop {
            let mut output_stream =
                chat(&prompt_input("\n> ").unwrap()).with_constraints(constraints.clone());
            print!("Bot: ");
            output_stream.to_std_out().await.unwrap();
        }
    });
}
