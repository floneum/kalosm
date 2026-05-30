use kalosm::language::*;

fn main() {
    pollster::block_on(async {
        let llm = Llama::new().await.unwrap();
        let prompt = "Five US states in central US are ";

        println!("# with constraints");
        print!("{prompt}");
        let states = [
            "Alabama",
            "Alaska",
            "Arizona",
            "Arkansas",
            "California",
            "Colorado",
            "Connecticut",
            "Delaware",
            "Florida",
            "Georgia",
            "Hawaii",
            "Idaho",
            "Illinois",
            "Indiana",
            "Iowa",
            "Kansas",
            "Kentucky",
            "Louisiana",
            "Maine",
            "Maryland",
            "Massachusetts",
            "Michigan",
            "Minnesota",
            "Mississippi",
            "Missouri",
            "Montana",
            "Nebraska",
            "Nevada",
            "New Hampshire",
            "New Jersey",
            "New Mexico",
            "New York",
            "North Carolina",
            "North Dakota",
            "Ohio",
            "Oklahoma",
            "Oregon",
            "Pennsylvania",
            "Rhode Island",
            "South Carolina",
            "South Dakota",
            "Tennessee",
            "Texas",
            "Utah",
            "Vermont",
            "Virginia",
            "Washington",
            "West Virginia",
            "Wisconsin",
            "Wyoming",
        ];
        let states_parser = states
            .into_iter()
            .map(LiteralParser::from)
            .collect::<Vec<_>>();

        let index_parser = IndexParser::new(states_parser);

        let validator = index_parser.then(LiteralParser::from(", ")).repeat(1..=5);
        let mut stream = llm(prompt).with_constraints(validator);

        stream.to_std_out().await.unwrap();

        println!(
            "\n{:#?}",
            stream
                .await
                .unwrap()
                .iter()
                .map(|x| states[x.0 .0])
                .collect::<Vec<_>>()
        );

        println!("\n\n# without constraints");
        print!("{prompt}");

        let mut stream = llm(prompt);
        stream.to_std_out().await.unwrap();
    });
}
