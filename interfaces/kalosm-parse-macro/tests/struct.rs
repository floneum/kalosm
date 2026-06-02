#![allow(unused)]

use kalosm::language::{kalosm_sample, Parse, Schema};
use pretty_assertions::assert_eq;

#[derive(Parse, Schema, Clone, PartialEq, Debug)]
#[parse(rename = "empty struct")]
struct EmptyNamedStruct {}

#[test]
fn empty_struct_schema() {
    let schema = EmptyNamedStruct::schema();
    let json = serde_json::from_str::<serde_json::Value>(&schema.to_string()).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "enum": ["empty struct"]
        })
    )
}

/// A named struct
#[derive(Parse, Schema, Clone)]
struct NamedStruct {
    /// The name of the person
    #[parse(rename = "field name")]
    name: String,
    /// The age of the person
    age: u32,
}

#[test]
fn named_struct_schema() {
    let schema = NamedStruct::schema();
    let json = serde_json::from_str::<serde_json::Value>(&schema.to_string()).unwrap();
    assert_eq!(
        json,
        serde_json::json!({
            "title": "NamedStruct",
            "description": "A named struct",
            "type": "object",
            "properties": {
                "field name": {
                    "description": "The name of the person",
                    "type": "string"
                },
                "age": {
                    "description": "The age of the person",
                    "type": "integer"
                }
            },
            "required": [
                "field name",
                "age"
            ],
            "additionalProperties": false
        })
    );
}

#[derive(Parse, Schema, Clone)]
struct WithStruct {
    #[parse(with = kalosm_sample::StringParser::new(1..=10))]
    name: String,
    #[parse(rename = "field name")]
    age: u32,
}
